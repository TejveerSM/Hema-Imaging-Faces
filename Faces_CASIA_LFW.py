from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CASIA dataset for training the network
csv_file = 'CASIA-WebFace/casia_data0.csv'
image_dir = '/home/tejveer/CASIA-WebFace/'

# LFW dataset for testing the performance
csv_file_same = 'Data_pairsDevTrain_match.csv'
csv_file_diff = 'Data_pairsDevTrain_mismatch.csv'
image_dir_test = '/RawData/LFW/lfw2/lfw2/'

data_train = pd.read_csv(csv_file, header=None)
data_same = pd.read_csv(csv_file_same, header=None)
data_diff = pd.read_csv(csv_file_diff, header=None)

# find indices of labels with more than 10 images
all_indices = []
for i in range(len(data_train)):
    if data_train.iloc[i,2] > 10:
        all_indices.append(i)

IndicesDataLoader = DataLoader(all_indices, batch_size=16, shuffle=True)

# network architecture: integer - number of convolution layers followed by batch normalization and ReLU, 'M' - max-pooling
architecture = [64, 'M', 128, 'M', 256, 'M', 256, 'M', 512, 512, 'M', 512, 512, 'M']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = self._make_layers(architecture)
        self.fc1 = nn.Linear(4608,128)
#        self.fc2 = nn.Linear(1024,128)

    def forward(self, I):
        out = self.features(I)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
#        out = self.fc2(out)
        out = F.normalize(out,p=2,dim=1)
        return out

    def _make_layers(self, architecture):
        layers = []
        in_channels = 1
        for x in architecture:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
#        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# returns the matrix of pair-wise distances of the input tensors
def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)  

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    dist[dist!=dist] = 0.0    #replace nan values with 0

# ideally root of the distance values has to be taken but doesn't work for some reason
#    return torch.sqrt(dist)
    return torch.clamp(dist, 0.0, np.inf)   # clamping the values between 0.0 and infinite

# naive way of finding valid triplets - takes more time for computation
def find_valid_triplets1(labels):
    a = labels.size(0)
    valid = torch.zeros(a, a, a, dtype=torch.float, device = device)
    for i in range(a):
        for j in range(a):
            for k in range(a):
                # 'i' is anchor, 'j' is positive, 'k' is negative
                # a triplet is valid if (i,j,k) are all distinct and label(i) is same as label(j) and label(i) is different from label(k)
                if (i!=j) and (i!=k) and (j!=k) and (labels[i]==labels[j]) and (labels[i]!=labels[k]):
                    valid[i,j,k] = 1.0
    return valid

# optimized way of finding the valid triplets - takes significantly lesser time for computation
def find_valid_triplets2(labels):
    equal = torch.eye(labels.size(0), dtype=torch.uint8)
    unequal = ~equal
    i_not_j = torch.unsqueeze(unequal,2)
    i_not_k = torch.unsqueeze(unequal,1)
    j_not_k = torch.unsqueeze(unequal,0)
    distinct = (i_not_j & i_not_k) & j_not_k

    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j & (~i_equal_k)

# 'valid' is a 3D matrix with valid[i,j,k] = 1.0 if (i,j,k) is a valid triplet, else valid[i,j,k] is 0.0
    valid = distinct & valid_labels
    return valid.type(torch.cuda.FloatTensor)

# returns batch-all triplet loss for the given batch of embeddings, labels, and margin
def batch_all_loss(embeddings, labels, margin, batch_no, epoch):
#    embeddings = embeddings.detach()
    dist_matrix = pairwise_distances(embeddings, embeddings)

    anc_pos_dist = torch.unsqueeze(dist_matrix,2)
    anc_neg_dist = torch.unsqueeze(dist_matrix,1)

# 3D matrix of triplet-loss values (matrix is computed by broadcasting)
    triplet_loss = anc_pos_dist - anc_neg_dist + margin

#    triplet_loss = torch.zeros(96,96,96,dtype=torch.float,device=device,requires_grad=True)
#    triplet_loss.requires_grad_()

    valid = find_valid_triplets2(labels)
# element-wise matrix multiplication, sets all the loss values of invalid triplets to zero
    triplet_loss = valid*triplet_loss
# sets all the negative loss values to 0.0, i.e., only useful triplets are being taken into account
    triplet_loss = torch.max(triplet_loss,torch.tensor(0.0,device=device))
# 'useful' is a 3D matrix with useful[i,j,k] = 1.0 if (i,j,k) is a useful triplet, else useful[i,j,k] is 0.0
    useful = torch.gt(triplet_loss,0.0).type(torch.FloatTensor)

# total number of useful triplets
    triplets = torch.sum(useful)
# prints the number of triplets for every 50 batches; a metric to evaluate the performce of the training
    if (batch_no%50) == 0:
        print('Epoch -',epoch,',',triplets.item(), file=open("triplets.txt", "a"))

    triplet_loss = torch.sum(triplet_loss)/(triplets+1e-16)

    return triplet_loss

# naive way of finding valid and useful triplets; returns a list of triplets
def batch_all_triplets(embeddings, labels, margin):
    emb = embeddings.cpu().detach().numpy()
    dist_matrix = euclidean_distances(emb, emb)

    r = emb.shape[0]
    triplets = []

    for i in range(r):
        valid = np.zeros(r)
        valid_dist = np.zeros(r)

        for j in range(r):
            if i == j:
                valid[j] = 0
            elif labels[i] == labels[j]:
                valid[j] = 1
            else:
                valid[j] = -1

        for k in range(r):
            valid_dist[k] = valid[k]*dist_matrix[i,k]

        for j in range(r):
            if valid_dist[j] > 0:
                for k in range(r):
                    if valid_dist[k] < 0:
                        if (valid_dist[j]+valid_dist[k]+margin)>0:
                            triplets.append([i,j,k]) 

    return triplets

def main():
    net = Net()
    torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # parallelizes the network on all the available GPUs
    net.to(device)
#    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    margin = 0.2
    
    for epoch in range(64):
        # TRAINING - start
        start = time.time()
        running_loss = 0.0

        print('Epoch -', epoch+1, file=open("diff.txt", "a"))
        print('Margin -', margin, file=open("diff.txt", "a"))

        for batch_no,(indices) in enumerate(IndicesDataLoader):
            data = []

            # for every index, find its label 'a', select 8 images randomly 'rn', read each image, convert to FloatTensor and append to 'data' list
            for i in indices:
                a = str(data_train.iloc[i.item(),1]).zfill(7)
                rn = random.sample(range(1, data_train.iloc[i.item(),2]+1), 8)
                for j in rn:
                    pth = os.path.join(image_dir, a, str(j).zfill(3) + '.jpg')
                    img = io.imread(pth, as_gray=True)
                    img_tensor = torch.FloatTensor(np.expand_dims(img, axis=0))
                    data.append([img_tensor, i.item()])

            ImageLoader = DataLoader(data, batch_size=128, shuffle=False)

            # forward pass, loss computation, backward pass, and update weights
            for img,lbl in ImageLoader:
                optimizer.zero_grad()
                img = img.to(device)
                img_emb = net(img)
                loss = batch_all_loss(img_emb, lbl, 0.4, batch_no+1, epoch+1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # print the cumulative loss for every 50 batches and set it back to zero
            if batch_no % 50 == 49:
                print('[%d, %5d] loss: %.5f' %(epoch + 1, batch_no + 1, running_loss / 50), file=open("loss.txt", "a"))
                running_loss = 0.0

        print('Time taken for the current epoch (training time) -', np.around(time.time()-start, decimals=3), file=open("loss.txt", "a"))
        print(' ', file=open("loss.txt", "a"))
        # TRAINING - end

        # increase margin by 0.1 after every 8 epochs
        if epoch%8 == 7:
            margin = margin + 0.1

        # ANALYSIS - start
        start = time.time()

        # select every alternate index of all indices
        select_indices = all_indices[::2]

        # centroids - size(number of selected indices, 128); each row stores the centroid of an image label
        centroids = torch.zeros(len(select_indices), 128, device=device, dtype=torch.float)
        # centroids_avg_dist - size(number of selected indices); each element stores mean value of image_embedding-centroid distance
        centroids_avg_dist = torch.zeros(len(select_indices), device=device, dtype=torch.float)

        k = 0
        # for every index, select its label 'a', load first 10 images, find their embeddings
        # then calculate centroid and mean distance of image embeddings from centroid 
        for i in select_indices:
            data = []
            a = str(data_train.iloc[i,1]).zfill(7)
            for j in range(10):
                pth = os.path.join(image_dir, a, str(j+1).zfill(3) + '.jpg')
                img = io.imread(pth, as_gray=True)
                img_tensor = torch.cuda.FloatTensor(np.expand_dims(img, axis=0))
                data.append(img_tensor)
            ImageLoader = DataLoader(data, batch_size=10, shuffle=False)
            for img in ImageLoader:
                img_emb = net(img)
            img_emb = img_emb.detach()
            centroids[k] = torch.sum(img_emb,0)/10
            a = (img_emb - centroids[k])**2
            centroids_avg_dist[k] = torch.sum(torch.sqrt(torch.sum(a,1)))/10
            k += 1

        centroid_pairwise_distances = pairwise_distances(centroids,centroids)
        # mean value of all centroid pair-wise distances
        mean_pairwise_distance = torch.sum(centroid_pairwise_distances)/( len(select_indices)*(len(select_indices)-1) )
        # mean value of all the embedding-centroid distances
        mean_centroids_avg_dist = torch.sum(centroids_avg_dist)/len(select_indices)

        print('Epoch -', epoch+1, file=open("analysis.txt", "a"))
        print('Mean centroids pairwise distance:', np.around(mean_pairwise_distance.item(), decimals=3), file=open("analysis.txt", "a"))
        print('Mean images-centroids distance:', np.around(mean_centroids_avg_dist.item(), decimals=3), file=open("analysis.txt", "a"))

        if epoch == 0:  # save the centroids in first epoch
            old_centroids = centroids
        else:           # calculate the distances of centroids' movement and their mean
            m = (centroids - old_centroids)**2
            centroids_movement = torch.sum(torch.sqrt(torch.sum(m,1)))/len(select_indices)
            print('Mean centroids movement:', np.around(centroids_movement.item(), decimals=3), file=open("analysis.txt", "a"))
            old_centroids = centroids
            
        print('Time taken for analysis -', np.around(time.time()-start, decimal=3), file=open("analysis.txt", "a"))
        print(' ', file=open("analysis.txt", "a"))
        # ANALYSIS - end

        # DISTANCES DISTRIBUTION & TESTING - start
        start = time.time()

        svm_values = []
        svm_labels = []

        far0, far1, far2, far3 = 0,0,0,0
        pos_avg = 0.0

        # find the distances between embeddings of 1000 same label pairs, and their mean
        for i in range(1000):
            n1 = data_same.iloc[i,1]
            n2 = data_same.iloc[i,2]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = np.expand_dims(image1, axis=0)
            img1_tensor = torch.cuda.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = np.expand_dims(image2, axis=0)
            img2_tensor = torch.cuda.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            pos_avg = pos_avg + dist[0,0]
            svm_values.append([dist[0,0]])
            svm_labels.append(1)

            if(dist[0,0] < 0.5):
                far0 += 1
            elif (dist[0,0] < 1.0):
                far1 += 1
            elif (dist[0,0] < 2.0):
                far2 += 1
            else:
                far3 += 1
        
        print('Epoch -', epoch+1, file=open("results.txt", "a"))
        print('    < 0.5 :', far0, file=open("results.txt", "a"))
        print('0.5 - 1.0 :', far1, file=open("results.txt", "a"))
        print('1.0 - 2.0 :', far2, file=open("results.txt", "a"))
        print('    > 2.0 :', far3, file=open("results.txt", "a"))
        print('Mean same label distance -', pos_avg/len(data_same), file=open("results.txt", "a"))

        far0, far1, far2, far3 = 0,0,0,0
        neg_avg = 0.0

        # find the distances between embeddings of 1000 different label pairs, and their mean
        for i in range(1000):
            n1 = data_diff.iloc[i,1]
            n2 = data_diff.iloc[i,3]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_diff.iloc[i,0], data_diff.iloc[i,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = np.expand_dims(image1, axis=0)
            img1_tensor = torch.cuda.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_diff.iloc[i,2], data_diff.iloc[i,2] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = np.expand_dims(image2, axis=0)
            img2_tensor = torch.cuda.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            neg_avg = neg_avg + dist[0,0]
            svm_values.append([dist[0,0]])
            svm_labels.append(0)

            if(dist[0,0] < 0.5):
                far0 += 1
            elif (dist[0,0] < 1.0):
            	far1 += 1
            elif (dist[0,0] < 2.0):
            	far2 += 1
            else:
            	far3 += 1
        
        print('    < 0.5 :', far0, file=open("results.txt", "a"))
        print('0.5 - 1.0 :', far1, file=open("results.txt", "a"))
        print('1.0 - 2.0 :', far2, file=open("results.txt", "a"))
        print('    > 2.0 :', far3, file=open("results.txt", "a"))
        print('Mean different label distance -', neg_avg/len(data_diff), file=open("results.txt", "a"))

        # difference between the means - a metric evaluate the separation
        print('Epoch -', epoch+1, file=open("diff.txt", "a") )
        print('DIFFERENCE BETWEEN MEANS -', (neg_avg/len(data_diff)) - (pos_avg/len(data_same)), file=open("diff.txt", "a") )

        # SVM classifier to separate the same-image and different-image pairs
        vals = np.array(svm_values)
        lbls = np.array(svm_labels)
        clf = svm.SVC(gamma='auto')
        clf.fit(vals,lbls)

        pos = 0
        neg = 0

        # based on the SVM model, predict the labels of same-image and different-image pairs (face verification testing)
        for i in range(100):
            n1 = data_same.iloc[i+1000,1]
            n2 = data_same.iloc[i+1000,2]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = np.expand_dims(image1, axis=0)
            img1_tensor = torch.cuda.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = np.expand_dims(image2, axis=0)
            img2_tensor = torch.cuda.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            l = clf.predict(dist)
            if l == 1:
                pos += 1

        print('Positives - ', pos, file=open("results.txt", "a"))

        for i in range(100):
            n1 = data_diff.iloc[i+1000,1]
            n2 = data_diff.iloc[i+1000,3]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_diff.iloc[i+1000,0], data_diff.iloc[i+1000,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = np.expand_dims(image1, axis=0)
            img1_tensor = torch.cuda.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_diff.iloc[i+1000,2], data_diff.iloc[i+1000,2] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = np.expand_dims(image2, axis=0)
            img2_tensor = torch.cuda.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            l = clf.predict(dist)
            if l == 0:
                neg += 1

        print('Negatives - ', neg, file=open("results.txt", "a"))
        print(' ', file=open("results.txt", "a"))
        print('Time taken for validation and testing -', np.around(time.time()-start, decimals=3), file=open("results.txt", "a"))
        print(' ', file=open("results.txt", "a"))
        # DISTANCES DISTRIBUTION & TESTING - end

if __name__ == '__main__':
    main()

# print("Hello stackoverflow!", file=open("CASIA_cpu.txt", "a"))

'''
    for i in range(r):
        for j in range(r):
            if labels[i] != labels[j]:
                dist_matrix[i,j] = (-1*dist_matrix[i,j])
        for j in range(r):
            if dist_matrix[i,j] > 0:
                for k in range(r):
                    if dist_matrix[i,k] < 0:
                        if (dist_matrix[i,j]+dist_matrix[i,k]+margin)>0:
                            triplets.append([i,j,k]) 
'''

'''
            for img,lbl in ImageLoader:
                start = time.time()

                img = img.to(device)
                img_emb = net(img)
                triplet_indices = batch_all_triplets(img_emb, lbl, 0.5)
                print(len(triplet_indices))

                if len(triplet_indices) > 0:
                    TripletLoader = DataLoader(triplet_indices, batch_size=40, shuffle=True)

                    for triplets in TripletLoader:
                        optimizer.zero_grad()
                        a_emb = net(img[np.array(triplets[0])])
                        p_emb = net(img[np.array(triplets[1])])
                        n_emb = net(img[np.array(triplets[2])])
                        loss = criterion(a_emb, p_emb, n_emb)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                print('Batch', batch_no+1, '-', time.time()-start)
'''

'''
        centroid_labels = [234,6337,1298,10500,98] #[4000,8000,108,5,1000]
        data = []

        for i in centroid_labels:
            a = str(data_train.iloc[i,1]).zfill(7)
            for j in range(8):
                pth = os.path.join(image_dir, a, str(j+1).zfill(3) + '.jpg')
                img = io.imread(pth, as_gray=True)
                img_tensor = torch.cuda.FloatTensor(np.expand_dims(img, axis=0))
                data.append(img_tensor)

        ImageLoader = DataLoader(data, batch_size=40, shuffle=False)

        for img in ImageLoader:
            img_emb = net(img)

        img_emb = img_emb.detach()

        centroids = torch.zeros(5, 128, device=device, dtype=torch.float)
        for i in range(5):
            temp = torch.zeros(128, device=device, dtype=torch.float)
            for j in range(8):
                temp = temp + img_emb[i*8+j]
            centroids[i] = temp/8

        centroid_pairwise_distances = pairwise_distances(centroids,centroids)

        print('Epoch -', epoch+1, file=open("analysis.txt", "a"))
        print('Pairwise distance:', file=open("analysis.txt", "a"))
        print(np.around(centroid_pairwise_distances.cpu().numpy(), decimals=3), file=open("analysis.txt", "a"))

        centroids_avg_dist = torch.zeros(5, device=device, dtype=torch.float)
        for i in range(5):
            temp = torch.tensor(0.0, device=device)
            for j in range(8):
                a = (centroids[i] - img_emb[i*8+j])**2
                d = torch.sqrt(torch.sum(a))
                temp = temp + d
            centroids_avg_dist[i] = temp/8

        print('Average distance of images from centroids:', file=open("analysis.txt", "a"))
        print(np.around(centroids_avg_dist.cpu().numpy(), decimals=3), file=open("analysis.txt", "a"))

        if epoch == 0:
            old_centroids = centroids
        else:
            centroids_movement = (centroids - old_centroids)**2
            centroids_movement = torch.sum(centroids_movement, 1)
            centroids_movement = torch.sqrt(centroids_movement)
            print('Centroids Movement:', file=open("analysis.txt", "a"))
            print(np.around(centroids_movement.cpu().numpy(), decimals=3), file=open("analysis.txt", "a"))
            old_centroids = centroids
        print(' ', file=open("analysis.txt", "a"))
'''