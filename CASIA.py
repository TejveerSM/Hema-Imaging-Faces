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

csv_file = 'CASIA-WebFace/casia_data0.csv'
image_dir = '/home/tejveer/CASIA-WebFace/'

csv_file_same = 'Data_pairsDevTrain_match.csv'
csv_file_diff = 'Data_pairsDevTrain_mismatch.csv'
image_dir_test = '/RawData/LFW/lfw2/lfw2/'

data_train = pd.read_csv(csv_file, header=None)
data_same = pd.read_csv(csv_file_same, header=None)
data_diff = pd.read_csv(csv_file_diff, header=None)

ind = []
for i in range(len(data_train)):
    if data_train.iloc[i,2] > 5:
        ind.append(i)

TrainDataLoader = DataLoader(ind, batch_size=32, shuffle=True)
print()

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

def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)

    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)  

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist[dist!=dist] = 0

    return torch.clamp(dist, 0.0, np.inf)

def find_valid_triplets1(labels):
    a = labels.size(0)
    valid = torch.zeros(a, a, a, dtype=torch.float, device = device)
    for i in range(a):
        for j in range(a):
            for k in range(a):
                if (i!=j) and (i!=k) and (j!=k) and (labels[i]==labels[j]) and (labels[i]!=labels[k]):
                    valid[i,j,k] = 1.0
    return valid

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

    valid = distinct & valid_labels
    return valid.type(torch.cuda.FloatTensor)

def batch_all_loss(embeddings, labels, margin):
#    embeddings = embeddings.detach()
    dist_matrix = pairwise_distances(embeddings, embeddings)

    anc_pos_dist = torch.unsqueeze(dist_matrix,2)
    anc_neg_dist = torch.unsqueeze(dist_matrix,1)

#    triplet_loss = torch.zeros(64,64,64,dtype=torch.float,device=device,requires_grad=True)
    triplet_loss = anc_pos_dist - anc_neg_dist + margin

    valid = find_valid_triplets2(labels)
    triplet_loss = valid*triplet_loss
    triplet_loss = torch.max(triplet_loss,torch.tensor(0.0,device=device))
    useful = torch.gt(triplet_loss,0.0).type(torch.FloatTensor)

    triplets = torch.sum(useful)
    triplet_loss = torch.sum(triplet_loss)/(triplets+1e-16)

    return triplet_loss

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
        net = nn.DataParallel(net)
    net.to(device)
#    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    
    for epoch in range(64):
        running_loss = 0.0

        for batch_no,(ind) in enumerate(TrainDataLoader):
            data = []

            for i in ind:
                a = str(data_train.iloc[i.item(),1]).zfill(7)
                rn = random.sample(range(1, data_train.iloc[i.item(),2]+1), 4)
                for j in rn:
                    pth = os.path.join(image_dir, a, str(j).zfill(3) + '.jpg')
                    img = io.imread(pth, as_gray=True)
                    img_tensor = torch.FloatTensor(np.expand_dims(img, axis=0))
                    data.append([img_tensor, i.item()])

            ImageLoader = DataLoader(data, batch_size=128, shuffle=False)

            for img,lbl in ImageLoader:
                img = img.to(device)
                img_emb = net(img)
                loss = batch_all_loss(img_emb, lbl, 0.4)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if batch_no % 50 == 49:
                print('[%d, %5d] loss: %.5f' %(epoch + 1, batch_no + 1, running_loss / 50))
                running_loss = 0.0

        svm_values = []
        svm_labels = []

        correct = 0
        far, far1, far2 = 0,0,0
        pos_avg = 0.0

        for i in range(1000):
            n1 = data_same.iloc[i,1]
            n2 = data_same.iloc[i,2]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = torch.FloatTensor(np.expand_dims(image1, axis=0))
            img1_tensor = torch.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = torch.FloatTensor(np.expand_dims(image2, axis=0))
            img2_tensor = torch.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            pos_avg = pos_avg + dist[0,0]
            svm_values.append([dist[0,0]])
            svm_labels.append(1)

            if(dist[0,0] < 0.5):
                correct += 1
            elif (dist[0,0] < 1.0):
                far1 += 1
            elif (dist[0,0] < 2.0):
                far2 += 1
            else:
                far += 1
        
        print()
        print('  Correct -', correct)
        print('0.5 - 1.0 :', far1)
        print('1.0 - 2.0 :', far2)
        print('    > 2.0 :', far)
        print()
        print('Average positive distance -', pos_avg/len(data_same))

        wrong = 0
        far, far1, far2 = 0,0,0
        neg_avg = 0.0

        for i in range(1000):
            n1 = data_diff.iloc[i,1]
            n2 = data_diff.iloc[i,3]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_diff.iloc[i,0], data_diff.iloc[i,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = torch.FloatTensor(np.expand_dims(image1, axis=0))
            img1_tensor = torch.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_diff.iloc[i,2], data_diff.iloc[i,2] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = torch.FloatTensor(np.expand_dims(image2, axis=0))
            img2_tensor = torch.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            neg_avg = neg_avg + dist[0,0]
            svm_values.append([dist[0,0]])
            svm_labels.append(0)

            if(dist[0,0] < 0.5):
                wrong += 1
            elif (dist[0,0] < 1.0):
            	far1 += 1
            elif (dist[0,0] < 2.0):
            	far2 += 1
            else:
            	far += 1
        
        print()
        print('    Wrong -', wrong)
        print('0.5 - 1.0 :', far1)
        print('1.0 - 2.0 :', far2)
        print('    > 2.0 :', far)
        print()
        print('Average negative distance -', neg_avg/len(data_diff))

        print()
        vals = np.array(svm_values)
        lbls = np.array(svm_labels)
        clf = svm.SVC(gamma='auto')
        clf.fit(vals,lbls)

        pos = 0
        neg = 0

        for i in range(100):
            n1 = data_same.iloc[i+1000,1]
            n2 = data_same.iloc[i+1000,2]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = torch.FloatTensor(np.expand_dims(image1, axis=0))
            img1_tensor = torch.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = torch.FloatTensor(np.expand_dims(image2, axis=0))
            img2_tensor = torch.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            l = clf.predict(dist)
            if l == 1:
                pos += 1

        print('Positives - ', pos)

        for i in range(100):
            n1 = data_diff.iloc[i+1000,1]
            n2 = data_diff.iloc[i+1000,3]
    
            im1 = '_' + str(n1).zfill(4) + '.jpg'
            im2 = '_' + str(n2).zfill(4) + '.jpg'

            pth1 = os.path.join(image_dir_test, data_diff.iloc[i+1000,0], data_diff.iloc[i+1000,0] + im1)
            image1 = io.imread(pth1, as_gray=True)
            img1_tensor = torch.FloatTensor(np.expand_dims(image1, axis=0))
            img1_tensor = torch.FloatTensor(np.expand_dims(img1_tensor, axis=0))
            pth2 = os.path.join(image_dir_test, data_diff.iloc[i+1000,2], data_diff.iloc[i+1000,2] + im2)
            image2 = io.imread(pth2, as_gray=True)
            img2_tensor = torch.FloatTensor(np.expand_dims(image2, axis=0))
            img2_tensor = torch.FloatTensor(np.expand_dims(img2_tensor, axis=0))

            emb1 = net(img1_tensor)
            emb2 = net(img2_tensor)

            emb1 = emb1.cpu().detach().numpy()
            emb2 = emb2.cpu().detach().numpy()

            dist = euclidean_distances(emb1, emb2)
            l = clf.predict(dist)
            if l == 0:
                neg += 1

        print('Negatives - ', neg)
        print()

if __name__ == '__main__':
    main()

#print("Hello stackoverflow!", file=open("output.txt", "a"))

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

                print('Batch', batch_no+1, '-', time.time()-start) '''