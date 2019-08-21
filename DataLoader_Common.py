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

class FER2013Data(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.data_frame.iloc[idx, 2:10].to_numpy()
        class_label = np.argmax(label)
        if self.transform:
            image = self.transform(image)
        return np.expand_dims(image, axis=0), class_label

def FER():
    TrainData = FER2013Data(csv_file='/RawData/fer2013/FERPlus/data/FER2013Train/label.csv', root_dir='/RawData/fer2013/FERPlus/data/FER2013Train/')
    TestData = FER2013Data(csv_file='/RawData/fer2013/FERPlus/data/FER2013Test/label.csv', root_dir='/RawData/fer2013/FERPlus/data/FER2013Test/')

    return TrainData, TestData

def LFW_Labels(img_lbl):
    csv_file_train = 'LFW_Data_Train.csv'
    image_dir = '/RawData/LFW/lfw2/lfw2/'

    labels = []

    data_train = pd.read_csv(csv_file_train, header=None)

    for i in range(len(data_train)):
        if data_train.iloc[i,1] > img_lbl:
            labels.append(i)

    return labels

def LFW_Images(img_lbl):
    csv_file_train = 'LFW_Data_Train.csv'
    image_dir = '/RawData/LFW/lfw2/lfw2/'

    data_train = pd.read_csv(csv_file_train, header=None)

    labels = LFW_Labels(img_lbl)
    LabelsDataLoader = DataLoader(labels, batch_size=10, shuffle=True)

    data = []
    for ind in LabelsDataLoader:
        for i in ind:
            rn = random.sample(range(1, data_train.iloc[i.item(),1]+1), 4)
            for j in rn:
                pth = os.path.join(image_dir, data_train.iloc[i.item(),0], data_train.iloc[i.item(),0] + '_' + str(j).zfill(4) + '.jpg')
                img = io.imread(pth, as_gray=True)
                img_tensor = np.expand_dims(img, axis=0)
                data.append([img_tensor, i.item()])

    return data

def LFW_RecognitionTest():
    csv_file_test = 'LFW_Data_Test.csv'
    image_dir = '/RawData/LFW/lfw2/lfw2/'

    testset1, testset2, testset3 = [],[],[]
    testlabel1, testlabel2, testlabel3 = [],[],[]

    data_test = pd.read_csv(csv_file_test, header=None)

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 1:
            pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_0002.jpg')
            img = io.imread(pth)
            img_tensor = np.expand_dims(img, axis=0)
            testset1.append([img_tensor,i])

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 4:
            for j in range(5):
                pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_000' + str(j+1) + '.jpg')
                img = io.imread(pth)
                img_tensor = np.expand_dims(img, axis=0)
                testset2.append([img_tensor,i])

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 8:
            for j in range(9):
                pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_000' + str(j+1) + '.jpg')
                img = io.imread(pth)
                img_tensor = np.expand_dims(img, axis=0)
                testset3.append([img_tensor,i])

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 2:
            pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_0003.jpg')
            img = io.imread(pth)
            img_tensor = np.expand_dims(img, axis=0)
            testlabel1.append(img_tensor)

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 5:
            pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_0006.jpg')
            img = io.imread(pth)
            img_tensor = np.expand_dims(img, axis=0)
            testlabel2.append(img_tensor)

    for i in range(len(data_test)):
        if data_test.iloc[i,1] > 9:
            pth = os.path.join(image_dir, data_test.iloc[i,0], data_test.iloc[i,0] + '_0010.jpg')
            img = io.imread(pth)
            img_tensor = np.expand_dims(img, axis=0)
            testlabel3.append(img_tensor)

    return testset1, testset2, testset3, testlabel1, testlabel2, testlabel3

def LFW_VerificationTest():
    csv_file_same = 'Data_pairsDevTrain_match.csv'
    csv_file_diff = 'Data_pairsDevTrain_mismatch.csv'
    image_dir_test = '/RawData/LFW/lfw2/lfw2/'

    data_same = pd.read_csv(csv_file_same, header=None)
    data_diff = pd.read_csv(csv_file_diff, header=None)

    validation_images_same = []
    validation_images_diff = []

    test_images_same = []
    test_images_diff = []

    for i in range(1000):
        n1 = data_same.iloc[i,1]
        n2 = data_same.iloc[i,2]
    
        im1 = '_' + str(n1).zfill(4) + '.jpg'
        im2 = '_' + str(n2).zfill(4) + '.jpg'

        pth1 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im1)
        image1 = io.imread(pth1, as_gray=True)
        img1_tensor = np.expand_dims(image1, axis=0)
        img1_tensor = np.expand_dims(img1_tensor, axis=0)
        pth2 = os.path.join(image_dir_test, data_same.iloc[i,0], data_same.iloc[i,0] + im2)
        image2 = io.imread(pth2, as_gray=True)
        img2_tensor = np.expand_dims(image2, axis=0)
        img2_tensor = np.expand_dims(img2_tensor, axis=0)

        validation_images_same.append([img1_tensor,img2_tensor])

    for i in range(1000):
        n1 = data_diff.iloc[i,1]
        n2 = data_diff.iloc[i,3]
    
        im1 = '_' + str(n1).zfill(4) + '.jpg'
        im2 = '_' + str(n2).zfill(4) + '.jpg'

        pth1 = os.path.join(image_dir_test, data_diff.iloc[i,0], data_diff.iloc[i,0] + im1)
        image1 = io.imread(pth1, as_gray=True)
        img1_tensor = np.expand_dims(image1, axis=0)
        img1_tensor = np.expand_dims(img1_tensor, axis=0)
        pth2 = os.path.join(image_dir_test, data_diff.iloc[i,2], data_diff.iloc[i,2] + im2)
        image2 = io.imread(pth2, as_gray=True)
        img2_tensor = np.expand_dims(image2, axis=0)
        img2_tensor = np.expand_dims(img2_tensor, axis=0)

        validation_images_diff.append([img1_tensor,img2_tensor])

    for i in range(100):
        n1 = data_same.iloc[i+1000,1]
        n2 = data_same.iloc[i+1000,2]
    
        im1 = '_' + str(n1).zfill(4) + '.jpg'
        im2 = '_' + str(n2).zfill(4) + '.jpg'

        pth1 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im1)
        image1 = io.imread(pth1, as_gray=True)
        img1_tensor = np.expand_dims(image1, axis=0)
        img1_tensor = np.expand_dims(img1_tensor, axis=0)
        pth2 = os.path.join(image_dir_test, data_same.iloc[i+1000,0], data_same.iloc[i+1000,0] + im2)
        image2 = io.imread(pth2, as_gray=True)
        img2_tensor = np.expand_dims(image2, axis=0)
        img2_tensor = np.expand_dims(img2_tensor, axis=0)

        test_images_same.append([img1_tensor,img2_tensor])

    for i in range(100):
        n1 = data_diff.iloc[i+1000,1]
        n2 = data_diff.iloc[i+1000,3]
    
        im1 = '_' + str(n1).zfill(4) + '.jpg'
        im2 = '_' + str(n2).zfill(4) + '.jpg'

        pth1 = os.path.join(image_dir_test, data_diff.iloc[i+1000,0], data_diff.iloc[i+1000,0] + im1)
        image1 = io.imread(pth1, as_gray=True)
        img1_tensor = np.expand_dims(image1, axis=0)
        img1_tensor = np.expand_dims(img1_tensor, axis=0)
        pth2 = os.path.join(image_dir_test, data_diff.iloc[i+1000,2], data_diff.iloc[i+1000,2] + im2)
        image2 = io.imread(pth2, as_gray=True)
        img2_tensor = np.expand_dims(image2, axis=0)
        img2_tensor = np.expand_dims(img2_tensor, axis=0)

        test_images_diff.append([img1_tensor,img2_tensor])

    return validation_images_same, validation_images_diff, test_images_same, test_images_diff

def CASIA_Labels(img_lbl):
    csv_file = 'CASIA-WebFace/casia_data0.csv'
    image_dir = '/home/tejveer/CASIA-WebFace/'

    data_train = pd.read_csv(csv_file, header=None)

    ind = []
    for i in range(len(data_train)):
        if data_train.iloc[i,2] > img_lbl:
            ind.append(i)

    return ind

def CASIA_Images(img_lbl):
    csv_file = 'CASIA-WebFace/casia_data0.csv'
    image_dir = '/home/tejveer/CASIA-WebFace/'

    data_train = pd.read_csv(csv_file, header=None)

    labels = CASIA_Labels(img_lbl)
    LabelsDataLoader = DataLoader(labels, batch_size=16, shuffle=True)

    data = []
    for ind in LabelsDataLoader:
        for i in ind:
            a = str(data_train.iloc[i.item(),1]).zfill(7)
            rn = random.sample(range(1, data_train.iloc[i.item(),2]+1), 4)
            for j in rn:
                pth = os.path.join(image_dir, a, str(j).zfill(3) + '.jpg')
                img = io.imread(pth, as_gray=True)
                img_tensor = np.expand_dims(img, axis=0)
                data.append([img_tensor, i.item()])

    return data