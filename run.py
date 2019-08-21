from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DataLoader_Common import FER, LFW_Labels, LFW_Images, LFW_RecognitionTest, LFW_VerificationTest, CASIA_Labels, CASIA_Images
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=64)
    args = parser.parse_args()

    if (args.dataset == 1):
        # images (as numpy arrays) and class labels (integers)
        TrainData, TestData = FER()

        # PyTorch DataLoader
        TrainDataLoader = DataLoader(TrainData, batch_size=16, shuffle=True)
        TestDataLoader = DataLoader(TestData, batch_size=16, shuffle=False)

        print('FER data loaded')

    elif (args.dataset == 2):
        # labels with images greater than 'img_lbl'
        img_lbl = 3

        # 'labels' is a list of face labels (integers)
        labels = LFW_Labels(img_lbl)

        # PyTorch DataLoader
        LabelsDataLoader = DataLoader(labels, batch_size=10, shuffle=True)

        # 'Images_Data' is a list of [images,labels]; images as numpy arrays and labels as integers
        Images_Data = LFW_Images(img_lbl)

        # PyTorch DataLoder
        ImageDataLoader = DataLoader(Images_Data, batch_size=64, shuffle=False)

        print('LFW data loaded')

    elif (args.dataset == 3):
        # labels with images greater than 'img_lbl'
        img_lbl = 5

        # 'labels' is a list of face labels (integers)
        labels = CASIA_Labels(img_lbl)

        # PyTorch DataLoader
        LabelsDataLoader = DataLoader(labels, batch_size=16, shuffle=True)

        # 'Images_Data' is a list of [images,labels]; images as numpy arrays and labels as integers
        Images_Data = CASIA_Images(img_lbl)

        # PyTorch DataLoder
        ImageDataLoader = DataLoader(Images_Data, batch_size=64, shuffle=False)

        print('CASIA data loaded')

    elif (args.dataset == 4):
        # Image pairs (as numpy arrays) for validation (to find the best margin value) and testing the performance
        # Validation - 1000 pairs each, Test - 100 pairs each
        LFW_Validation_Same, LFW_Validation_Diff, LFW_Test_Same, LFW_Test_Diff = LFW_VerificationTest()

        print('LFW Test Data (images pairs) loaded -- for face verification')

    elif (args.dataset == 5):        
        # 'TestSet1', 'TestSet2', 'TestSet3' are lists of [images,labels]; images as numpy arrays and labels as integers
        # 'TestSet1', 'TestSet2', 'TestSet3' has 1,5,9 images/label respectively; used as test database
        # 'TestLabels1', 'TestLabels2', 'TestLabels3' are lists of images (numpy arrays) for predicting the labels (recognition)
        TestSet1, TestSet2, TestSet3, TestLabels1, TestLabels2, TestLabels3 = LFW_RecognitionTest()
        
        print('LFW Test Data loaded -- for face recognition')

if __name__ == '__main__':
    main()