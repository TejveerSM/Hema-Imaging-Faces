from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

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
        return torch.FloatTensor(np.expand_dims(image, axis=0)), torch.tensor(class_label)

TrainData = FER2013Data(csv_file='/RawData/fer2013/FERPlus/data/FER2013Train/label.csv', root_dir='/RawData/fer2013/FERPlus/data/FER2013Train/')
TestData = FER2013Data(csv_file='/RawData/fer2013/FERPlus/data/FER2013Test/label.csv', root_dir='/RawData/fer2013/FERPlus/data/FER2013Test/')

TrainDataLoader = DataLoader(TrainData, batch_size=16, shuffle=True)
TestDataLoader = DataLoader(TestData, batch_size=16, shuffle=False)

architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(architecture)
        self.fc1 = nn.Linear(2304,1024)
        self.fc2 = nn.Linear(1024,8)

    def forward(self, I):
        out = self.features(I)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def _make_layers(self, architecture):
        layers = []
        in_channels = 1
        for x in architecture:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels,x,kernel_size=3,padding=1),nn.BatchNorm2d(x),nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)

def main():
    net = VGG()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    
    for epoch in range(5):
        running_loss = 0.0
        for batch_no, (images, labels) in enumerate(TrainDataLoader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_no % 200 == 199:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, batch_no + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in TestDataLoader:
            images,labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy:')
    print(correct/total)

if __name__ == '__main__':
    main()