import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2
import glob

class FacadeDataset(Dataset):
    def __init__(self, flag, dataDir='./cifar_image/', data_range=(0, 8)):
        assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])

        self.dataset = []
        for i in range(data_range[0], data_range[1]):
            img = Image.open(os.path.join(dataDir,flag,'%04d.jpg' % i))

            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
            img = img.astype('float') / 128.0

            img = np.transpose(img, [2,0,1])
            img_L = img[0,::].reshape([1, img.shape[1], img.shape[2]])
            img = img[1:, ::]

            self.dataset.append((img_L, img))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.FloatTensor(img), torch.FloatTensor(label)

class FacadeDataset_256(Dataset):
    def __init__(self, flag, dataDir='./image256/', data_range=(0, 8)):
        assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])

        self.dataset = []
        image_names = glob.glob(dataDir + flag + '/*.jpg')
        for i in range(data_range[0], data_range[1]):
            img = Image.open(image_names[i])

            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
            img = img.astype('float') / 128.0

            img = np.transpose(img, [2,0,1])
            img_L = img[0,::].reshape([1, img.shape[1], img.shape[2]])
            img = img[1:, ::]

            self.dataset.append((img_L, img))
        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.FloatTensor(img), torch.FloatTensor(label)
