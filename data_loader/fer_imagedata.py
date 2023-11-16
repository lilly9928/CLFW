import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
import pandas as pd
import random
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import  Dataset
from skimage import exposure
from skimage.feature import hog
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class FERimageData(Dataset):

    def __init__(self, csv_file, img_dir, datatype, transform):


        self.csv_file = pd.read_csv(csv_file)
        self.lables = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype
        self.index = self.csv_file.index.values



    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        src = cv2.imread(self.img_dir + self.datatype + str(idx) + '.jpg')
        img = cv2.resize(src, (224, 224))
        lables = np.array(self.lables[idx])
        lables = torch.from_numpy(lables).long()

        if self.datatype == 'train': #train 데이터 일 경우

            ## 해당 anchor가 아닌 것들중에서 Label 같은 것들의 index를 가지고 옮
            positive_list = self.index[self.index != idx][self.lables[self.index != idx] == int(lables)]

            positive_item = random.choice(positive_list)
            positive_src = cv2.imread(self.img_dir + self.datatype + str(positive_item) + '.jpg')
            positive_img = cv2.resize(positive_src, (224, 224))

            ## 해당 anchor가 아닌 것들중에서 Label 다른 것들의 index를 가지고 옮
            negative_list = self.index[self.index != idx][self.lables[self.index != idx] != int(lables)]

            nagative_item = random.choice(negative_list)
            negative_src = cv2.imread(self.img_dir + self.datatype + str(nagative_item) + '.jpg')
            negative_img = cv2.resize(negative_src, (224, 224))

            if self.transform:
                anchor_img = self.transform(img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, lables

        else:
            srcc = self.img_dir + self.datatype + str(idx) + '.jpg'
            if self.transform: #val 데이터 일 경우
                anchor_img = self.transform(img)

            return anchor_img,srcc,lables


if __name__ == "__main__":

    def imshow(img, text=None, should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
    traindir = "D:/data/FER/ck_images/Images/ck_train/"
    val_csvdir= 'D:/data/FER/ck_images/ck_val.csv'
    valdir = "D:/data/FER/ck_images/Images/ck_val/"
    batch_size = 8
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),])

    train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
        imshow(anchor_img[0])
        imshow(positive_img[0])
        imshow(negative_img[0])
        exit()