# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import cv2
import torch.utils.data as data
import torch
import pandas as pd
import random
import numpy as np
from torchvision import transforms
from skimage.feature import hog
import matplotlib.pyplot as plt
import torchvision
from skimage import exposure


def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

class RafDataset(data.Dataset):
    def __init__(self, path, phase, basic_aug=True, transform=None):
        self.path =path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(os.path.join(self.path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)

        name_c = 0 # 0 인덱스 (파일명)
        label_c = 1 # 1 인덱스 (라벨)
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')] #0번째 인덱스(이름)에서 train으로 시작하는 파일
            self.index =dataset.index.values
        else:
            df = pd.read_csv(os.path.join(self.path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')] #0번째 인덱스(이름)에서 test으로 시작하는 파일
            self.index = dataset.index.values

        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        # self.clean = ('list_patition_label.txt' == 'list_patition_label.txt')

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(path, "Image/aligned/aligned", f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]


        # if not self.clean:
        #     image1 = image
        #     image1 = self.aug_func[0](image)
        #     image1 = self.transform(image1)

        if self.phase == 'train':
            positive_list =self.index[self.index!=idx][self.label[self.index!=idx]==int(label)]
            positive_item = random.choice(positive_list)
            positive_image = cv2.imread(self.file_paths[positive_item])

            negative_list=self.index[self.index!=idx][self.label[self.index != idx]!= int(label)]
            negative_item = random.choice(negative_list)
            negative_image = cv2.imread(self.file_paths[negative_item])
        #??
        # if self.phase == 'train':
        #     if self.basic_aug and random.uniform(0, 1) > 0.5:
        #         image = self.aug_func[1](image)

        # if self.phase == 'train' or self.phase == 'test':
        #    fd,image = hog(image, orientations=24, pixels_per_cell=(16, 16),
        #                cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        #    image = exposure.rescale_intensity(image, in_range=(0, 10))

            if self.transform is not None:
                image = self.transform(image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

                return image, positive_image, negative_image, label

        else:

            if self.transform is not None:
                image = self.transform(image)

            return image, self.file_paths,label


        # if self.clean:
        #     image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, positive_image, negative_image, label

if __name__ == "__main__":
   #check sample images
    def show(img, y=None):
        npimg = img.numpy()
        npimg_tr = np.transpose(npimg, (1, 2, 0))
        plt.imshow(npimg_tr)

        if y is not None:
            plt.title('labels:' + str(y))

#compound
    # classes = ['Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised',
    #            'Sadly Disgusted', 'Fearfully Angry',
    #            'Fearfully Surprised', 'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised']

   #basic
    classes = [ 'Surprise', 'Fear','Disgust','Happiness','Sadness','Anger','Neutral']

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
       # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
       # transforms.Normalize(mean=[0.485, 0.456, 0.406],
       #                      std=[0.229, 0.224, 0.225]),
       # transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    dataset=RafDataset(path='D:\\data\\FER\\RAF\\basic',phase='test', transform=train_transforms)

    grid_size=4
    rnd_ind = np.random.randint(0, len(dataset), grid_size)

    print(dataset[rnd_ind[1]][1])

    x_grid = [dataset[i][0] for i in rnd_ind]
    y_grid = [classes[dataset[i][1]] for i in rnd_ind]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=grid_size, padding=2)
    plt.figure(figsize=(10,10))
    show(x_grid,y_grid)
    plt.show()
