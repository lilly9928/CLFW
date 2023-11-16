
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.imagedata import ImageData
from torchsummary import summary
from models.transformNetwork import Network, init_weights
from models.tripletloss import TripletLoss
from torch.utils.tensorboard import SummaryWriter


class TestNetwork(nn.Module):
    def __init__(self, emb_dim=128):
        super(TestNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),  ## 28*28*1 -> 24*24*32
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 24*24*32 -> 12*12*32
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),  # 12*12*3 -> 8*8*64
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # 8*8*64-> 4*4*32
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

#hyperparmeters
batch_size = 1000
epochs =50
learning_rate=(0.3 * 8 / 256)
embedding_dims = 2

#data_aug
train_df = pd.read_csv("C:/Users/1315/Desktop/data/train.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/data/finaltest_q.csv")

train_ds = ImageData(train_df,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                #    transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                 ]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)


test_ds = ImageData(test_df, train=False, transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


anchor,positive,negative,labels = next(iter(train_loader))

anchor[0].shape
torch_image = torch.squeeze(anchor[0])
image = torch_image.numpy()
print(anchor.shape)

#device, model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = TripletLoss(margin=1.0)

model.train()
summary(model, input_size=(1, 48, 48))

writer = SummaryWriter(f'runs/FER2013/MiniBatchsize {batch_size} LR {learning_rate}')
classes = ['0','1','2','3','4','5','6']
for epoch in range(epochs):
    running_loss = []
    accuracies = []

    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_loader)):

        #get data_aug to cuda
        anchor_img, positive_img, negative_img =\
            anchor_img.to(device), positive_img.to(device), negative_img.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        img_grid = torchvision.utils.make_grid(anchor_img)
        img_grid1 = torchvision.utils.make_grid(positive_img)
        img_grid2 = torchvision.utils.make_grid(negative_img)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.cpu().detach().numpy())

        features = anchor_img.reshape(anchor_img.shape[0], -1)
        class_labels = [classes[label] for label in anchor_label]

        writer.add_image('anchor', img_grid)
        writer.add_image('positive', img_grid1)
        writer.add_image('negative', img_grid2)

        writer.add_scalar('Training Loss',loss)

        writer.add_embedding(features, metadata=class_labels, label_img=anchor_img)
        step += 1

    print(f'Mean Loss this epoch was {sum(running_loss) / len(running_loss)}')
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

# train_results = []
# labels = []
# classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# model.eval()
# with torch.no_grad():
#     for img, _, _, label in train_loader:
#         img, label = img.to(device), label.to(device)
#         train_results.append(model(img).cpu().numpy())
#         labels.append(label.cpu())
#
# train_results = np.concatenate(train_results)
# labels = np.concatenate(labels)
# labels.shape

# ## visualization
# plt.figure(figsize=(15, 10), facecolor="azure")
# for label in np.unique(labels):
#     tmp = train_results[labels == label]
#     plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])
#
# plt.legend()
# plt.show()
#
# tree = XGBClassifier(seed=180)
# tree.fit(train_results, labels)
#
# test_results = []
# test_labels = []
#
# model.eval()
# with torch.no_grad():
#     for img in test_loader:
#         img = img.to(device)
#         test_results.append(model(img).cpu().numpy())
#         test_labels.append(tree.predict(model(img).cpu().numpy()))
#
# test_results = np.concatenate(test_results)
# test_labels = np.concatenate(test_labels)
#
# plt.figure(figsize=(15, 10), facecolor="azure")
# for label in np.unique(test_labels):
#     tmp = test_results[test_labels == label]
#     plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])
#
# plt.legend()
# plt.show()
#
# # accuracy
# true_ = (tree.predict(test_results) == test_labels).sum()
# len_ = len(test_labels)
# print(tree.predict(test_results))
# print(test_labels)
# print("Accuracy :{}%".format((true_ / len_) * 100))  ##100%
