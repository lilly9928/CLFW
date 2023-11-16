
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader.imagedata import ImageData
from torchsummary import summary
from models.transformNetwork import Network, init_weights
from models.tripletloss import TripletLoss
from torch.utils.tensorboard import SummaryWriter


#hyperparmeters
batch_size = 8
epochs =5
learning_rate=0.001

#data_aug
train_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_train.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/clean/data/ck_val.csv")

train_ds = ImageData(train_df,

                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                 ]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)


test_ds = ImageData(test_df,  transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


anchor,positive,negative,labels = next(iter(train_loader))

# anchor[0].shape
# torch_image = torch.squeeze(anchor[0])
# image = torch_image.numpy()
# image.shape

#device, model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)
model.train()
summary(model, input_size=(1, 48, 48))
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = TripletLoss()
writer = SummaryWriter(f'runs/CK/MiniBatchsize {batch_size} LR {learning_rate}')

for epoch in range(epochs):
    running_loss = []
    #losses = []
    accuracies = []
    step = 0

    for batch_idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):

        #get data_aug to cuda
        anchor_img, positive_img, negative_img, anchor_label =\
            anchor_img.to(device), positive_img.to(device), negative_img.to(device), anchor_label.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        loss = criterion(anchor_out, positive_out, negative_out)


        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        writer.add_scalar('Training Loss',loss,global_step=step)
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
