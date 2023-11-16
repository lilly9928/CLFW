from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader.raf_imagedata import RafDataset
from torchvision import transforms
import torch
from models.transformNetwork448 import Network,Bottleneck
import numpy as np

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor()

    ])


train_dataset= RafDataset(path='D:/data/FER/RAF/basic', phase='train', transform=eval_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network(Bottleneck, [3, 4, 6, 3],num_layers=3).to(device)

model.load_state_dict(torch.load(
    "../weights/train_raf_scale_0.9_acc_tensor(0.8734, device='cuda_0', dtype=torch.float64).pt"))
model.eval()

tsne = TSNE(n_components=2)



test_predictions  = []
test_targets  = []

test_embeddings = torch.zeros((0, 100), dtype=torch.float32)
#??

classes =  ['SU','FE','DI','HA','SA','AN','NE']

actual = []
deep_features = []

model.eval()

for img,_,_,label in train_loader:
    images, labels = img.to(device), label.to(device)
    features,out = model(images)

    deep_features += features.cpu().numpy().tolist()
    actual += labels.cpu().numpy().tolist()

tsne = TSNE(n_components=2, random_state=0)
cluster = np.array(tsne.fit_transform(np.array(deep_features)))
actual = np.array(actual)

plt.figure(figsize=(7, 7))

for i, label in zip(range(7), classes):
    idx = np.where(actual == i)
    plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

plt.legend()
plt.show()