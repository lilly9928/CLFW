import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from data_loader.imagedata import ImageData
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicStn(nn.Module):

    def __init__(self, parallel, in_feature, **kwargs):
        super(BasicStn, self).__init__()
        self.conv = conv1x1(in_feature, 128)
        self.fc_loc = nn.Sequential(
            nn.Linear(128*7*7, 64),
            nn.Tanh(),
            nn.Linear(64, 2*len(parallel)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
       # print(x.shape)
        x = x.view(-1, 128*7*7)
        x = self.fc_loc(x)
        return x


class BasicFc(nn.Module):

    def __init__(self, in_feature, out_feature, p=0, **kwargs):
        super(BasicFc, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class StnFc975(nn.Module):

    def __init__(self, parallel, in_feature, out_feature,layer_num):
        super(StnFc975, self).__init__()
        self.parallel = parallel
        self.out_feature = out_feature
        self.stn = BasicStn(parallel, in_feature)
        self.fc1 = BasicFc(in_feature, out_feature)
        self.fc2 = BasicFc(in_feature, out_feature)
        self.fc3 = BasicFc(in_feature, out_feature)
        self.fc4 = BasicFc(in_feature, out_feature)
        self.layer_num = layer_num

    def forward(self, feature):
        x = self.fc1(feature)
        thetas = self.stn(feature) # theta 값 추출

        for i in range(0,self.layer_num):
            theta = thetas[:, (i) * 2:(i + 1) * 2]
            theta = theta.view(-1, 2, 1)
            crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
            crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
            theta = torch.cat((crop_matrix, theta), dim=2)  # [32,2,3]
            grid = F.affine_grid(theta, feature.size(),align_corners=True)  # [n,h,w,2] 2,7,7
            xs = F.grid_sample(feature, grid,align_corners=True)  # channel 2048
            x += self.fc2(xs)


        # theta = thetas[:, (i)*2:(i+1)*2]
        # theta = theta.view(-1, 2, 1)
        # crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        # crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        # theta = torch.cat((crop_matrix, theta), dim=2) #[1,2,3]
        # grid = F.affine_grid(theta, feature.size()) #[n,h,w,2] 2,7,7
        # xs = F.grid_sample(feature, grid) #channel 2048
        # x += self.fc2(xs)
        # i += 1
        #
        # theta = thetas[:, (i)*2:(i+1)*2]
        # theta = theta.view(-1, 2, 1)
        # crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        # crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        # theta = torch.cat((crop_matrix, theta), dim=2)
        # grid = F.affine_grid(theta, feature.size())
        # xs = F.grid_sample(feature, grid)
        # x += self.fc3(xs)
        # i += 1
        #
        # theta = thetas[:, (i)*2:(i+1)*2]
        # theta = theta.view(-1, 2, 1)
        # crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        # crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        # theta = torch.cat((crop_matrix, theta), dim=2)
        # grid = F.affine_grid(theta, feature.size())
        # xs = F.grid_sample(feature, grid)
        # x += self.fc4(xs)

        return x


class Network(nn.Module):

    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, p=0, parallel=[0.9, 0.7, 0.5],num_layers=1):
        super(Network, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        self.stn_fc = StnFc975(parallel, 512 * block.expansion, num_classes,num_layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x) #input [8,3,448,448] output [8,64,224,224]
        out = self.bn1(out) #output [2,64,224,224]
        out = self.relu(out) #output [2,64,224,224]
        out = self.maxpool(out) #[2,64,112,112]

        out = self.layer1(out) #[2,256,112,112]
        out = self.layer2(out) #[2,512,56,56]
        out = self.layer3(out)#[2,1024,28,28]
        feature = self.layer4(out) #512 #[8,2048,14,14]

        out = self.stn_fc(feature)
        return feature,out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':

    # data_aug
    train_csvdir = 'D:/data/FER/ck_images/ck_train.csv'
    traindir = "D:/data/FER/ck_images/Images/ck_train/"
    val_csvdir = 'D:/data/FER/ck_images/ck_val.csv'
    valdir = "D:/data/FER/ck_images/Images/ck_val/"

    transformation = transforms.Compose([transforms.ToTensor()])

    train_dataset =ImageData(csv_file = train_csvdir, img_dir = traindir, datatype = 'ck_train',transform = transformation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network(Bottleneck, [3, 4, 6, 3]).to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # for epoch in range(1):
    #     for batch_idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
    #         # get data_aug to cuda
    #         anchor_img, positive_img, negative_img = \
    #             anchor_img.to(device), positive_img.to(device), negative_img.to(device)
    #
    #
    #         anchor_label = anchor_label.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         anchor_feature, anchor_out = model(anchor_img)
    #         entropy_loss = criterion(anchor_out, anchor_label)
    #         loss = entropy_loss
    #
    #         loss.backward()
    #         optimizer.step()
    #         _, preds = torch.max(anchor_out, 1)


    summary(model, input_size=(3, 224, 224))

