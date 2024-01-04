import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from discriminator import Discriminator
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
root_dir = './path/to/your/dataset/'

dataset = torchvision.datasets.ImageFolder(root=root_dir,transform=transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
num_channels = 1
    
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])    # Define the data loaders for the training and testing sets

train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

#saving all images passed in train_dataset
# for i, (img, _) in enumerate(train_dataset):
#     save_image(img, f'./output_ir_disc/input/image_{i}.png')

#residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.identity_mapping(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        return x
    
#resnet based encoder
    
class ResNet102Encoder(nn.Module):
    def __init__(self):
        super(ResNet102Encoder, self).__init__()
        
        # Encoder (ResNet18 without the final fully connected layers)
        resnet = models.resnet101(pretrained=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-2]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

#resnet based decoder

class ResNet102Decoder(nn.Module):
    def __init__(self, in_channels=2048, num_channels=64, num_classes=1):
        super(ResNet102Decoder, self).__init__()

        channels = [in_channels, 1024, 512, 256, 128, 64]

        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.layer1 = self._make_layer(ResidualBlock, channels[1], channels[1], 3)
        self.layer2 = self._make_layer(ResidualBlock, channels[1], channels[2], 4)
        self.layer3 = self._make_layer(ResidualBlock, channels[2], channels[3], 6)
        self.layer4 = self._make_layer(ResidualBlock, channels[3], channels[4], 3)
        self.layer5 = self._make_layer(ResidualBlock, channels[4], channels[5], 3)

        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], channels[-2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[-2]),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(channels[-2], channels[-3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[-3]),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(channels[-3], channels[-4], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[-4]),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(channels[-4], num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        # Final convolutional layer
        self.final = nn.Conv2d(num_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.final(x)
        return x


class ResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ResNetAutoencoder, self).__init__()

        self.encoder = ResNet102Encoder()
        self.decoder = ResNet102Decoder()



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

@staticmethod
def adopt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
        disc_factor = value
    return disc_factor
    


model = ResNetAutoencoder().to(device)
disc = Discriminator().to(device)
# print(model)
# print(disc)

disc_factor = 1.
criteria = nn.MSELoss()
autoencoder_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
discriminator_optimizer = torch.optim.Adam(disc.parameters(),
                                    lr=1e-4, eps=1e-08, betas=(0.0, 0.9))

best_autoencoder_loss = float('inf')
best_discriminator_loss = float('inf')

num_epochs = 1000
best_model = 10000
for epoch in range(num_epochs):
    with tqdm(range(len(train_loader))) as t:
        for i, data in enumerate(train_loader):
            img, _ = data
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            loss = criteria(output, img)
            disc_real = disc(img)
            disc_fake = disc(output.detach())

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

            # ===================backward====================
            autoencoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Backpropagate through the autoencoder
        
            discriminator_optimizer.zero_grad()
            gan_loss.backward()  # Backpropagate through the discriminator
            autoencoder_optimizer.step()
            discriminator_optimizer.step()

            disc_factor = adopt_weight(disc_factor, epoch * len(train_loader) + i, 10000, 0.1)
            t.set_description(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}, gan_loss:{gan_loss.item():.4f}')
            t.update()

        def to_img(tensor): 
            tensor = tensor.clamp(0, 1)
            tensor = tensor.view(tensor.size(0), 1, 256, 256)
            return tensor
        
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, f'./output_autoenc_discriminator/image_{epoch}.png')

    #saving best model
        if loss < best_autoencoder_loss:
            best_autoencoder_loss = loss
            torch.save(model.state_dict(), './output_autoenc_discriminator/best_autoencoder_model.pth')

    # Track the best discriminator model based on GAN loss
        if gan_loss < best_discriminator_loss:
            best_discriminator_loss = gan_loss
            torch.save(disc.state_dict(), './output_autoenc_discriminator/best_discriminator_model.pth')



    
    






