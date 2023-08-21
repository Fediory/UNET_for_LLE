import os
import time

from tqdm import tqdm

from UNet import UNet
from MyDataset import MyDataset
import numpy as np
from PIL import Image
from net.net import net
from loss import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pytorch_ssim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

transform = transforms.Compose([
            transforms.ToTensor()
        ])

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
else:
    device = torch.device("cpu")

# Train
epoch = 20
batch_size = 100
lr = 0.05

# 划分训练集和验证集
print("== start to read dataset")
now_time = time.time()
dataset = MyDataset("D:/Desktop/大学课程/闫-计算机视觉/UNET/datasets/train", f2_max=20, image_max=5)
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
train_dataset, val_dataset = data.random_split(dataset, (train_size, val_size))

print(f"time use: {time.time() - now_time}")
print(f"dataset length is {len(dataset)}\n")

print("== start to load dataset")
now_time = time.time()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=1)
print(f"time use: {time.time() - now_time}\n")

print("== start to train")
now_time = time.time()

# model with unet
model = net()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)
for now_epoch in range(epoch):
    batches = len(dataset)
    pbar = tqdm(total=batches, desc=f"Epoch {now_epoch + 1} / {epoch}: ")
    running_loss = 0.0
    folder_path2 = './progress'.replace('\\', '/')
    for iteration, (images, labels) in enumerate(dataset):
        file_name = f"image_epoch{now_epoch}_number{iteration}.png"
        progress_path = os.path.join(folder_path2, file_name).replace('\\', '/')
        L1, R1, X1 = model(images)
        L2, R2, X2 = model(labels)
        
        loss1 = C_loss(R1, R2)
        images = images.unsqueeze(0)
        loss2 = R_loss(L1, R1, images, X1)
        loss3 = P_loss(images, X1)
        loss =  loss1 * 1 + loss2 * 1 + loss3 * 500
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  
        
        I = torch.pow(L1,0.14) * R1
        I = I.squeeze(0)
        output_image = transforms.ToPILImage()(I)
        if iteration == 0:
            output_image.save(progress_path)  
        output_image.show()
             
        pbar.update(1)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {now_epoch + 1}/{epoch}, Loss: {epoch_loss:.4f}")
    if now_epoch == (epoch-1):
        save_path = os.path.join(model_dir, time_stamp + f"-E{now_epoch}-L{epoch_loss:.4f}.pt").replace("\\", '/')
        torch.save(model.state_dict(), save_path)



