import os
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataloader import Dataset
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, file_path: str, f2_max: int = 99999, image_max: int = 99999) -> None:
        super().__init__()
        
        self.picture_list: List[torch.Tensor()] = []
        self.f2_max = f2_max
        self.image_max = image_max
        
        folder1 = 'high_sharp_scaled'
        folder1_path = os.path.join(file_path, folder1).replace('\\', '/')
        folder2_list = os.listdir(folder1_path)
        f2_max_stop = self.f2_max
        for folder2 in folder2_list:
            f2_max_stop -= 1
            if f2_max_stop < 0: break
            folder2_path = os.path.join(folder1_path, folder2).replace('\\', '/')
            image_list = os.listdir(folder2_path)
            max = self.image_max
            for image in image_list:
                max -= 1
                if max < 0: break
                image_path = os.path.join(folder2_path, image).replace('\\', '/')
                raw_img = Image.open(image_path)
                sharp_image = transforms.ToTensor()(raw_img)
                self.picture_list.append(sharp_image)
                
        self.picture_list2: List[torch.Tensor()] = []
        self.f2_max2 = f2_max
        self.image_max2 = image_max
        
        folder3 = 'low_blur_noise'
        folder3_path = os.path.join(file_path, folder3).replace('\\', '/')
        folder4_list = os.listdir(folder3_path)
        f2_max2_stop = self.f2_max2
        for folder4 in folder4_list:
            f2_max2_stop -= 1
            if f2_max2_stop < 0: break
            folder4_path = os.path.join(folder3_path, folder4).replace('\\', '/')
            image_list = os.listdir(folder4_path)
            max = self.image_max2
            for image2 in image_list:
                max -= 1
                if max < 0: break
                image_path2 = os.path.join(folder4_path, image2).replace('\\', '/')
                raw_img2 = Image.open(image_path2)
                sharp_image = transforms.ToTensor()(raw_img2)
                self.picture_list2.append(sharp_image)
    
    def __len__(self):
        return len(self.picture_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.picture_list2[index], self.picture_list[index]