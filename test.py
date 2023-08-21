from UNet import UNet

import os
from typing import List

import numpy as np
from PIL import Image
from net.net import net

import torch
from torchvision import transforms
from loss import *


total_img_num = 28
if __name__ == "__main__":
    net = net()
    # 加载模型参数
    net.load_state_dict(torch.load('./models/epoch_60.pth'))
    torch.set_grad_enabled(False)
    net.eval()
    # 读取所有图片路径
    picture_list: List[torch.Tensor] = []
    image_max = total_img_num
    
    folder1 = './origin'.replace('\\', '/')
    folder2_list = os.listdir(folder1)
    for image in folder2_list:
        image_max -= 1
        if image_max < 0: break
        image_path = os.path.join(folder1, image).replace('\\', '/')
        raw_img = Image.open(image_path)
        test_image = transforms.ToTensor()(raw_img)
        picture_list.append(test_image)
        
    picture_list2: List[Image.Image] = []   
    image_max = total_img_num   
    folder3 = './ground'.replace('\\', '/')
    folder4_list = os.listdir(folder3)
    for image in folder4_list:
        image_max -= 1
        if image_max < 0: break
        image_path = os.path.join(folder3, image).replace('\\', '/')
        raw_img = Image.open(image_path)
        picture_list2.append(raw_img)
        
    psnr = []
    ssim = []
    image_max = total_img_num  
    # 遍历所有图片
    pic_psnr = 0
    pic_ssim = 0
    for i, test in enumerate(picture_list):
        # 保存结果地址
        folder_path = './test'.replace('\\', '/')
        file_name = f"image_{i+1}.png"
        file_path = os.path.join(folder_path, file_name).replace('\\', '/')
        # 读取图片
        img = picture_list[i]
        # 预测
        with torch.no_grad():
            L, R, X = net(img)
            D = img - X        
            I = torch.pow(L,0.14) * R

        L_img = transforms.ToPILImage()(L.squeeze(0).repeat(3, 1, 1))
        R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))                
        D_img = transforms.ToPILImage()(D.squeeze(0))  

        L_img.save(folder_path + '/L/' + file_name)
        R_img.save(folder_path + '/R/' + file_name)
        I_img.save(folder_path + '/I/' + file_name)  
        D_img.save(folder_path + '/D/' + file_name)
        p_psnr = calculate_psnr(I_img, picture_list2[i])
        p_ssim = calculate_ssim(I_img, picture_list2[i]) 
        pic_psnr += p_psnr
        pic_ssim += p_ssim
        print(f"picture {i+1}: psnr = {p_psnr:.3f}, ssim = {p_ssim:.3f}")
    print(f"psnr: {(pic_psnr/image_max):.3f}")
    print(f"ssim: {(pic_ssim/image_max):.3f}")   
    # psnr.append("{:.3f}".format(pic_psnr/image_max))
    # ssim.append("{:.3f}".format(pic_ssim/image_max))
    torch.set_grad_enabled(True)
        
    print("测试完成")
    print(psnr)
    print(ssim)