from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os


def load_data(data_dir = "./", batch_size = 1, train=True):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomGrayscale(p=0.3),
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    if train == True:
        image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['train'])
    else:
        image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['valid'])
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def resize_img(img, imgsize, min_size = 448, max_size = 448):
    W, H = imgsize
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = cv2.resize(img.permute(1,2,0).numpy(), (int(W*scale), int(H*scale)))
    return img, scale

def resize_box(bbox, in_size, out_size):
    bbox = np.array(bbox).copy()
    y_scale = float(out_size[0]) / int(in_size[0])
    x_scale = float(out_size[1]) / int(in_size[1])
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    return bbox

def Dataloader(dataloader, i):
    img = dataloader.dataset[i][0]
    path = dataloader.dataset.imgs[i][0]
    path = path[:-3]+'npz'
    bndbox = np.load(path) 
    boxlabel = torch.ones((len(bndbox['bndbox']), 1))
    #img, scale = resize_img(img, imgsize[:-1])
    #bndbox = resize_box(bndbox, imgsize[:-1], [scale*ele for ele in imgsize[:-1]])
    #return torch.from_numpy(img).permute(2,0,1).unsqueeze(0), torch.from_numpy(bndbox), boxlabel, flag, scale
    return img, bndbox['bndbox'], boxlabel

def Unnormalize_Orgsizeimg(img, orgsize):
    std = np.array([0.485, 0.456, 0.406])
    mean = np.array([0.229, 0.224, 0.225])
    H, W = img.shape[:2]
    img = cv2.resize(img, (orgsize[1], orgsize[0]))
    img = img*std + mean
    return img