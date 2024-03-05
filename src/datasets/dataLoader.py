import albumentations as alb
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import glob
import os
import copy

from utils.helper import *


######################################################################################
#
# CLASS DESCRIBING THE DATALOADER FOR the MVTecAD dataset
# 
######################################################################################

class MVTecADDataset(Dataset):

    def __init__(self, rootDir, imgTransforms, toTensorTransform, grayTransform, customTransform, train): 
        self.train           = train 
        self.imgTransforms   = imgTransforms
        self.customTransform = customTransform
        self.grayTransform   = grayTransform
        self.toTensorTransform = toTensorTransform
        if train:
            imgDir  = os.path.join(rootDir, 'train')
            maskDir = None
        else:
            imgDir  = os.path.join(rootDir, 'test')
            maskDir = os.path.join(rootDir, 'ground_truth')
        self.imgList = sorted(glob.glob(os.path.join(imgDir, '**/*.png')))
        self.imgDir  = imgDir
        self.maskDir = maskDir
        

    def __getitem__(self, idx) -> Image.Image :
        
        # File Name
        fileName = os.path.split(self.imgList[idx])[-1]
        # Load images
        img = cv2.imread(self.imgList[idx])
        if self.grayTransform: 
            img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),-1)
        else: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load masks (if test) and generate labels
        if self.train: 
            imgCopy = copy.deepcopy(img)
            label   = 'clean'
        else : 
            folder, file = os.path.split(self.imgList[idx])[-2], os.path.split(self.imgList[idx])[-1]

            if folder == 'good': 
                mask  = np.zeros((img.shape[0], img.shape[1]))
                label = 'clean' 
            else: 
                #fileName = file.split('.png')[0] + '_mask.png'
                #folder = folder.replace("test","ground_truth")
                mask     = cv2.imread(os.path.join(self.maskDir, folder, file))

                mask     = np.expand_dims(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),-1) 
                label    = folder

        # apply transforms if any
        if self.imgTransforms: 
            augmented   = self.imgTransforms(image=img)
            augImg      = augmented['image']
        if self.train: # generate the synthetic corrupted image and mask
            corrImg = alb.ReplayCompose.replay(augmented['replay'], image=imgCopy)['image']
            if self.customTransform:
                corrImg = self.customTransform(corrImg)
            augMask = diff(corrImg, augImg) > 1
        else: 
            augMask = alb.ReplayCompose.replay(augmented['replay'], image=mask)['image']
            corrImg = np.zeros((img.shape[0], img.shape[1]))

        if self.toTensorTransform:  
            corrImg, originalImg = np.divide(corrImg,255).astype(np.float32), np.divide(augImg,255).astype(np.float32)
            corrImg              = self.toTensorTransform(image=corrImg)['image']
            originalImg          = self.toTensorTransform(image=originalImg)['image']
            augMask              = self.toTensorTransform(image=augMask)['image'] 

        return corrImg.float(), originalImg.float(), augMask, label

    def __len__(self) -> int:
        return len(self.imgList)


  

    