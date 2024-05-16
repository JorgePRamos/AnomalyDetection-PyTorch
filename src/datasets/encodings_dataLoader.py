import glob
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class EncodingsDataset(Dataset):
    def __init__(self, rootDir, train = True, vqvae = False):
        self.vqvae = vqvae
        self.rootDir = rootDir
        if train:
            encDir = rootDir / r'train/'
        elif not train and not vqvae:
            encDir = rootDir / r'test/'
        else:
            encDir = rootDir
            print("#db inferloader: ", encDir)
        
        self.encDir = encDir

        if not vqvae:
            self.encList = sorted(glob.glob(os.path.join(encDir, '**/*.npy')))
        else:
            self.encList = sorted(glob.glob(os.path.join(encDir, '*.npy')))

    def __len__(self):
        return len(self.encList)

    def __getitem__(self, idx):
        enc = np.load(self.encList[idx])
        if not self.vqvae:
            return torch.from_numpy(enc).squeeze(),self.encList[idx]
        else:
            return torch.from_numpy(enc),self.encList[idx]

    
