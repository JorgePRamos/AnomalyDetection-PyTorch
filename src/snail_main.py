import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
from models.pixelSnail import PixelSNAIL
from pathlib import Path
from tqdm import tqdm
#from scheduler import CycleScheduler


# Define your dataset class
class EncodingsDataset(Dataset):
    def __init__(self, rootDir, train = True):
        self.rootDir = rootDir
        if train:
            encDir = rootDir / r'train/'
        self.encDir = encDir
        self.encList = sorted(glob.glob(os.path.join(encDir, '**/*.npy')))

    def __len__(self):
        return len(self.encList)

    def __getitem__(self, idx):
        enc = np.load(self.encList[idx])

        return torch.from_numpy(enc),self.encList[idx]
    

# Training loop
def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        
        target = top
        out, _ = model(top)



        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
             


"""
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        epoch_loss = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

"""

if __name__ == '__main__':

    batchSize = 64
    epochs = 100
    scheduled = False
    lr = 0.01
    # Input dim of the encoded
    inputDim = [16, 16]
    # Num classes = possible pixel values
    numClass = 50
    # Num channels
    channels = 50
    # Kernel size
    kernel = 5
    blocks = 4
    resBlocks = 4
    resChannels = 50
    # Bottom False Top True
    attention = True
    dropout = 0.1
    # Number of conditional residual blocks in the conditional ResNet
    condResBlocks = 5
    # Number of channels in the conditional ResNet
    condResChannels = 50
    # Size of the kernel in the conditional ResNet
    condResKernel = 3 
    # Number of residual blocks in the output layer
    outResBlock = 0

    input("kkk")

    model = PixelSNAIL(inputDim,
            numClass,
            channels,
            kernel,
            blocks,
            resBlocks,
            resChannels,
            attention,
            dropout,
            condResBlocks,
            condResKernel,
            outResBlock)
    # Load your dataset
    rootDir = Path("E:/mvtec_encodings/bottle/")
    dataset = EncodingsDataset(rootDir)
    loader = DataLoader(dataset, batchSize, shuffle=True, num_workers=4, drop_last=True)

    model = model.to("cuda")


    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None

    # for i, (x, y) in enumerate(iter(loader)):
    #     print(x.shape)
    #     print(i)


    # test = dataset.__getitem__(0)
    
    #scheduler = CycleScheduler(optimizer,lr, n_iter=len(loader) * epochs, momentum=None)
    for i in range(epochs):
        train(i, loader, model, optimizer, scheduler, "cuda")
        torch.save(
            {'model': model.module.state_dict()},
            f'checkpoint/pixelsnail_{str(i + 1).zfill(3)}.pt',
        )