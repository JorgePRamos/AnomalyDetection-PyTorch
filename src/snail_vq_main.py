import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
from models.vqPixelSnail import PixelSNAIL
from pathlib import Path
from tqdm import tqdm
import wandb
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
        
        return torch.from_numpy(enc).squeeze(),torch.from_numpy(enc).squeeze()
    

# Training loop
def train(epoch, loader, model, optimizer, scheduler, device,wandbObj):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (enc, label) in enumerate(loader):
        model.zero_grad()

        enc = enc.to(device)

        
        target = enc
        out, _ = model(enc)



        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']
        
        if wandbObj is not None:
            wandb.log({"epoch": epoch+1, "Loss Train": loss,
                        "Acc Train": accuracy,"Learning rate": lr})
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
    epochs = 220
    scheduled = True
    lr = 0.01
    # Input dim of the encoded
    inputDim = (16,16)
    # Num classes = possible pixel values
    numClass = 256
    # Num channels intermediate feature representation
    channels = 128 
    # Kernel size
    kernel = 5
    blocks = 2 #default
    resBlocks = 4
    resChannels = 128
    # Bottom False Top True
    attention = True
    dropout = 0.1
    # Number of channels in the conditional ResNet
    condResChannels = 0 #default
    # Size of the kernel in the conditional ResNet
    condResKernel = 3 #default

    # Number of conditional residual blocks in the conditional ResNet
    condResBlocks = 0 #default
    # Number of residual blocks in the output layer
    outResBlock = 0 #default


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

    wandbObject = wandb.init(project="PixelSnail embeddings ROS")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_decay = 0.999995
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
 
    workingDir = os.getcwd()
  
    for i in range(epochs):
        train(i, loader, model, optimizer, scheduler, "cuda",wandbObject)
        torch.save(model.state_dict(), Path(workingDir + "/pixSnailResults/checkpoint/" + f'/mnist_{str(i + 1).zfill(3)}.pt'))