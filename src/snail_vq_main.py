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
import argparse
from Experiments import predict_comparison as spc
from utils import data_tools as dt
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('-train', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-wb', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-save', default=True, type=lambda x: (str(x).lower() == 'true'))


# Define your dataset class
class EncodingsDataset(Dataset):
    def __init__(self, rootDir, train = True):
        self.rootDir = rootDir
        if train:
            encDir = rootDir / r'train/'
        else:
            encDir = rootDir / r'test/'

        self.encDir = encDir
        self.encList = sorted(glob.glob(os.path.join(encDir, '**/*.npy')))

    def __len__(self):
        return len(self.encList)

    def __getitem__(self, idx):
        enc = np.load(self.encList[idx])
        
        return torch.from_numpy(enc).squeeze(),self.encList[idx]
    

def getPrediction(model, dataLoader):
    latestWeights = getLatestWeights(Path(workingDir + "/Results_Snail/"))
    loadWeights(model,latestWeights)
    model.train(False)
    model.eval()
    resultsPath = dt.createResultsFolderStructure(str(latestWeights).replace(".pkl",""))

    for i, (enc, label) in enumerate(dataLoader):
        print(">> proc: ",i)
        enc = enc.to(device)
        prediction, _ = model(enc)
        dt.saveToNpy(prediction,resultsPath)
    

# Training loop
def train(epoch, dataLoader, model, optimizer, scheduler, device):
    dataLoader = tqdm(dataLoader)

    criterion = nn.CrossEntropyLoss()
    totalTrainAcc, totalTrainLoss, totalTrainLr= [], [], []
    for i, (enc, label) in enumerate(dataLoader):
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
        """
        if epoch % 100 == 0:

            wantedSamples = 10
            cnt = 1
            for predSamp, originalSamp, sampName in zip(pred,target,label):
                if cnt == wantedSamples:
                    break
                spc.showIncorrectPrediction(originalSamp,predSamp,sampName)
                cnt += 1
        """

        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        
        # Append to total acc and loss
        lr = optimizer.param_groups[0]['lr']
        totalTrainAcc.append(accuracy)
        totalTrainLoss.append(loss)
        totalTrainLr.append(lr)
        
        # Tqdm loader
        dataLoader.set_description(
            (
                f'epoch: {epoch + 1}; Train_loss: {loss.item():.5f}; '
                f'Train_acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
    return np.mean(totalTrainAcc), np.mean(totalTrainLoss), np.mean(totalTrainLr)

def validate(model, dataLoader, device):
    model.eval()
    dataLoader = tqdm(dataLoader)

    criterion = nn.CrossEntropyLoss()

    totalValAcc, totalValLoss = [], []
    for i, (enc, label) in enumerate(dataLoader):

        enc = enc.to(device)
        target = enc
        out, _ = model(enc)
        loss = criterion(out, target)
        
        # Accuracy calc
        _, pred = out.max(1)

        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        
        # Append to total acc and loss
        totalValAcc.append(accuracy)
        totalValLoss.append(loss)
        
        # Tqdm loader
        dataLoader.set_description((f'Val_loss: {loss.item():.5f}; 'f'Val_acc: {accuracy:.5f};'))

    return np.mean(totalValAcc), np.mean(totalValLoss)


def test(model, dataLoader):
    print(">> Beginning  testing")
    model.train(False)
    model.eval()
    getPrediction(model, dataLoader)

def getLatestWeights(weightsDir):
    listFiles = glob.glob(str(weightsDir) + '/*.pkl')
    return max(listFiles, key=os.path.getctime)
    

def loadWeights(model, wghtsPath): 
    model.load_state_dict(torch.load(wghtsPath ))        
#def eval(model,):

if __name__ == '__main__':
    # arguments snail_vq_main.py -train True -wb True -save True
    device = "cuda"
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

    parser = parser.parse_args()
    useWb = parser.wb
    saveWeights = parser.save
    trn = parser.train

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
    # Load your train dataset
    rootDir = Path("E:/mvtec_encodings/bottle/")
    trainDataset = EncodingsDataset(rootDir)
    validationSplit  = 0.2
    datasetSize      = len(trainDataset)
    indices          = list(range(datasetSize))
    split            = int(np.floor(validationSplit * datasetSize))
    trainIdx, valIdx  = indices[split:], indices[:split]
    trainSampler = SubsetRandomSampler(trainIdx)
    validSampler = SubsetRandomSampler(valIdx)

    trainLoader = DataLoader(trainDataset, batchSize, sampler = trainSampler, num_workers=4, drop_last=True)
    valLoader = DataLoader(trainDataset, batchSize, sampler = validSampler,  num_workers=4, drop_last=True)

    model = model.to("cuda")
    
    trainingName = "default"
    wandbObject = None
    if useWb:
        wandbObject = wandb.init(project="PixelSnail-embeddings-VQ")
        trainingName = wandbObject.name
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_decay = 0.999995
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
 
    workingDir = os.getcwd()
    
    # Train
    if trn:
        print(">> Beginning  training")
        for epoch in range(epochs):
            # np.mean(totalTrainAcc), np.mean(totalTrainLoss), np.mean(totalTrainLr)
            trainAcc, trainLoss, trainLr = train(epoch, trainLoader, model, optimizer, scheduler, device)
            valAcc, valLoss, = validate(model, valLoader, device)
                    
            if wandbObject is not None:
                wandb.log({"epoch": epoch+1, "Loss Train": trainLoss, "Loss Val": valLoss,
                            "Acc Train": trainAcc, "Acc Val": valAcc, "Learning rate": trainLr})



        if saveWeights: 
            torch.save(model.state_dict(), Path(workingDir + "/Results_Snail/" + trainingName+'.pkl'))
    

    
    # Load your eval dataset
    rootDir = Path("E:/mvtec_encodings/bottle/")
    evalDataset = EncodingsDataset(rootDir)

    # Eval
    testSet = EncodingsDataset(rootDir, train=False)
    testlDataLoader = DataLoader(testSet, batch_size=8, shuffle=False, num_workers=4)
    test(model, testlDataLoader)
    