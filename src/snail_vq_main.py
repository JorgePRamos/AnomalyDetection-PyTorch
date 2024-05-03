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
    

def getPrediction(model, dataLoader, graphDiff = False):
    latestWeights = getLatestWeights(Path(workingDir + "/Results_Snail/"))
    loadWeights(model,latestWeights)
    runName = str(latestWeights).replace(".pkl","").split(os.path.sep)[-1]
    model.train(False)
    model.eval()

    
    resultsPath = dt.createResultsFolderStructure(runName)
    


    for i, (enc, label) in enumerate(dataLoader):
        
        enc = enc.to(device)
        prediction, _ = model(enc)
        prediction = prediction.to("cpu")
        _, prediction = prediction.max(1)
        
        for e, p, l in zip(enc,prediction,label):
            id = str(l).split(os.path.sep)[-1]
            
            if graphDiff:
                spc.showIncorrectPrediction(e,p,l)  
            print(">> Saving: ", id)
            dt.saveToNpy(p,Path(resultsPath / id))
        
    

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

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        

        _, pred = out.max(1)


        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        
        accuracy,loss = accuracy.to('cpu'), loss.to('cpu').detach().numpy()
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


    return np.mean(totalTrainAcc),np.mean(totalTrainLoss),np.mean(totalTrainLr)

def validate(model, dataLoader, device):
    print(">> Validation")
    
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
        
        accuracy, loss = accuracy.to('cpu'), loss.to('cpu').detach().numpy()
        # Append to total acc and loss
        totalValAcc.append(accuracy)
        totalValLoss.append(loss)
        
        # Tqdm loader
        dataLoader.set_description((f'Val_loss: {loss.item():.5f}; 'f'Val_acc: {accuracy:.5f};'))

    return np.mean(totalValAcc), np.mean(totalValLoss)


def test(model, dataLoader, graphTest):
    print(">> Beginning  testing")
    model.train(False)
    model.eval()
    getPrediction(model, dataLoader, graphTest)

def getLatestWeights(weightsDir):
    listFiles = glob.glob(str(weightsDir) + '/*.pkl')
    return max(listFiles, key=os.path.getctime)
    

def loadWeights(model, wghtsPath): 
    model.load_state_dict(torch.load(wghtsPath ))        
#def eval(model,):

if __name__ == '__main__':
    # arguments snail_vq_main.py -train True -wb True -save True

    device = "cuda"
    config = {
    "batchSize": 64,
    "epochs": 400,
    "scheduled": True,
    "lr": 0.0001,
    "inputDim": (16,16), # Input dim of the encoded
    "numClass": 256, # Num classes = possible pixel values
    "channels": 256, # Num channels intermediate feature representation
    "kernel": 5, # Kernel size
    "blocks": 4,
    "resBlocks": 4,
    "resChannels": 256,
    "attention": True,
    "dropout": 0.4,
    "condResChannels": 64, # Number of channels in the conditional ResNet
    "condResKernel": 4, # Size of the kernel in the conditional ResNet
    "condResBlocks": 4, # Number of conditional residual blocks in the conditional ResNet
    "outResBlock": 4 # Number of residual blocks in the output layer
    }
    parser = parser.parse_args()
    useWb = parser.wb
    saveWeights = parser.save
    trn = parser.train

    model = PixelSNAIL(config["inputDim"],
            config["numClass"],
            config["channels"],
            config["kernel"],
            config["blocks"],
            config["resBlocks"],
            config["resChannels"],
            config["attention"],
            config["dropout"],
            config["condResBlocks"],
            config["condResKernel"],
            config["outResBlock"])
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

    trainLoader = DataLoader(trainDataset,config["batchSize"], sampler = trainSampler, num_workers=4, drop_last=False)
    valLoader = DataLoader(trainDataset, config["batchSize"], sampler = validSampler,  num_workers=4, drop_last=False)

    model = model.to("cuda")
    
    trainingName = "default"
    wandbObject = None
    if useWb:
        wandbObject = wandb.init(project="PixelSnail-embeddings-VQ", config=config)
        trainingName = wandbObject.name
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_decay = 0.999995
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
 
    workingDir = os.getcwd()

    # Train
    if trn:
        print(">> Beginning  training")
        for epoch in range(config["epochs"]):
            # np.mean(totalTrainAcc), np.mean(totalTrainLoss), np.mean(totalTrainLr)
            trainAcc, trainLoss, trainLr = train(epoch, trainLoader, model, optimizer, scheduler, device)
            valAcc, valLoss, = validate(model, valLoader, device)
                    
            if wandbObject is not None:
                wandb.log({"epoch": epoch+1, "Loss Train": trainLoss, "Loss Val": valLoss,
                            "Acc Train": trainAcc, "Acc Val": valAcc, "Learning rate": trainLr})


        # Save weights
        if saveWeights: 
            torch.save(model.state_dict(), Path(workingDir + "/Results_Snail/" + trainingName+'.pkl'))
    

    
    # Load your eval dataset
    rootDir = Path("E:/mvtec_encodings/bottle/")
    evalDataset = EncodingsDataset(rootDir)

    # Eval
    testSet = EncodingsDataset(rootDir, train=False)
    testlDataLoader = DataLoader(testSet, batch_size=8, shuffle=False, num_workers=4)
    graphResults = False
    test(model, testlDataLoader, graphResults)
    