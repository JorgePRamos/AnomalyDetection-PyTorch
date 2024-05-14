import yaml
import numpy as np
import importlib
import copy

from datasets.mvtec_dataLoader import *
from datasets.vqvae_dataAugmentation import *
from utils.helper import *
from utils.makeGraphs_SuperClass import *

import torch
torch.manual_seed(2018)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchgeometry import losses
from torch import optim
from torchsummary import summary
from piq import ssim, psnr

from tqdm import tqdm
from termcolor import colored

import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# This class contains default procedures
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # --------------------------------------------------------------------------------
    def __init__(self, thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath):
        self.folderPath  = folderPath
        # Read the configuration files 
        self.imgCat      = thisImgCat
        self.expName     = thisDA + '-' + thisDS + '-' + thisModel + '-' + thisTrain
        datasetStream    = open('configs/dataset/' + thisDS + '.yaml', 'r')
        modelStream      = open('configs/model/' + thisModel + '.yaml', 'r')
        trainStream      = open('configs/train/' + thisTrain + '.yaml', 'r')
        dataAugStream    = open('configs/dataAugmentation/' + thisDA + '.yaml', 'r')
        self.dsConfig    = yaml.safe_load(datasetStream)
        self.modelConfig = yaml.safe_load(modelStream)
        self.trainConfig = yaml.safe_load(trainStream)
        self.DAConfig    = yaml.safe_load(dataAugStream)

        # Configure the dataset and the data augmentation
        imgTransforms, toTensorTransform, grayTransform, customTransform = getTransforms(self.dsConfig, self.DAConfig, train=True)

        # Training and Validation sets
        self.trainSet    = MVTecADDataset(folderPath, imgTransforms, toTensorTransform, grayTransform, customTransform, train=True)
        validationSplit  = 0.2
        datasetSize      = len(self.trainSet)
        indices          = list(range(datasetSize))
        split            = int(np.floor(validationSplit * datasetSize))
        np.random.seed(42)
        np.random.shuffle(indices)
        trainIdx, valIdx  = indices[split:], indices[:split]
        self.trainSampler = SubsetRandomSampler(trainIdx)
        self.validSampler = SubsetRandomSampler(valIdx)
        
        self.trainDataLoader = DataLoader(self.trainSet, batch_size=8, sampler=self.trainSampler, num_workers=4, pin_memory=True, drop_last=False)
        self.valDataLoader   = DataLoader(self.trainSet, batch_size=8, sampler=self.validSampler, num_workers=4, pin_memory=True, drop_last=False)

        # Training configuration
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        module           = importlib.import_module('models.' + thisModel.split('_')[0])
        cls              = getattr(module, thisModel.split('_')[0])
        self.model       = cls(self.dsConfig, self.modelConfig)
        self.model.to(self.device)
        optimMethod      = getattr(optim, self.trainConfig['Optimizer'])
        self.optimizer   = optimMethod(self.model.parameters(), lr=self.trainConfig['Learning_rate'])
        self.epoch       = self.trainConfig['Epoch']
        if self.trainConfig['LRScheduler'] == 'OneCycleLR':
            self.lrScheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, epochs=self.epoch, 
                                    steps_per_epoch = int(datasetSize/self.trainConfig['Batch_size']), anneal_strategy='linear')
        elif self.trainConfig['LRScheduler'] == 'ReduceLROnPlateau':
            self.lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, threshold=0.01, patience=10)
        else: 
            print(colored('Unsupported LR Scheduler', 'red'))
        self.criterion   = torch.nn.MSELoss()

        # Evaluation 
        imgTransforms, toTensorTransform, grayTransform, customTransform = getTransforms(self.dsConfig, self.DAConfig, train=False)
        self.testSet         = MVTecADDataset(folderPath, imgTransforms, toTensorTransform, grayTransform, customTransform, train=False)
        self.testlDataLoader = DataLoader(self.testSet, batch_size=8, shuffle=False, num_workers=0)


    # ---------------------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS 
    # ---------------------------------------------------------------------------------------
    def loadWeights(self, wghtsPath): 
        self.model.load_state_dict(torch.load(wghtsPath ))
    

    # ---------------------------------------------------------------------------------------
    # TRAINING AND VALIDATION PROCEDURES
    # ---------------------------------------------------------------------------------------
    def train(self, resultPath): 
        bestPSNR = 0
        allLoss, allPSNR, allSSIM = {'train': [], 'val': []}, {'train': [], 'val': []}, {'train': [], 'val': []}
        for i in range(self.epoch):
            trainLoss, trainPSNR, trainSSIM = self._train()
            valLoss, valPSNR, valSSIM       = self._validate()
            allLoss['train'].append(trainLoss)
            allLoss['val'].append(valLoss)
            allPSNR['train'].append(trainPSNR)
            allPSNR['val'].append(valPSNR)
            allSSIM['train'].append(trainSSIM)
            allSSIM['val'].append(valSSIM)

            print(colored( 'Epoch [%d/%d]' % (i+1, self.epoch), 'blue') )
            print(' '*5 + 'Train Loss: %.4f - Validation Loss: %.4f' % (trainLoss, valLoss))
            print(' '*5 + 'Train PSNR: %.4f - Validation PSNR: %.4f' % (trainPSNR, valPSNR))
            print(' '*5 + 'Train SSIM: %.4f - Validation SSIM: %.4f' % (trainSSIM, valSSIM))

            if valPSNR > bestPSNR:
                bestPSNR     = valPSNR
                bestModelWts = copy.deepcopy(self.model.state_dict())
                print( colored( ' '*5 + 'New Best Validation PSNR: %.4f \n' % (valPSNR), 'green') )
            else: 
                print(' '*5 + 'Old Best Validation PSNR: %.4f \n' % (bestPSNR))
            if self.trainConfig['LRScheduler'] == 'ReduceLROnPlateau':
                self.lrScheduler.step(valLoss)
            else:
                self.lrScheduler.step()

        createFolder(resultPath)
        printLearningCurves(allLoss, allPSNR, allSSIM, resultPath)

        wghtsPath  = resultPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(bestModelWts, wghtsPath + '/wghts.pkl')

    
    def _train(self): 
        self.model.train()
        batchIter   = tqdm(enumerate(self.trainDataLoader), 'Training', total=len(self.trainDataLoader), leave=True, 
                            ascii=' >=', bar_format='{desc:<7}{percentage:3.0f}%|{bar:20}{r_bar}')
        trainLosses = []
        trainPSNR, trainSSIM     = [], []
        for thisBatch, (corrImg, cleanImg, _, _) in batchIter:
            corrImg, cleanImg = corrImg.to(self.device), cleanImg.to(self.device)
            self.optimizer.zero_grad()
            # Results 
            outputs = self.model(corrImg)
            loss    = self.criterion(outputs, cleanImg)
            currentPSNR, currentSSIM = psnr(outputs, cleanImg), ssim(outputs, cleanImg)
            trainLosses.append(loss.item())
            trainPSNR.append(currentPSNR.item())
            trainSSIM.append(currentSSIM.item())
            # Backprop + optimize
            loss.backward()
            self.optimizer.step()
            # Print the log info
            batchIter.set_description('[%d/%d] Loss: %.4f' % (thisBatch+1, len(self.trainDataLoader), loss.item()))
        batchIter.close()
        return np.mean(trainLosses), np.mean(trainPSNR), np.mean(trainSSIM)


    def _validate(self): 
        self.model.eval()
        valLosses = []
        valPSNR, valSSIM     = [], []
        for _ in range(5):
            for (corrImg, cleanImg, _, _) in self.valDataLoader:
                corrImg, cleanImg = corrImg.to(self.device), cleanImg.to(self.device)
                with torch.no_grad(): 
                    outputs = self.model(corrImg)
                    loss    = self.criterion(outputs, cleanImg)
                    currentPSNR, currentSSIM = psnr(outputs, cleanImg), ssim(outputs, cleanImg)
                    valPSNR.append(currentPSNR.item())
                    valSSIM.append(currentSSIM.item())
                    valLosses.append(loss.item())
        return np.mean(valLosses), np.mean(valPSNR), np.mean(valSSIM)


    # ---------------------------------------------------------------------------------------
    # EVALUATION PROCEDURE
    # ---------------------------------------------------------------------------------------
    def getPrediction(self, dataLoader, isTest=True): 
        allInputs, allPreds, allLabels, allMasks = [], [], [], []
        for (corrupted, image, mask, label) in dataLoader:
            if isTest:
                image = image.to(self.device)
            else: 
                image = corrupted.to(self.device)
            predictions = self.model(image)
            predictions = predictions.to(self.device)

            image, predictions = image.to('cpu'), predictions.to('cpu')

            allInputs.extend(image.data.numpy())
            allLabels.extend(label)
            allPreds.extend(predictions.data.numpy())
            allMasks.extend(mask.data.numpy())

        allInputs = np.multiply(np.array(allInputs),255).astype(np.uint8)
        allLabels = np.array(allLabels)
        allPreds  = np.multiply(np.array(allPreds),255).astype(np.uint8)
        allMasks  = np.array(allMasks).astype(np.uint8)

        allInputs = np.transpose(allInputs, (0,2,3,1))
        allPreds  = np.transpose(allPreds,  (0,2,3,1))
        allMasks  = np.transpose(allMasks,  (0,2,3,1))
        return allInputs, allPreds, allLabels, allMasks


    def evaluate(self, resultPath, printPrediction=False): 
        self.model.train(False)
        self.model.eval()

        # Compute ROC Curves: anomaly map is diff between input and prediction
        allInputs, allPreds, allLabels, allMasks = self.getPrediction(self.testlDataLoader)
        allAM = []
        for x, y in zip(allInputs, allPreds): 
            allAM.extend([diff(x,y)])
        self.computeROC(np.array(allAM), allLabels, allMasks, resultPath, printPrediction)

        # Print predictions (LR)
        if printPrediction : 
            subsets = np.unique(allLabels)
            for thisSubset in subsets: 
                thisresultPath = resultPath + '/' + thisSubset + '_prediction/'
                createFolder(thisresultPath)
                sbt = allLabels==thisSubset
                for i, (input, pred, mask) in enumerate(zip(allInputs[sbt], allPreds[sbt], allMasks[sbt])):
                    printPredAndAM(input, pred, mask, thisresultPath + str(i) +'.png')

    def computeROC(self, allAM, allLabels, allMasks, resultPath, printPrediction): 
        # Compute Image-wise ROC curve
        anoMaps = np.concatenate( (allAM[allLabels=='clean'], allAM[allLabels!='clean']) )
        
        binary_test_labels = np.concatenate( (allLabels[allLabels=='clean'], allLabels[allLabels!='clean']) )
        binary_test_labels = binary_test_labels != 'clean'

        ImageWisePath      = resultPath / 'IMAGEWISE_ROC_clean_vs_realDefaults.json'
        all_TP, all_FP     = compute_ROC(anoMaps, binary_test_labels)
        write_json({'all_TP': all_TP, 'all_FP': all_FP}, ImageWisePath)
        
        # Compute Pixel-wise ROC curve
        masks         = np.concatenate( (allMasks[allLabels=='clean'], allMasks[allLabels!='clean']) )
        TP, FP        = compute_ROC_pixel_wise(anoMaps, masks)

        PixelWisePath = resultPath / 'PIXELWISE_ROC_clean_vs_realDefaults.json'
        write_json({'TP': TP, 'FP': FP}, PixelWisePath) 

        # Plot result
        print_ROC(ImageWisePath, PixelWisePath, resultPath / 'ROC_Curves.png')



        
        