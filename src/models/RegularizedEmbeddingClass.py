import yaml
import numpy as np
import importlib
import copy
import shutil

from datasets.dataLoader import *
from datasets.dataAugmentation import *
from utils.helper import *
from utils.makeGraphs_RegularizedEmbeddingClass import *

from models.SuperClass import Network_Class

import torch
import torch.nn as nn
torch.manual_seed(2018)
from torch.utils.data import DataLoader

from tqdm import tqdm
from termcolor import colored
from piq import ssim, psnr

from pathlib import Path

import os
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb


""" -----------------------------------------------------------------------------------------
NETWORK CLASS for VQVAE type networks 
----------------------------------------------------------------------------------------- """ 

def createFolder(desiredPath): 
    
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


class RegularizedEmbedding(Network_Class): 
    # ---------------------------------------------------------------------------------------
    # Initialisation of class variables
    # ---------------------------------------------------------------------------------------
    def __init__(self, thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath):
        super(RegularizedEmbedding, self).__init__(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)


    # ---------------------------------------------------------------------------------------
    # Traning & Validation procedures
    # ---------------------------------------------------------------------------------------
    def train(self, resultPath, wandbObj): 
        bestPSNR = 0
        allLoss,allLossMSE, allPSNR, allSSIM = {'train': [], 'val': []}, {'train': [], 'val': []}, {'train': [], 'val': []}, {'train': [], 'val': []}
        for i in range(self.epoch):
            trainLoss, trainLossMSE, trainPSNR, trainSSIM = self._train()
            valLoss, valLossMSE, valPSNR, valSSIM         = self._validate()
            allLoss['train'].append(trainLoss)
            allLoss['val'].append(valLoss)
            allLossMSE['train'].append(trainLossMSE)
            allLossMSE['val'].append(valLossMSE)
            allPSNR['train'].append(trainPSNR)
            allPSNR['val'].append(valPSNR)
            allSSIM['train'].append(trainSSIM)
            allSSIM['val'].append(valSSIM)
            
            if wandbObj is not None:
                wandb.log({"epoch": i, "Loss Train": trainLoss, "Loss Val": valLoss, 
                            "PSNR Train": trainPSNR, "PSNR Val": valPSNR,
                            "SSIM Train": trainSSIM, "SSIM Val": valSSIM})

            print(colored( 'Epoch [%d/%d]' % (i+1, self.epoch), 'blue') )
            print(' '*5 + 'Train Loss: %.4f - Validation Loss: %.4f' % (trainLoss, valLoss))
            print(' '*5 + 'Train LossMSE: %.4f - Validation LossMSE: %.4f' % (trainLossMSE, valLossMSE))
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
        printLearningCurves(allLoss, allLossMSE, allPSNR, allSSIM, resultPath)

        wghtsPath  = resultPath / '_Weights'
        createFolder(wghtsPath)
        torch.save(bestModelWts, wghtsPath / 'wghts.pkl')

    
    def _train(self): 
        self.model.train()
        batchIter   = tqdm(enumerate(self.trainDataLoader), 'Training', total=len(self.trainDataLoader), leave=True, 
                            ascii=' >=', bar_format='{desc:<7}{percentage:3.0f}%|{bar:20}{r_bar}')
        trainLosses    = []
        trainLossesMSE = []
        trainPSNR, trainSSIM     = [], []
       
        for thisBatch, (corrImg, cleanImg, _, _) in batchIter:
            corrImg, cleanImg = corrImg.to(self.device), cleanImg.to(self.device)
            self.optimizer.zero_grad()
            # Results 
            outputs, lossVQVAE, quantized_embeddings = self.model(corrImg)
            lossMSE            = self.criterion(outputs, cleanImg)
            loss               = lossVQVAE + lossMSE
            currentPSNR, currentSSIM = psnr(outputs, cleanImg), ssim(outputs, cleanImg)
            trainLosses.append(loss.item())
            trainLossesMSE.append(lossMSE.item())
            trainPSNR.append(currentPSNR.item())
            trainSSIM.append(currentSSIM.item())
            
           
            
            # Backprop + optimize
            loss.backward()
            self.optimizer.step()
            # Print the log info
            batchIter.set_description('[%d/%d] Loss: %.4f' % (thisBatch+1, len(self.trainDataLoader), loss.item()))
        batchIter.close()
        
        return np.mean(trainLosses), np.mean(trainLossesMSE), np.mean(trainPSNR), np.mean(trainSSIM)


    def _validate(self): 
        self.model.eval()
        valLosses = []
        valLossesMSE = []
        valPSNR, valSSIM     = [], []
        for _ in range(5):
            for (corrImg, cleanImg, _, _) in self.valDataLoader:
                corrImg, cleanImg = corrImg.to(self.device), cleanImg.to(self.device)
                with torch.no_grad(): 
                    outputs, lossVQVAE, quantized_embeddings = self.model(corrImg)
                    lossMSE            = self.criterion(outputs, cleanImg)
                    loss               = lossVQVAE + lossMSE
                    currentPSNR, currentSSIM = psnr(outputs, cleanImg), ssim(outputs, cleanImg)
                    valPSNR.append(currentPSNR.item())
                    valSSIM.append(currentSSIM.item())
                    valLosses.append(loss.item())
                    valLossesMSE.append(lossMSE.item())
        return np.mean(valLosses), np.mean(valLossesMSE), np.mean(valPSNR), np.mean(valSSIM)

    # ---------------------------------------------------------------------------------------
    # Auxiliary networks
    # ---------------------------------------------------------------------------------------

    def getNetworks(self): 
        from models.VQVAE import VQVAE

        class getTrainingEncodings(VQVAE): 
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x, modifiedEncodings=None):
                convLayers    = [None]*(self.depth)
                convLayers[0] = self.e1(x)
                for i, encLayer in enumerate(self.encoder):
                    convLayers[i+1] = encLayer(convLayers[i])
                loss, quantized, perplexity, encodings = self.quantization_module(convLayers[-1])
                return encodings
        class getEncodingsNet(VQVAE): 
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                convLayers    = [None]*(self.depth)
                convLayers[0] = self.e1(x)
                for i, encLayer in enumerate(self.encoder):
                    convLayers[i+1] = encLayer(convLayers[i])
                loss, quantized, perplexity, encodings = self.quantization_module(convLayers[-1])
                return encodings, quantized

        class getcustomPredNet(VQVAE): 
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x, modifiedEncodings=None):
                convLayers    = [None]*(self.depth)
                convLayers[0] = self.e1(x)
                for i, encLayer in enumerate(self.encoder):
                    convLayers[i+1] = encLayer(convLayers[i])

                deconvLayers  = [None]*(self.depth-1)
                for i, decLayer in enumerate(self.decoder):
                    if i == 0: 
                        loss, quantized, perplexity, encodings = self.quantization_module(convLayers[-1], modifiedEncodings)
                        deconvLayers[0] = decLayer(quantized)
                    else: 
                        deconvLayers[i] = decLayer(deconvLayers[i-1])
                        
                out = self.Conv_1x1(deconvLayers[-1])
                out = self.sigmoid(out)

                return out

        return getEncodingsNet(self.dsConfig, self.modelConfig), getcustomPredNet(self.dsConfig, self.modelConfig)

    # ---------------------------------------------------------------------------------------
    # Evaluation of the model
    # ---------------------------------------------------------------------------------------
    def getPrediction(self, dataLoader, resultPath, isTest=True): 

        self.getEncodings, self.newNet = self.getNetworks()
        # Network to get the (unmodified) encodings of the VQVAE
        self.getEncodings.to(self.device)  
        self.getEncodings.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))
        # Network to get the modified reconstruction from a modified encoding
        self.newNet.to(self.device)  
        self.newNet.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))

        allInputs, allPreds, allLabels, allMasks = [], [], [], []
        allEncodings = []
        allQuantized = []
        self.getEncodings.train(False)
        self.getEncodings.eval()
        self.newNet.train(False)
        self.newNet.eval()
        for (corrupted, image, mask, label) in dataLoader:
            if isTest:
                image = image.to(self.device)
            else: 
                image = corrupted.to(self.device)

            encodings,quantized = self.getEncodings(image)
            predictions = self.newNet(image)
            predictions = predictions.to(self.device)

           
            image, predictions, encodings, quantized= image.to('cpu'), predictions.to('cpu'), encodings.to('cpu'),quantized.to('cpu')
            allInputs.extend(image.data.numpy())
            allLabels.extend(label)
            allPreds.extend(predictions.data.numpy())
            allMasks.extend(mask.data.numpy())

            # Save encodings and quantized embeddings for visualization
            allEncodings.extend(encodings.data.numpy())
            allQuantized.extend(quantized.data.numpy())

        allInputs = np.multiply(np.array(allInputs),255).astype(np.uint8)
        allLabels = np.array(allLabels)
        allPreds  = np.multiply(np.array(allPreds),255).astype(np.uint8)
        allMasks  = np.array(allMasks).astype(np.uint8)

        allInputs = np.transpose(allInputs, (0,2,3,1))
        allPreds  = np.transpose(allPreds,  (0,2,3,1))
        allMasks  = np.transpose(allMasks,  (0,2,3,1))
        return allInputs, allPreds, allLabels, allMasks, allEncodings,allQuantized


    def evaluate(self, resultPath, printPrediction=False, wandbObj=None, printPredForPaper=False): 
        self.model.train(False)
        self.model.eval()

        # Compute ROC Curves: anomaly map is diff between input and prediction
        allInputs, allPreds, allLabels, allMasks, allEncodings,allQuantized = self.getPrediction(self.testlDataLoader, resultPath)
        allAM = []

        for x, y in zip(allInputs, allPreds): 
            allAM.extend([diff(x,y)])
        self.computeROC(np.array(allAM), allLabels, allMasks, resultPath, printPrediction)

        # ---------------------------------------------------------------------------------------
        # Latent space visualizations
        # ---------------------------------------------------------------------------------------

        #self.visualizeSeparatedFeatureEncoding(allEncodings, allLabels, resultPath)

        subsets = np.unique(allLabels)
        for thisSubset in subsets: 
            thisresultPath = resultPath / thisSubset / '_prediction/'
            #createFolder(thisresultPath)
            sbt = allLabels==thisSubset
            for i, (input, pred, quand) in enumerate(zip(allInputs, allPreds, allQuantized)):
                iter = str(i) +'.png'

                self.visualizeFeatureEncoding(input, pred, quand, thisresultPath / iter)
        
                
        # Print predictions (LR)
        if printPrediction : 
            subsets = np.unique(allLabels)
            for thisSubset in subsets: 
                thisresultPath = resultPath / thisSubset / '_prediction/'
                createFolder(thisresultPath)
                sbt = allLabels==thisSubset
                for i, (input, pred, mask) in enumerate(zip(allInputs[sbt], allPreds[sbt], allMasks[sbt])):
                    iter = str(i) +'.png'
                    printPredAndAM(input, pred, mask, thisresultPath / iter)

        # Print predictions for paper (HR)
        if printPredForPaper: 
            idx = [1,5,9]
            thisprintPath = resultPath / '_ImgsForPaper'
            createFolder(thisprintPath)
            for i, (input, pred) in enumerate(zip(allInputs[idx], allPreds[idx])):
                printPredAndAM_singleFile(input, pred, thisprintPath  / 'Test_' + str(idx[i]), Vmin=0, Vmax=1)


    # Visualize quantized features
    def visualizeSeparatedFeatureEncoding(self,allEncodings, allLabels, resultPath):
        num_samples = len(allEncodings)
        #(16, 16, 50)
        for i in range(num_samples):
            encoding = allEncodings[i]
            label = allLabels[i]

            # Determine the shape of the encoding
            height, width, num_channels = encoding.shape

            # Plot each channel separately
            fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
            for j in range(num_channels):
                axes[j].imshow(encoding[:, :, j], cmap='viridis')  # Choose a suitable colormap
                axes[j].set_title([j+1])
                axes[j].axis('off')

            plt.suptitle(f"Feature Encoding (Label: {label})")
            #plt.tight_layout()
            #plt.savefig(resultPath / f'feature_encoding_{i}.png')
            plt.show()
    # Visualize quantized embedigs
    def visualizeFeatureEncoding(self, input, pred, quand, thisresultPath):

        print("shape quad ----> ",quand.shape)
        mean_image = np.mean(quand, axis=0)
        plt.imshow(mean_image, cmap='gray')  # Assuming encoding is a grayscale image
        plt.title('Quantized embed')
        plt.colorbar()
        plt.show()