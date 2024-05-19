import yaml
import numpy as np
import importlib
import copy
import shutil

from datasets.mvtec_dataLoader import *
from datasets.vqvae_dataAugmentation import *
from utils.helper import *
from utils.makeGraphs_RegularizedEmbeddingClass import *

import utils.data_tools as udt

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
from datasets import encodings_dataLoader as encdata

""" -----------------------------------------------------------------------------------------
NETWORK CLASS for VQVAE type networks 
----------------------------------------------------------------------------------------- """ 

def createFolder(desiredPath): 
    
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


class RegularizedEmbedding(Network_Class): 
    # ---------------------------------------------------------------------------------------
    # Initialization of class variables
    # ---------------------------------------------------------------------------------------
    def __init__(self, thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath):
        super(RegularizedEmbedding, self).__init__(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)


    # ---------------------------------------------------------------------------------------
    # Training & Validation procedures
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
            
        class getDecode(VQVAE): 
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):

                deconvLayers  = [None]*(self.depth-1)
                for i, decLayer in enumerate(self.decoder):
                    if i == 0:
                        #possible error: convLayers[-1] instead of x
                        loss, quantized, perplexity, encodings = self.quantization_module(x, snailEncodings = x )
                        deconvLayers[0] = decLayer(quantized)
                        
                    else: 
                        deconvLayers[i] = decLayer(deconvLayers[i-1])
                        
                out = self.Conv_1x1(deconvLayers[-1])
                out = self.sigmoid(out)

                return out

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

        return getEncodingsNet(self.dsConfig, self.modelConfig), getcustomPredNet(self.dsConfig, self.modelConfig), getTrainingEncodings(self.dsConfig, self.modelConfig), getDecode(self.dsConfig, self.modelConfig)

    # ---------------------------------------------------------------------------------------
    # Evaluation of the model
    # ---------------------------------------------------------------------------------------
    def getPrediction(self, dataLoader, resultPath, isTest=True): 

        self.getEncodings, self.newNet, self.snailEncodings, _ = self.getNetworks()
        # Network to get the (unmodified) encodings of the VQVAE
        self.getEncodings.to(self.device)  
        self.getEncodings.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))
        # Network to get the modified reconstruction from a modified encoding
        self.newNet.to(self.device)  
        self.newNet.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))

        # Network for encoding extraction
        self.snailEncodings.to(self.device)  
        self.snailEncodings.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))
        
        allInputs, allPreds, allLabels, allMasks = [], [], [], []
        allEncodings = []
        allQuantized = []
        allSnailEncodings = []
        self.getEncodings.train(False)
        self.getEncodings.eval()
        self.newNet.train(False)
        self.newNet.eval()

        self.snailEncodings.train(False)
        self.snailEncodings.eval()

        for (corrupted, image, mask, label) in dataLoader:
            if isTest:
                image = image.to(self.device)
            else: 
                image = corrupted.to(self.device)


            # Infer
            encodings,quantized = self.getEncodings(image)
            predictions = self.newNet(image)

            snailEncodings = self.snailEncodings(image)
            print(">>> #db getPred; snailEncodings shape: ", snailEncodings.shape)

            predictions = predictions.to(self.device)
            image, predictions, encodings, quantized,snailEncodings= image.to('cpu'), predictions.to('cpu'), encodings.to('cpu'),quantized.to('cpu'),snailEncodings.to('cpu')
            allInputs.extend(image.data.numpy())
            allLabels.extend(label)
            allPreds.extend(predictions.data.numpy())
            allMasks.extend(mask.data.numpy())

           
            # Save encodings and quantized embeddings for visualization
            allEncodings.extend(encodings.data.numpy())
            allQuantized.extend(quantized.data.numpy())
            allSnailEncodings.extend(snailEncodings.data.numpy())

        allInputs = np.multiply(np.array(allInputs),255).astype(np.uint8)
        allLabels = np.array(allLabels)
        allPreds  = np.multiply(np.array(allPreds),255).astype(np.uint8)
        allMasks  = np.array(allMasks).astype(np.uint8)
        
 
        allInputs = np.transpose(allInputs, (0,2,3,1))
        allPreds  = np.transpose(allPreds,  (0,2,3,1))
       
        allMasks  = np.transpose(allMasks,  (0,2,3,1))
        # H,W,C
        allSnailEncodings = np.transpose(allSnailEncodings,  (0,1,2,3))
        print(">>> #db getPred; AllSnailEncodings trans shape: ", allSnailEncodings.shape)
        return allInputs, allPreds, allLabels, allMasks, allEncodings,allQuantized,allSnailEncodings

    def extractEncodings(self, resultPath):
        
        trainTargetFolder, testTargetFolder = udt.createDataSetFolderStructure(os.path.split(resultPath)[-1])
        print(">> Created folder for encodings at: ",trainTargetFolder)
        
        # Get training Encodings
        print(">> Extraction of training data")
        trainInputs, trainPreds, trainLabels, trainMasks, trainEncodings, trainQuantized, trainSnailEncodings = self.getPrediction(
            self.trainDataLoader, resultPath, isTest=False)
        
        # Get val Encodings
        print(">> Extraction of val data")
        valInputs, valPreds, valLabels, valMasks, valEncodings, valQuantized, valSnailEncodings = self.getPrediction(
            self.valDataLoader, resultPath, isTest=False)
        
        # Concatenate to form train dataset
        allInputs = np.concatenate((trainInputs,valInputs))
        allLabels = np.concatenate((trainLabels,valLabels)) 
        allSnailEncodings = np.concatenate((trainSnailEncodings,valSnailEncodings)) 
        
        subsets = np.unique(allLabels)

        # Save training set snailEncodings to .npy
        for thisSubset in subsets: 
            
            sbt = allLabels==thisSubset
            for i, (input, label, sEncoding) in enumerate(zip(allInputs[sbt], allLabels[sbt], allSnailEncodings[sbt])):
                iter = '{:03}'.format(i)+".npy"
                print("Image_",iter)
                npyFilePath = Path(trainTargetFolder / iter)
                
                sEncoding  = np.transpose(sEncoding, (2, 0, 1))
                print(">>> #db extractEncodings; before squished: ", sEncoding.shape, " - Type: ",type(sEncoding))
                squished = np.argmax(sEncoding, axis = 0, keepdims = True)

                
                udt.saveToNpy(squished,npyFilePath)
                udt.encodingInfo(input,label,squished)
        
        # Get test Encodings
        print(">> Extraction of test data")

        testInputs, testPreds, testLabels, testMasks, testEncodings, testQuantized, testSnailEncodings = self.getPrediction(
        self.testlDataLoader, resultPath, isTest=True)
    
        
        
        testGood = []
        subsets = np.unique(testLabels)
        for thisSubset in subsets:
            if not "good" in thisSubset:
                continue
            
            sbt = testLabels==thisSubset
            for i, (input, label, sEncoding) in enumerate(zip(testInputs[sbt], testLabels[sbt], testSnailEncodings[sbt])):
                
                iter = '{:03}'.format(i)+".npy"
                print("Image_",iter)
                npyFilePath = Path(testTargetFolder / iter)
                sEncoding  = np.transpose(sEncoding, (2, 0, 1))
                squished = np.argmax(sEncoding, axis = 0, keepdims = True)
                testGood.append(sEncoding)
                udt.saveToNpy(squished,npyFilePath)
                udt.encodingInfo(input,label,squished)

            print(">> All encodings extracted")
            return testGood
            
            

    def evaluate(self, resultPath, printPrediction=False, wandbObj=None, printPredForPaper=False, decode = False): 
        self.model.train(False)
        self.model.eval()

        # Compute ROC Curves: anomaly map is diff between input and prediction
        allInputs, allPreds, allLabels, allMasks, allEncodings,allQuantized,allSnailEncodings = self.getPrediction(self.testlDataLoader, resultPath)
        allAM = []

        #Predict on training data for best encodings extraction
        self.extractEncodings(resultPath)
        #self.decodeEmbeddings(resultPath)

        for x, y in zip(allInputs, allPreds): 
            allAM.extend([diff(x,y)])
        self.computeROC(np.array(allAM), allLabels, allMasks, resultPath, printPrediction)

        # ---------------------------------------------------------------------------------------
        # Latent space visualizations
        # ---------------------------------------------------------------------------------------

        #self.visualizeSeparatedFeatureEncoding(allEncodings, allLabels, resultPath)
        """
        subsets = np.unique(allLabels)
        for thisSubset in subsets: 
            thisresultPath = resultPath / thisSubset / '_prediction/'
            #createFolder(thisresultPath)
            sbt = allLabels==thisSubset
            for i, (input, pred, quand) in enumerate(zip(allInputs, allPreds, allQuantized)):
                iter = str(i) +'.png'

                #self.visualizeFeatureEncoding(input, pred, quand, thisresultPath / iter)
        """        

  
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


    # ==== Deprecated
    # Visualize quantized features
    def visualizeSeparatedFeatureEncoding(self,allEncodings, allLabels, resultPath):
        num_samples = len(allEncodings)
        
        for i in range(num_samples):
            encoding = allEncodings[i]
            label = allLabels[i]

            # Determine the shape of the encoding
            height, width, num_channels = encoding.shape

            # Plot each channel separately
            fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
            for j in range(num_channels):
                axes[j].imshow(encoding[:, :, j], cmap='viridis')
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
    # =======


    def decodeEmbeddings(self,resultPath):   
        # Network for image decoding
        _, _, _, decodeModel = self.getNetworks()
        decodeModel.to(self.device)  
        decodeModel.load_state_dict(torch.load(resultPath / '_Weights/wghts.pkl'))
        decodeModel.train(False)
        decodeModel.eval()

        targetObject = os.path.split(resultPath)[-1]
        reconstructionTargetFolder = udt.createReconstructionResultsFolderStructure(targetObject)
        print(">> Created folder for reconstructed images at: ",reconstructionTargetFolder)
        print(">> Reconstructing data")


        


        # Bottle from snail
        #rootDir = Path("E:/snail_predictions/icy-sunset-92/")
        rootDir = Path("E:/mvtec_encodings/" + targetObject+r'/test/good/')
        
        # Laptop
        #rootDir = Path("C:/Users/jorge/Pictures/mvtec_encodings/" + targetObject)
        """
        tesDir = Path("E:/mvtec_encodings/" + targetObject+"/test/")
        encList = sorted(glob.glob(os.path.join(tesDir, '**/*.npy')))
        for i, enc in enumerate(encList):
            tempOg = np.transpose(originalEmbedings[i], (1, 2, 0))
            readEncoding = np.load(enc)
            oneHotEncodedTensor = udt.oneHotEncoding(torch.from_numpy(readEncoding), 256,1)

            hotSqueez = oneHotEncodedTensor.squeeze(0).numpy()
            print(">>>>>>>>>>>>> ogtensor transpose: ",tempOg.shape, " - ", type(tempOg))
            print(">>>>>>>>>>>>> fixed_HOT: ",hotSqueez.shape, " - ", type(hotSqueez))
            print(">>> #db equal; encodings [",i,"] equal = ", np.array_equal(hotSqueez,tempOg))"""



        testSet = encdata.EncodingsDataset(rootDir, train=False, vqvae=True)
        print(">>> TEST SET LEN: ", len(testSet)," Target Obj: ", targetObject)
        testlDataLoader = DataLoader(testSet, batch_size=8, shuffle=False, num_workers=4)



        for i, (enc, label) in enumerate(testlDataLoader):
            print(">>> read enc shape and type: ",enc.shape," and ",type(enc))
            batchSize = enc.shape[0]
            print(">>>  Used batch size: ", batchSize)
            oneHotEncodedTensor = udt.oneHotEncoding(enc, 256,batchSize)
            

            oneHotEncodedTensor = oneHotEncodedTensor.to("cuda")
            predictions = decodeModel(oneHotEncodedTensor.float())
            
            
            predictions = predictions.to("cpu")
            predictions_255 = np.multiply(np.array(predictions.data.numpy()),255).astype(np.uint8)
            print(">>> predictions_255 shape: ", predictions_255.shape)
            predictions_255 = np.transpose(predictions_255, (0,2,3,1))
            for p, l in zip(predictions_255,label):
                print(">>>> P SHAPE : ", p.shape)
                id = str(l).split(os.path.sep)[-1]
                udt.tensorToImage(p,reconstructionTargetFolder,id)