import os
import yaml
import platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as torchFunc
import torchvision.transforms as transforms
from PIL import Image as pilImage

def encodingInfo(input, label, sEncoding):
    print(">>> Input: ", input.shape," Label: ", label," sEncoding: ",sEncoding.shape)

def getDataSetLocation():
    hostName = platform.node()
    serverPath    = yaml.safe_load(r'configs\server\datasetLocation.yaml')
    serverCfg    = yaml.safe_load(open(serverPath, 'r'))

    dataSetLocation   = Path(serverCfg['DatasetLocation']['local_folder'])
    if hostName == "Bajoo" or hostName == "Betelgeuse":
        dataSetLocation   = Path(serverCfg['DatasetLocation']['remote_folder'])
    elif hostName == "DESKTOP-7B1KVSF":
        dataSetLocation   = Path(serverCfg['DatasetLocation']['local_folder_laptop'])

    return "/".join(os.path.split(dataSetLocation)[:-1])

def createFolder(folderPath, overwrite = True):
    # Check if the folder already exists
    if overwrite:
        if os.path.exists(folderPath):
            # If it exists, remove it
            try:
                os.rmdir(folderPath)
            except OSError as e:
                print(f"Error: {folderPath} : {e.strerror}")

    # Create the folder
    if not os.path.exists(folderPath):
        try:
            os.makedirs(folderPath)
        except OSError as e:
            print(f"Error: {folderPath} : {e.strerror}")
    else:
        print(">> Wanning {",folderPath,"} already exist")

def createDataSetFolderStructure(targetObject):
    
    targetTrainObjectDataSetPath = Path(getDataSetLocation() +"/mvtec_encodings/"+targetObject + "/train/good/")
    createFolder(targetTrainObjectDataSetPath)
    
    targetTestObjectDataSetPath = Path(getDataSetLocation() +"/mvtec_encodings/"+targetObject + "/test/good/")
    createFolder(targetTestObjectDataSetPath)

    return targetTrainObjectDataSetPath, targetTestObjectDataSetPath

def createSnailResultsFolderStructure(runName):
    resultsPath = Path(getDataSetLocation() +"/snail_predictions/"+runName + "/")
    createFolder(resultsPath)
    return resultsPath

def createReconstructionResultsFolderStructure(expName):
    resultsPath = Path(getDataSetLocation() +"/image_reconstruction/"+expName + "/")
    createFolder(resultsPath)
    return resultsPath


def saveToNpy(targetTensor,savePath):
    print(">>> #db npy to be saved: ", targetTensor.shape )
    
    np.save(savePath, targetTensor)


def oneHotEncoding(targetTensor, numClass, batchSize):
    """
    flattened = targetTensor.view(batchSize, -1)  # Shape: (batchSize, 256)
    one_hot = torchFunc.one_hot(flattened, num_classes=numClass)  # Shape: (batchSize, 256, 256)
    output_tensor = one_hot.view(batchSize, 256, 16, 16)  # Shape: (batchSize, 256, 16, 16)
    
    return output_tensor
    """

    targetTensor = targetTensor.squeeze(1) 


    one_hot_tensor = torch.nn.functional.one_hot(targetTensor.long(), num_classes=numClass)



    print(">>> #db oneHotEnc; shape of oneHotEnc: ", one_hot_tensor.shape)
    return one_hot_tensor

def tensorToImage(imageArray, targetFolder,label):
    
    print(">>>  Image array pre squeeze: ", imageArray.shape)
    imageArray = np.squeeze(imageArray, axis=-1)
    print(">>>  Image array post squeeze: ", imageArray.shape)

    print("Max: ",np.max(imageArray), " | min: ",np.min(imageArray))
    print("-------------------------------")
    print(imageArray)
    print("-------------------------------\n")
    
    image = pilImage.fromarray(imageArray)
    # Save the image
    label = label.replace(".npy","")
    image.save(Path(targetFolder / f"{label}.png"))
    print(">> Saved at: ",str(targetFolder) + f"/{label}.png")