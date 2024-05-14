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
    np.save(savePath, targetTensor)


def oneHotEncoding(targetTensor, numClass, batchSize):
    flattened = targetTensor.view(batchSize, -1)  # Shape: (8, 256)
    one_hot = torchFunc.one_hot(flattened, num_classes=numClass)  # Shape: (8, 256, 256)
    output_tensor = one_hot.view(batchSize, 256, 16, 16)  # Shape: (8, 256, 16, 16)
    
    return output_tensor

def tensorToImage(targetTensor, targetFolder,label):
    # Step 2: Convert the tensor to a NumPy array
    targetTensor = targetTensor.squeeze(0).to("cpu")
    imageArray = targetTensor.detach().numpy()
    print("-------------------------------")
    print(imageArray)
    print("-------------------------------\n")

    # Step 3: Scale the array values to the range [0, 255]
    imageArray = (imageArray * 255).astype(np.uint8)

    # Step 4: Create a PIL Image from the NumPy array
    image = pilImage.fromarray(imageArray)


    # Save the image
    image.save(Path(targetFolder / f"{label}.png"))
    print(">> Saved at: ",str(targetFolder) + f"/{label}.png")
