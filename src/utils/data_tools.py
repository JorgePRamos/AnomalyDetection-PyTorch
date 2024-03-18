import os
import yaml
import platform
from pathlib import Path
import numpy as np
import torch

def testFunction(input, label, sEncoding):
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
            print(f"Folder created: {folderPath}")
        except OSError as e:
            print(f"Error: {folderPath} : {e.strerror}")
    else:
        print(">>Error {",folderPath,"} already exist")

def createDataSetFolderStructure(targetObject):
    
    targetObjectDataSetPath = Path(getDataSetLocation() +"mvtec_encodings/"+targetObject + "/train/good/")
    createFolder(targetObjectDataSetPath)
    return targetObjectDataSetPath


def saveToNpy(targetTensor,savePath):
    array = targetTensor.numpy()
    np.save(savePath, array)