import os
import yaml
import platform
from pathlib import Path
import numpy as np
import torch

import matplotlib.pyplot as plt

def readEncodings(path,object):
    completePath = Path(path+object+'/train/good/')
    heatMapDir = Path(completePath / "heatMaps/")
    createFolder(heatMapDir)

    for file in os.listdir(completePath):
        if not ".npy" in file:
            continue
        enc = np.load(Path(completePath / file))
        enc_32 = enc.astype(np.float32)
        tensor = torch.from_numpy(enc_32)
        print(">> file: ",file," shape -> ",tensor.shape)
        createAndSaveHeatMap(tensor,Path(heatMapDir/file.replace("npy","png")))


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
        print(">>Error {",folderPath,"} already exist")

def createAndSaveHeatMap(targetTensor, saveDir):
    # Plot the heatmap
    plt.imshow(targetTensor.squeeze().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar indicating the scale
    plt.savefig(saveDir)  # Save the heatmap to the specified path
    plt.close()  # Close the figure to prevent displaying it

if __name__ == '__main__':

    readEncodings("E:/mvtec_encodings/","bottle")