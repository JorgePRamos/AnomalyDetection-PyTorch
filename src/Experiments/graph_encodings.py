"""
Graphical representation in the form of HeatMap of the extracted encodings from VQ-VAE
'encPath' Path where encodings are located and target object
"""
# For parent dir access in imports
import sys
sys.path.append('..')

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import data_tools as dt

def readPlotEncodings(path,object):
    completePath = Path(path+object+'/train/good/')
    heatMapDir = Path(completePath / "heatMaps/")
    dt.createFolder(heatMapDir)

    for file in os.listdir(completePath):
        if not ".npy" in file:
            continue
        enc = np.load(Path(completePath / file))
        enc_32 = enc.astype(np.float32)
        tensor = torch.from_numpy(enc_32).squeeze()
        print(">> file: ",file," shape -> ",tensor.shape)
        createAndSaveHeatMap(tensor,Path(heatMapDir/file.replace("npy","png")))


def createAndSaveHeatMap(targetTensor, saveDir):
    # Plot the heatmap
    plt.imshow(targetTensor.squeeze().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar indicating the scale
    plt.savefig(saveDir)  # Save the heatmap to the specified path
    plt.close()  # Close the figure to prevent displaying it

if __name__ == '__main__':
    encPath = "E:/mvtec_encodings/"
    object = "transistor"
    readPlotEncodings(encPath,object)