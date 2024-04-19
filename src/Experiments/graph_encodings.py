"""
Graphical representation in the form of HeatMap of the extracted encodings from VQ-VAE
'encPath' Path where encodings are located and target object
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils.data_tools as dt

def readPlotEncodings(path,object):
    completePath = Path(path+object+'/train/good/')
    heatMapDir = Path(completePath / "heatMaps/")
    dt.createFolder(heatMapDir)

    for file in os.listdir(completePath):
        if not ".npy" in file:
            continue
        enc = np.load(Path(completePath / file))
        enc_32 = enc.astype(np.float32)
        tensor = torch.from_numpy(enc_32)
        print(">> file: ",file," shape -> ",tensor.shape)
        createAndSaveHeatMap(tensor,Path(heatMapDir/file.replace("npy","png")))


def createAndSaveHeatMap(targetTensor, saveDir):
    # Plot the heatmap
    plt.imshow(targetTensor.squeeze().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar indicating the scale
    plt.savefig(saveDir)  # Save the heatmap to the specified path
    plt.close()  # Close the figure to prevent displaying it

if __name__ == '__main__':
    encPath = "E:/mvtec_encodings/",
    object = "bottle"
    readPlotEncodings(encPath,object)