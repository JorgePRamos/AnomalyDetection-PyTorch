# For parent dir access in import
import sys
sys.path.append('..')

import os
import yaml
import platform
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import data_tools as dt



def showIncorrectPrediction(original, predicted, sampleName):
    original = original.to('cpu')
    sampleName = str(sampleName).split(os.path.sep)[-1]
    difference = original - predicted

    # Identify pixels¡
    incorrect_pixels = np.where(difference != 0)
    correct_pixels = np.where(difference == 0)

    # Create a visualization
    visualization = np.copy(original)  # Copy ground truth tensor
    visualization[incorrect_pixels] = 1
    visualization[correct_pixels] = 0
    
    # Terminal ASCII visualization
    termVis = False
    if termVis:
        print("\n ------------- \n", "-->",sampleName,"\n")
        print(visualization)
    
    

    plt.imshow(visualization, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label='0: Correct, 1: Incorrect')
    plt.title('['+sampleName+'] Error map')
    plt.show()