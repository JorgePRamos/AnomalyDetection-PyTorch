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



def showIncorrectPrediction(original, predicted):
    original,predicted = original.to('cpu'), predicted.to('cpu')

    difference = original - predicted

    # Identify incorrect pixels¡
    incorrect_pixels = np.where(difference != 0)

    # Create a visualization
    visualization = np.copy(original)  # Copy ground truth tensor
    visualization[incorrect_pixels] = 2   # Set incorrect pixels to a different value (e.g., 2)
    plt.imshow(visualization, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='0: Correct, 1: Incorrect')
    plt.title('Visualization of Incorrect Pixels')
    plt.show()