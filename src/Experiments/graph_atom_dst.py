"""
Graphical representation of distribution of the atoms selected to construct the encodings
from VQ-VAE
'encPath' Path where encodings are located and target object
"""
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
def lineOccurrence(source):
    # Extract classes and occurrences into separate lists for plotting
    classes = list(source.keys())
    occurrences = list(source.values())

    # Plotting the line chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(classes, occurrences, marker='o', linestyle='-')
    plt.xlabel('Class')
    plt.ylabel('Occurrences')
    plt.title('Occurrences per Class')
    plt.grid(True)
    plt.show()

def barOccurrence(source):
    classes = list(source.keys())
    occurrences = list(source.values())

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(classes, occurrences, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Occurrences')
    plt.title('Occurrences per Class')
    plt.grid(True)
    plt.show()


def pointOccurrence(source):
    classes = list(source.keys())
    occurrences = list(source.values())

    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.scatter(classes, occurrences, color='skyblue', marker='o')
    plt.xlabel('Class')
    plt.ylabel('Occurrences')
    plt.title('Occurrences per Class')
    plt.grid(True)
    plt.show()

def maxOccurrences(source,showNum):
    sorted_class_counts = sorted(source.items(), key=lambda x: x[1], reverse=True)
    print("Top "+ str(showNum) +" classes:")
    for i, (class_name, count) in enumerate(sorted_class_counts[:showNum], 1):
        print(f"{i}. {class_name}: {count}")

def readEncodings(path,object):
    enc_vals = dict.fromkeys(range(256), 0)
    completePath = Path(path+object+'/train/good/')
    
    for file in os.listdir(completePath):
        if not ".npy" in file:
            continue

        enc = np.load(Path(completePath / file))
        enc_reshape = enc.reshape(16, 16)
   

        for i in range(enc_reshape.shape[0]):
            for j in range(enc_reshape.shape[1]):
                enc_vals[enc_reshape[i,j]] += 1

    return enc_vals


                

     


       



if __name__ == '__main__':
    
    encPath = "E:/mvtec_encodings/"
    object = "bottle"
    dist = readEncodings(encPath,object)
    
    maxOccurrences(dist, 50)
    # Plot
    barOccurrence(dist)
    lineOccurrence(dist)
    pointOccurrence(dist)
