import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from utils.helper import *

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rcParams['font.size'] = 16 


colors4 = ['gold','greenyellow','steelblue','midnightblue']
colors5 = ['gold','greenyellow','mediumaquamarine','steelblue','midnightblue']

# ---------------------------------------------------------------------------------------
# Print Learning cruves
# ---------------------------------------------------------------------------------------
def printLearningCurves(allLoss, allPSNR, allSSIM, curvesPath=None): 
    colors = {'Loss': 'darkcyan', 'Metrics':'crimson'}
    fig, ax = plt.subplots(3, figsize=(10,10), dpi=300, sharex=True)
    # First Line (Loss)
    ax[0].set_ylabel('MSE Loss')
    ax[0].plot(range(1,len(allLoss['train'])+1), allLoss['train'], color=colors['Loss'], label='Training', linewidth=1.5)
    ax[0].plot(range(1,len(allLoss['val'])+1), allLoss['val'], color=colors['Loss'], linestyle='dashed', label='Validation', linewidth=1.5)
    ax[0].grid(linestyle='-', linewidth=0.7)
    ax[0].legend(loc='upper right', prop={'size': 12})

    # Second Line (PSNR)
    ax[1].set_ylabel('PSNR')
    ax[1].plot(range(1,len(allPSNR['train'])+1), allPSNR['train'], color=colors['Metrics'], label='Training', linewidth=1.5)
    ax[1].plot(range(1,len(allPSNR['val'])+1), allPSNR['val'], color=colors['Metrics'], linestyle='dashed', label='Validation', linewidth=1.5)
    ax[1].grid(linestyle='-', linewidth=0.7)
    ax[1].legend(loc='lower right', prop={'size': 12})

    # Third Line (SSIM)
    ax[2].set_ylabel('SSIM')
    ax[2].plot(range(1,len(allSSIM['train'])+1), allSSIM['train'], color=colors['Metrics'], label='Training', linewidth=1.5)
    ax[2].plot(range(1,len(allSSIM['val'])+1), allSSIM['val'], color=colors['Metrics'], linestyle='dashed', label='Validation', linewidth=1.5)
    ax[2].grid(linestyle='-', linewidth=0.7)
    ax[2].legend(loc='lower right', prop={'size': 12})

    fig.tight_layout()

    if curvesPath is not None:
        plt.savefig(curvesPath + '/LearningCurves.png')
        plt.close()
   

# ---------------------------------------------------------------------------------------
# Show individual predictions and anomaly maps (AM) 
# ---------------------------------------------------------------------------------------s
def printPredAndAM(input, prediction, mask, filePath):

    if prediction.shape[-1] == 1: 
        prediction = np.squeeze(prediction, axis=-1)
        input      = np.squeeze(input, axis=-1)
        mask       = np.squeeze(mask, axis=-1)
    
    fig, ax = plt.subplots(2,2, figsize=(8,8), dpi=60)

    ax[0,0].set_title('Input', fontsize=16)
    ax[0,0].imshow(input, cmap = 'gray', interpolation = 'bicubic')
    ax[0,0].axis('off')
    
    ax[0,1].set_title('Mask', fontsize=16)
    ax[0,1].imshow(mask, cmap = 'gray', interpolation = 'bicubic')
    ax[0,1].axis('off')

    ax[1,0].set_title('Prediction', fontsize=16)
    ax[1,0].imshow(prediction, cmap = 'gray', interpolation = 'bicubic')
    ax[1,0].axis('off')
    
    diff_im = diff(input, prediction)
    ax[1,1].imshow(input, cmap = 'gray', interpolation = 'bicubic')
    am = ax[1,1].imshow(diff_im, cmap = 'cividis', interpolation = 'bicubic',alpha=.9)
    cbar = fig.colorbar(am, ax=ax[1,1])
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(10)
    ax[1,1].set_title('Diff |I-P|', fontsize=16)
    ax[1,1].axis('off')

    fig.tight_layout()

    plt.savefig(filePath)
    plt.close()


# ---------------------------------------------------------------------------------------
# Print ROC curve 
# ---------------------------------------------------------------------------------------
def print_ROC(ImageWisepath, PixelWisepath, file_path=None):

    fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=300)
    # IMAGE-WISE
    labels  = ['L0 Norm', 'L1 Norm', 'L2 Norm', 'L$\infty$ Norm']
    dict    = read_json(ImageWisepath)
    TP, FP  = dict['all_TP'], dict['all_FP']

    max_auc = 0
    for this_TP, this_FP, this_color, this_label in zip(TP, FP, colors4, labels):
        if len(TP) == 1: 
            this_color = colors4[-1]
            this_label = labels[-1]
        # Add AUC
        this_label += '('    
        this_auc    = round(auc(this_FP, this_TP),2)
        if this_auc > max_auc: 
            max_auc = this_auc
        this_label += str(this_auc)
        this_label += ')'
        ax[0].plot(this_FP, this_TP, color=this_color, linewidth=2, label=this_label)
    ax[0].set_xlabel('FP rate')
    ax[0].set_ylabel('TP rate')
    ax[0].legend(title= "Image-wise AUC: " + str(max_auc), loc='lower right', prop={'size': 12})

    # PIXEL-WISE
    dict    = read_json(PixelWisepath)
    TP, FP  = dict['TP'], dict['FP']
    ax[1].plot(FP, TP, color='crimson', linewidth=2, label='Pixel Difference')
    ax[1].set_xlabel('FP rate')
    ax[1].set_ylabel('TP rate')
    ax[1].legend(title= "Pixel-wise AUC: " + str(round(auc(FP, TP),2)), loc='lower right', prop={'size': 12})

    fig.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
        plt.close()
    else: 
        return fig
