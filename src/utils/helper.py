import os
from pathlib import Path
import json
import numpy as np 
import copy
from itertools import chain
from collections import OrderedDict

""" -----------------------------------------------------------------------------------------
Functions to compute Image-wise and Pixel-wise ROC curves 
----------------------------------------------------------------------------------------- """

# ---------------------------------------------------------------------------------------
# Error metrics between two images 
# INPUT: 
#    - anoMap: anomaly maps
#    - norm: string representing norm of interrest
# OUTPUT: 
#    - L0, L1 (Mean Absolute Error), L2 (Root Mean Squarred Error) or Linf (Max Absolute Error)
# ---------------------------------------------------------------------------------------
def l_metric(anoMap, norm): 
    anoMap = np.sum(anoMap.astype(np.float64), axis=-1)
    if norm == 'l0': 
        return np.count_nonzero(anoMap)
    elif norm == 'l1':
        W, H = anoMap.shape[0], anoMap.shape[1]
        return np.sum(anoMap) / (W*H)
    elif norm == 'l2': 
        W, H = anoMap.shape[0], anoMap.shape[1]
        return np.sqrt(np.sum(anoMap**2) / (W*H))
    elif norm == 'linf': 
        return np.max(anoMap)
    else: 
        return 0

# ---------------------------------------------------------------------------------------
# Computes FP and TP of a default matrix with threshold sorted in ascending order
# INPUT: 
#    - anoMaps:  anomaly map 
#    - y_labels: binary labels for classification
# OUTPUT: 
#    - TP and FP arrays for the ROC curve 
# ---------------------------------------------------------------------------------------

def compute_ROC(anoMaps, y_GT): 
    # Compute all thresholds
    thres_matrix   = {'l0': [], 'l1': [], 'l2': [], 'linf': []} 
    for thisMap in anoMaps: 
        for this_norm in ['l0', 'l1', 'l2', 'linf']:
            thres_matrix[this_norm].append(l_metric(thisMap, this_norm))


    # Compute ROC Curves for all norms
    all_TP, all_FP = [], []
    for this_norm in ['l0', 'l1', 'l2', 'linf']: 
        y_score = np.array(thres_matrix[this_norm])
        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score            = y_score[desc_score_indices]
        y_copy             = np.copy(y_GT)
        y_labels           = y_copy[desc_score_indices]

        # Keep transition thresholds
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs         = np.r_[distinct_value_indices, y_labels.size - 1]
        tps = np.cumsum(y_labels)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tps = [0] + (tps/tps[-1]).tolist()
        fps = [0] + (fps/fps[-1]).tolist()

        all_TP.append(tps)
        all_FP.append(fps)

    return all_TP, all_FP

# ---------------------------------------------------------------------------------------
# Computes FP and TP of a default matrix with threshold sorted in ascending order
# INPUT: 
#    - anoMaps:  anomaly map 
#    - y_labels: binary labels for classification
# OUTPUT: 
#    - TP and FP arrays for the ROC curve 
# --------------------------------------------------------------------------------------- 
def compute_ROC_pixel_wise(anoMaps_in, mask_in): 

    anoMaps, mask = copy.deepcopy(anoMaps_in), copy.deepcopy(mask_in) 

    thres_matrix   = []
    for thisMap in anoMaps: 
        if thisMap.shape[-1] != 1: 
            thisMap = np.sum(thisMap, axis=-1)
        else: 
            thisMap = np.squeeze(thisMap, axis=-1)
        thres_matrix.append( thisMap )
    thres_matrix = np.array(thres_matrix)

    thres_matrix = thres_matrix.ravel(order='C')
    y_GT         = mask.ravel(order='C')
    y_GT         = y_GT > 0
    
    # Compute ROC Curves
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(thres_matrix, kind="mergesort")[::-1]
    y_score            = thres_matrix[desc_score_indices]
    y_copy             = np.copy(y_GT)
    y_labels           = y_copy[desc_score_indices]

    # Keep transition thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs         = np.r_[distinct_value_indices, y_labels.size-1]

    tps = np.cumsum(y_labels)[threshold_idxs]

    fps = 1 + threshold_idxs - tps

    tps = [0] + (tps/tps[-1]).tolist()
    fps = [0] + (fps/fps[-1]).tolist()


    return tps, fps


""" -----------------------------------------------------------------------------------------
Other functions
----------------------------------------------------------------------------------------- """

# ---------------------------------------------------------------------------------------
# Concatenate multiple dictionaries
# ---------------------------------------------------------------------------------------
def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

# ---------------------------------------------------------------------------------------
# Absolute difference of two images in uint8 format 
# ---------------------------------------------------------------------------------------
def diff(img1, img2):
    diff16 = np.abs(np.int16(img1)-np.int16(img2))
    if diff16.shape[-1] == 3: 
        diff16 = np.expand_dims(np.divide(np.sum(diff16, axis=-1),3),-1)
    return np.uint8(diff16)

# ---------------------------------------------------------------------------------------
# Write a dictionnary in a json file in file_path location
# ---------------------------------------------------------------------------------------
def write_json(dict, file_path, sort_keys=True): 
    if not os.path.exists( str(Path(file_path).parent )):
        os.makedirs( str(Path(file_path).parent))
    with open(file_path, 'w') as fp:
        json.dump(dict, fp, sort_keys=sort_keys, indent=4)

# ---------------------------------------------------------------------------------------
# Read a dictionnary in a json file in file_path location
# ---------------------------------------------------------------------------------------
def read_json(file_path): 
    with open(file_path, 'r') as fp:
        return json.load(fp, object_pairs_hook=OrderedDict)