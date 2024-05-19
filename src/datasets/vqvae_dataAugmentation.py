import torch
torch.manual_seed(2018)
import albumentations.pytorch

import albumentations as alb

from matplotlib import pyplot as plt
import yaml
import PIL
import cv2
import math

import numpy as np
from random import randint, uniform
from numpy.random import uniform
from skimage.util import random_noise
from skimage.draw import ellipse_perimeter
from scipy.interpolate import interp1d 
from scipy.ndimage import gaussian_filter



""" -----------------------------------------------------------------------------------------
Get the list of transforms for data augmentation (Common data augmentation methods)
    - dsConfig (dict): dataset configuration (from the yaml file)
    - DAConfig (dict): data augmentation configuration (from the yaml file)
    - train (bool): True if the train set is concerned (False for the test set) 
----------------------------------------------------------------------------------------- """ 

def getTransforms(dsConfig, DAConfig, train): 
    imgTransformsList= []

    # Resolution
    resolutionStr       = dsConfig['resolution']
    resolutionTranform  = [alb.Resize(int(resolutionStr.split('x')[0]), int(resolutionStr.split('x')[1]))]
    imgTransformsList  += resolutionTranform

    # Grayscale
    # The ToGray() transform does not work when integrated here in the framework
    color         = dsConfig['color']
    grayTransform = not color

    # Rotations
    degree   = float(DAConfig['Rotation'])
    if degree > 0.0 and train: 
        rotationTranform    = [alb.Rotate(limit=degree)]
        imgTransformsList  += rotationTranform

    # Horizontal and vertical flip
    vFlip           = float(DAConfig['Vertical_flip'])
    hFlip           = float(DAConfig['Horizontal_flip'])
    if vFlip > 0.0 and train: 
        vFlipTranform       = [alb.VerticalFlip(p=vFlip)]
        imgTransformsList  += vFlipTranform
    if hFlip > 0.0 and train: 
        hFlipTranform       = [alb.HorizontalFlip(p=hFlip)]
        imgTransformsList  += hFlipTranform

    # ColorJitter
    bLim = float(DAConfig['BrightnessContrast']['brightness_limit'])
    cLim = float(DAConfig['BrightnessContrast']['contrast_limit'])
    p    = float(DAConfig['BrightnessContrast']['p'])
    if train: 
        jitterTransform     = [alb.RandomBrightnessContrast(brightness_limit=bLim, contrast_limit=cLim, p=p)]
        imgTransformsList  += jitterTransform

    imgTransforms     = alb.ReplayCompose(imgTransformsList)
    toTensorTransform = alb.pytorch.transforms.ToTensorV2()

    # Custom transforms
    if DAConfig['custom'] is not '' and train: 
        try :
            customDAStream     = open('configs/dataAugmentation/' + DAConfig['custom'], 'r')
            customDAConfig     = yaml.safe_load(customDAStream)
            customTransform    = SyntheticCorruption(customDAConfig)
        except:
            print("Unknown custom data augmentation config") 
            customTransform = None
    else:
        customTransform = None
        
    return imgTransforms, toTensorTransform, grayTransform, customTransform

""" -----------------------------------------------------------------------------------------
Custom data augmentation methods
    - customDAConfig (dict): data augmentation configuration (from the yaml file)
----------------------------------------------------------------------------------------- """ 

class SyntheticCorruption(object):

    def __init__(self, customDAConfig):
        self.config = customDAConfig

    def __call__(self, img):
        if 'Stain' in self.config.keys(): 
            stainPrm    = self.config['Stain']
            newImg      = self._addStain(np.array(img), stainPrm['size'], stainPrm['color'], stainPrm['irregularity'], stainPrm['blur'])
        elif 'Scratch' in self.config.keys(): 
            scratchPrm  = self.config['Scratch']
            newImg      = self._addScratch(np.array(img), scratchPrm['color'])
        elif 'Gaussian' in self.config.keys(): 
            gaussianPrm = self.config['Gaussian']
            newImg      = self._addGaussian(np.array(img), gaussianPrm['sigma'])
        return newImg

    # ---------------------------------------------------------------------------------------
    # Stain model 
    #   - img (np array): input image 
    #   - size (int-int): size range of the stain (between 0 and 100)
    #   - color (int-int): color range of the stain (between 0 and 255)
    #   - irregularity (float): 0.0 for a regular ellipse and >0.0 for irregular contour
    #   - blur (float):  0.0 for sharp countour and >0.0 for bluerred countour
    # ---------------------------------------------------------------------------------------
    def _addStain(self, img, size, color, irregularity, blur):
        # Fix Color 
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color                = randint(min_color, max_color)
        # Fix Size, Location and Rotation of the ellipse
        col, row             = img.shape[1], img.shape[0]
        min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
        a, b                 = randint(max(int(min_range/100.*col),1), int(max_range/100.*col)), randint(max(int(min_range/100.*row),1), int(max_range/100.*row))
        rotation             = uniform(0, 2*np.pi)

        cx, cy   = randint(max(a,b), int(col-max(a,b))), randint(max(a,b), int(row-max(a,b)))
        x,y      = ellipse_perimeter(cy, cx, a, b, rotation)
        contour  = np.array([[i,j] for i,j in zip(x,y)])

        # Change the shape of the ellipse 
        if irregularity > 0: 
            contour = perturbate_ellipse(contour, cx, cy, (a+b)/2, irregularity)

        mask = np.zeros((row, col, 1)) 
        mask = cv2.drawContours(mask, [contour], -1, 1, -1)

        if blur != 0 : 
            mask = gaussian_filter(mask, max(a,b)*blur)

        if img.shape[-1] == 3: 
            mask = np.dstack([mask]*3)
    
        not_modified = np.subtract(np.ones(img.shape), mask)
        stain        = 255*random_noise(np.zeros(img.shape), mode='gaussian', mean = color/255., var = 0.05/255.)
        result       = np.add( np.multiply(img, not_modified), np.multiply(stain, mask) ) 


        return result.astype(np.uint8)

    # ---------------------------------------------------------------------------------------
    # Scratch model 
    #   - img (np array): input image 
    #   - color (int-int): color range of the stain (between 0 and 255)
    # ---------------------------------------------------------------------------------------
    def _addScratch(self, img, color): 
        # Fix Color 
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color                = randint(min_color, max_color)
        max_x, max_y     = img.shape[1], img.shape[0]
        start_point_x    = randint(int(max_x/6), int(max_x - max_x/6 + 1))
        start_point_y    = randint(int(max_y/6), int(max_y - max_y/6 + 1))
        length_scratch_x = randint(start_point_x, max_x)/4
        length_scratch_y = randint(start_point_y, max_y)/4

        scratch_shape     = ['line', 'sin', 'root']
        scratch_direction = ['right', 'left', 'down', 'up']
        shape             = randint(0,2)
        direction         = randint(0,3)

        list_point = []
        if scratch_shape[shape] == 'line':
            list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _line, img.shape)
        elif scratch_shape[shape] == 'sin':
            list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _sin, img.shape)
        elif scratch_shape[shape] == 'root':
            list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _root, img.shape)

        for x,y in list_point:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    point_x, point_y = (x + dx, y + dy)
                    if 0 <= point_x < max_x and 0 <= point_y < max_y:
                        new_val = np.random.normal(color, 20, 1)
                        if new_val > 0:
                            img[point_y, point_x] = new_val
                        else:
                            img[point_y, point_x] = 0
        return img

    # ---------------------------------------------------------------------------------------
    # Gaussian noise model 
    #   - img (np array): input image 
    #   - sigma (float): noise standard deviation
    # ---------------------------------------------------------------------------------------
    def _addGaussian(self, img, sigma): 
        noise  = 255*random_noise(np.zeros(img.shape), mode='gaussian', mean = 0, var = sigma)
        result = np.add(img, noise)
        return result.astype(np.uint8)

# ---------------------------------------------------------------------------------------
# (helper functions)
# ---------------------------------------------------------------------------------------

def perturbate_ellipse(contour, cx, cy, diag, irregularity):
    # Keep only some points
    if len(contour) < 20: 
        pts = contour
    else: 
        pts = contour[0::int(len(contour)/20)]

    # Perturbate coordinates
    for idx,pt in enumerate(pts): 
        pts[idx] = [pt[0]+randint(-int(diag*irregularity),int(diag*irregularity)), pt[1]+randint(-int(diag*irregularity),int(diag*irregularity))]
    pts = sorted(pts, key=lambda p: clockwiseangle(p, cx, cy))
    pts.append([pts[0][0], pts[0][1]])

    # Interpolate between remaining points
    i = np.arange(len(pts))
    interp_i = np.linspace(0, i.max(), 10 * i.max())
    xi = interp1d(i, np.array(pts)[:,0], kind='cubic')(interp_i)
    yi = interp1d(i, np.array(pts)[:,1], kind='cubic')(interp_i) 
 
    return np.array([[int(i),int(j)] for i,j in zip(yi,xi)])

def clockwiseangle(point, cx, cy):
    refvec = [0 , 1]
    vector = [point[0]-cy, point[1]-cx]
    norm   = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if norm == 0:
        return -math.pi
    normalized = [vector[0]/norm, vector[1]/norm]
    dotprod    = normalized[0]*refvec[0] + normalized[1]*refvec[1] 
    diffprod   = refvec[1]*normalized[0] - refvec[0]*normalized[1] 
    angle      = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2*math.pi+angle
    return angle

def func_x(start_x, length_x, start_y, length_y, direction, func, shape):
    set_point = list()
    if direction == 'up':
        for x in np.arange(0.0, length_x, 0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'down':
        for x in np.arange(0.0, -1*length_x, -0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'right':
        for y in np.arange(0.0, length_y, 0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'left':
        for y in np.arange(0.0, -1*length_y, -0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))

    return set_point

def _line(x):
    return 2*x

def _sin(x):
    return 16*math.sin(math.radians(x*4))

def _root(x):
    return 10*math.sqrt(x) if x >= 0 else 10*math.sqrt(-1*x)
