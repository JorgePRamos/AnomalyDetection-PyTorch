import numpy as np

import torch
import torch.nn as nn

""" -----------------------------------------------------------------------------------------
Auto-encoder with Skip-connections model (AESc)
    - dsConfig (dict): dataset configuration (from the yaml file)
    - modelConfig (dict): model configuration (from the yaml file)
----------------------------------------------------------------------------------------- """ 

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride):
        super(down_conv,self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding='valid'),
		    nn.BatchNorm2d(ch_out),
			nn.LeakyReLU()
        )
    def forward(self,x):
        x = self.down(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, output_padding):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, output_padding=output_padding),
		    nn.BatchNorm2d(ch_out),
			nn.LeakyReLU()
        )
    def forward(self,x):
        x = self.up(x)
        return x


class AESc(nn.Module):
    def __init__(self, dsConfig, modelConfig):
        super(AESc,self).__init__()
        # Usefull variables
        self.imgChannels = 3 if dsConfig['color'] else 1
        self.kernelSize  = int(modelConfig['Kernel_size'])
        self.depth       = int(modelConfig['Depth'])
        self.nbChannels  = int(modelConfig['Nb_feature_maps'])
        resolutionStr    = dsConfig['resolution']
        self.spatialDims = [[ int(resolutionStr.split('x')[0]), int(resolutionStr.split('x')[1]) ]]
        self.inputDim    = (self.imgChannels, self.spatialDims[0][0], self.spatialDims[0][1] )
        for _ in range(self.depth-1): 
            w,h               = self.spatialDims[-1][0], self.spatialDims[-1][1]
            self.spatialDims += [[ int(np.ceil(float(w-self.kernelSize+1)/2)), int(np.ceil(float(h-self.kernelSize+1)/2)) ]] 

        self.channelList = []
        for idx in range(self.depth-1): 
            self.channelList.append( [int(self.nbChannels*(2**idx)), int(self.nbChannels*(2**(idx+1)))] )
        
        # ------------
        # Encoder part
        # ------------
        self.e1      = nn.Conv2d(self.imgChannels, self.nbChannels, self.kernelSize, padding='same')
        self.encoder = nn.ModuleList(
            [down_conv(self.channelList[idx][0], self.channelList[idx][1], self.kernelSize, 2) for idx in range(self.depth-1)]
        )

        # ------------
        # Decoder part
        # ------------
        zeroPad = []
        for idx in range(self.depth-1):
            # Encoder dimensions 
            he, we = self.spatialDims[self.depth-2-idx][0], self.spatialDims[self.depth-2-idx][1]
            # Decoder dimensions (without padding)
            hd, wd = (self.spatialDims[self.depth-1-idx][0]-1) * 2 + self.kernelSize, (self.spatialDims[self.depth-1-idx][1]-1) * 2 + self.kernelSize
            zeroPad.append((he-hd, we-wd))       
        
        self.decoder = nn.ModuleList(
            [up_conv(self.channelList[self.depth-idx-2][1], self.channelList[self.depth-idx-2][0], self.kernelSize, 2, zeroPad[idx]) for idx in range(self.depth-1)]
        )
        self.Conv_1x1 = nn.Conv2d(self.nbChannels, self.imgChannels, kernel_size=self.kernelSize, padding='same')
        self.sigmoid  = nn.Sigmoid()


    def forward(self, x):
        convLayers    = [None]*(self.depth)
        convLayers[0] = self.e1(x)
        for i, encLayer in enumerate(self.encoder):
            convLayers[i+1] = encLayer(convLayers[i])

        deconvLayers  = [None]*(self.depth-1)
        for i, decLayer in enumerate(self.decoder):
            if i == 0: 
                deconvLayers[0] = decLayer(convLayers[-1])
            else: 
                deconvLayers[i] = decLayer(deconvLayers[i-1])
                deconvLayers[i] = torch.add(convLayers[self.depth-2-i], deconvLayers[i])
        out = self.Conv_1x1(deconvLayers[-1])
        out = self.sigmoid(out)

        return out

