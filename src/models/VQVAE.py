import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from os.path import dirname, abspath
import os
""" -----------------------------------------------------------------------------------------
Auto-encoder with Vector Quantization Module (VQVAE)
# Inspired from https://github.com/rosinality/vq-vae-2-pytorch 
    - dsConfig (dict): dataset configuration (from the yaml file)
    - modelConfig (dict): model configuration (from the yaml file)
----------------------------------------------------------------------------------------- """ 


class VectorQuantizer(nn.Module):
    def __init__(self, dim, n_embed, commitment_cost):
        super().__init__()
        self._embedding_dim  = dim
        self._num_embeddings = n_embed

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs, modifiedEncodings=None):
        # convert inputs from BCHW -> BHWC
        inputs      = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        if modifiedEncodings is None: 
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings        = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

        else: 
            encodings = modifiedEncodings.view(-1, self._num_embeddings)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized  = inputs + (quantized - inputs).detach()
        avg_probs  = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_shape     = list(input_shape)
        encoding_shape[-1] = self._num_embeddings
        
        #print(">> quantized:  ",  quantized.permute(0, 3, 1, 2))
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings.view(encoding_shape)



class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=2),
		    nn.BatchNorm2d(ch_out),
			nn.LeakyReLU()
        )
    def forward(self,x):
        x = self.down(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=2, output_padding=1),
		    nn.BatchNorm2d(ch_out),
			nn.LeakyReLU()
        )
    def forward(self,x):
        x = self.up(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, dsConfig, modelConfig):

        super(VQVAE, self).__init__()
        # Usefull variables for the classical Autoencoder part 
        self.imgChannels = 3 if dsConfig['color'] else 1
        self.kernelSize  = int(modelConfig['Kernel_size'])
        self.depth       = int(modelConfig['Depth'])
        self.nbChannels  = int(modelConfig['Nb_feature_maps'])
        self.spatialDims = [int(dsConfig['resolution'].split('x')[0]), int(dsConfig['resolution'].split('x')[1])]
        self.inputDim    = (self.imgChannels, self.spatialDims[0], self.spatialDims[1] )

        self.channelList = []
        for idx in range(self.depth-1): 
            self.channelList.append( [int(self.nbChannels*(2**idx)), int(self.nbChannels*(2**(idx+1)))] )



        # ------------
        # Encoder part
        # ------------
        self.e1      = nn.Conv2d(self.imgChannels, self.nbChannels, self.kernelSize, padding=2)
        self.encoder = nn.ModuleList(
            [down_conv(self.channelList[idx][0], self.channelList[idx][1], self.kernelSize, 2) for idx in range(self.depth-1)]
        )

        # -------------------
        # Quantization module
        # -------------------
        self.atom_dim        = int(self.nbChannels*(2**(self.depth-1)))
        self.nb_atoms        = int(modelConfig['Nb_atoms'])
        self.commitment_cost = float(modelConfig['Commitment_cost'])

        self.quantization_module = VectorQuantizer(self.atom_dim, self.nb_atoms, self.commitment_cost)
   
        # ------------
        # Decoder part
        # ------------
        self.decoder = nn.ModuleList(
            [up_conv(self.channelList[self.depth-idx-2][1], self.channelList[self.depth-idx-2][0], self.kernelSize, 2) for idx in range(self.depth-1)]
        )
        self.Conv_1x1 = nn.Conv2d(self.nbChannels, self.imgChannels, kernel_size=self.kernelSize, padding=2)
        self.sigmoid  = nn.Sigmoid()


    def forward(self, x):
        convLayers    = [None]*(self.depth)
        convLayers[0] = self.e1(x)
        for i, encLayer in enumerate(self.encoder):
            convLayers[i+1] = encLayer(convLayers[i])
        deconvLayers  = [None]*(self.depth-1)
        for i, decLayer in enumerate(self.decoder):
            if i == 0: 
                loss, quantized, perplexity, quantized_embeddings = self.quantization_module(convLayers[-1])
                deconvLayers[0] = decLayer(quantized)
            else: 
                deconvLayers[i] = decLayer(deconvLayers[i-1])
                #deconvLayers[i] = torch.add(convLayers[self.depth-2-i], deconvLayers[i])
                
        out = self.Conv_1x1(deconvLayers[-1])
        out = self.sigmoid(out)
        filename = Path(dirname(dirname(abspath(__file__)))+"/quantized_embeddings.txt")
        """
        print(">>> writing in:  ",filename)
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode) as file:
            file.write(str(quantized_embeddings)+"\n\n")
            """
        return out, loss,quantized_embeddings

    
