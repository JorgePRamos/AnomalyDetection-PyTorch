import argparse
import yaml
import os
from os.path import dirname, abspath
import importlib
import itertools
from termcolor import colored
from multiprocessing import Process
import multiprocessing as mp

from torchsummary import summary

from models.SuperClass import *
from datasets.dataAugmentation import *

from pathlib import Path
import wandb
import torch
import platform
rootPath = dirname(dirname(abspath(__file__)))+r'/src/'


parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='default')
parser.add_argument('-train', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-decode', default=False, type=lambda x: (str(x).lower() == 'true'))

######################################################################################
#
# MAIN PROCEDURE 
# launches experiments whose parameters are described in a yaml file  
# 
# Example of use in the terminal: 
# python main.py -exp DefaultExp
# with 'DefaultExp' being the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
# By using: 
# python main.py -exp DefaultExp -train true
# You can (re-)train the network instead of importing pretrained weights (default)
######################################################################################
def call_training(thisExp, parser):
        (thisImgCat, thisDS, thisModel, thisTrain, thisDA,serverConfig) = thisExp
        serverPath = Path('configs/server/' + serverConfig + '.yaml')
        serverStream = open(serverPath, 'r')
        serverCfg    = yaml.safe_load(serverStream)
        hostName = platform.node()
        print(">> Current host name: ",hostName)
        folderPath   = Path(serverCfg['DatasetLocation']['local_folder']+ '/' + thisImgCat)
        if hostName == "Bajoo" or hostName == "Betelgeuse":
            folderPath   = Path(serverCfg['DatasetLocation']['remote_folder'] + '/' + thisImgCat)
        elif hostName == "DESKTOP-7B1KVSF":
            folderPath   = Path(serverCfg['DatasetLocation']['local_folder_laptop']+ '/' + thisImgCat)

        
        print(">> Loaded data set from: ", folderPath)
        if not os.path.exists(folderPath):
            print('ERROR : Unknown dataset location. Please update the "code/configs/server/datasetLocation.yaml" file')
            exit(1)


        # ------------------------
        # 1. NETWORK INSTANTIATION 
        # ------------------------
        modelStream     = open('configs/model/' + thisModel + '.yaml', 'r')
        thisModelConfig = yaml.safe_load(modelStream)
        if thisModelConfig['Model_class'] == 'Super': 
            myNet = Network_Class(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)
        else: 
            module = importlib.import_module('models.'+ thisModelConfig['Model_class'] +'Class')
            class_ = getattr(module, thisModelConfig['Model_class']) 
            myNet  = class_(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)
        print(colored('CURRENT EXPERIMENT :  ' + thisImgCat + ' : ' + myNet.expName, 'green'))
        #summary(myNet.model, myNet.model.inputDim)

        # ------------------
        # 2. TRAIN THE MODEL  
        # ------------------
        resultPath = Path(rootPath + '/Results_VQVAE/' + myNet.expName +'/' + myNet.imgCat)
        if parser.train:
            wandbObject = wandb.init(project="Anomaly_Detection")
            print(colored('Start to train the network', 'red'))
            myNet.train(resultPath,wandbObject)
            print(colored('The network is trained', 'red'))

        # ---------------------
        # 3. EVALUATE THE MODEL  
        # --------------------- 
        weightsPath = Path(rootPath + '/Results_VQVAE/' + myNet.expName +'/' + myNet.imgCat + '/_Weights/wghts.pkl')
        myNet.loadWeights(weightsPath)
        myNet.evaluate(resultPath = resultPath, printPrediction = True, decode = parser.decode)




def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    todoListPath = Path('Todo_List/' + parser.exp + '.yaml')
    #stream = todoListPath.read_text()
    stream = open(todoListPath, 'r')
    args   = yaml.safe_load(stream)

    # Get all experiments parameter
    imgCategory   = args['DatasetFolder']['subfolder']
    datasetConfig = args['Dataset']
    modelConfig   = args['Model']
    trainConfig   = args['Train']
    dataAugConfig = args['DataAugmentation']
    serverConfig  = args['Server']

    allConfigs   = [imgCategory, datasetConfig, modelConfig, trainConfig, dataAugConfig,serverConfig]
    allExps      = list(itertools.product(*allConfigs))
    print(">> Safely loaded: ",todoListPath)


    for thisExp in allExps:
        try:
            mp.set_start_method('spawn')
        except:
            print(">> Start method already set")
        
        p = Process(target=call_training, args=(thisExp,parser,))
        p.start()
        p.join()


if __name__ == '__main__':

    print(">> torch cuda available: ",torch.cuda.is_available())
    print(">> torch cuda device available: ", torch.cuda.get_device_name(0))
    parser = parser.parse_args()
    main(parser)
    print("### Execution Finished ###")