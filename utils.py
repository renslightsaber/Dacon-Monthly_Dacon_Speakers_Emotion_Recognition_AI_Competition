import re
import os
import gc
import time
import random
import string

import copy
from copy import deepcopy

import numpy as np
import pandas as pd

# Utils
from tqdm.auto import tqdm, trange

import matplotlib.pyplot as plt

## Pytorch Import
import torch 
import torch.nn as nn
import torch.optim as optim

# from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


############## Scheduler ##################
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


############# Set Seeed ####################
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    

########### add speaker info ################
def add_speaker_info(train):
    train.loc[:, "Utterance_2"] = train.Speaker.str.upper() + ": " +  train.Utterance
    print(train.head())
    return train
  

########## make essay ####################
def make_essay(df, id_num = 0, num_sentences= 4):
    
    temp = df[df.Dialogue_ID == id_num]
    length = len(temp)
    
    for num, index in enumerate(temp.index):
        
        main_sentence = " "+ "</s></s>" + " " + temp.loc[index, 'Utterance_2'] + " " + "</s></s>" + " "

        if length >= num_sentences + 1:
            if num == 0:
                previous_sentences = ""
                after_sentences = ""
                # after_sentences = " ".join(temp.loc[index + 1 : index + 1 + num_sentences , 'Utterance_2'].to_list())
            elif num >=1:
                previous_sentences = " ".join(temp.loc[index - num_sentences : index, 'Utterance_2'].to_list())
                after_sentences = ""
                # after_sentences = " ".join(temp.loc[index + 1 : index + 1 + num_sentences, 'Utterance_2'].to_list())
            elif num == len(temp) -1:
                previous_sentences = " ".join(temp.loc[index - num_sentences :index, 'Utterance_2'].to_list())
                after_sentences = ""
            
        else:
            if num == 0:
                previous_sentences = ""
                after_sentences = ""
                # after_sentences = " ".join(temp.loc[index+1:, 'Utterance_2'].to_list())
            elif num >=1:
                previous_sentences = " ".join(temp.loc[:index, 'Utterance_2'].to_list())
                after_sentences = ""
                # after_sentences = " ".join(temp.loc[index+1:, 'Utterance_2'].to_list())
            elif num == len(temp) -1:
                previous_sentences = " ".join(temp.loc[:index, 'Utterance_2'].to_list())
                after_sentences = ""
    
        text = previous_sentences + main_sentence + after_sentences

        df.loc[index, 'essay'] = text
        
        
        
############### Data ##################
def dacon_competition_data(base_path, 
                           add_speaker = True, 
                           make_essay_option= True):

    train = pd.read_csv(base_path + 'train.csv')
    test = pd.read_csv(base_path + 'test.csv')
    ss = pd.read_csv(base_path + 'sample_submission.csv')

    print("Train Data Shape: ", train.shape)
    print("Test Data Shape: ", test.shape)
    print("Submission Data Shape: ", ss.shape)

    # add_speaker
    if add_speaker:
        train = add_speaker_info(train)
        test = add_speaker_info(test)
    else:
        # Column Rename to 'Utterance_2'
        train.rename(columns={'Utterance':'Utterance_2'}, inplace = True)
        test.rename(columns={'Utterance':'Utterance_2'}, inplace = True)

    # make_essay   
    if make_essay_option:
        for num in train.Dialogue_ID.unique():
            make_essay(df = train, id_num = num)
        for num in test.Dialogue_ID.unique():
            make_essay(df = test, id_num = num)
    else:
        # Column Rename to 'essay'
        train.rename(columns={'Utterance_2':'essay'}, inplace = True)
        test.rename(columns={'Utterance_2':'essay'}, inplace = True)

    return train, test, ss

  
  
################ Visualize #####################
def make_plot(result, stage = "Loss"):

    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'

    elif stage == "Acc":
        trains = "Train Acc"
        valids = "Valid Acc"

    elif stage == "F1":
        trains = "Train F1"
        valids = "Valid F1"

    plt.figure(figsize=(10, 6))
    
    plt.title(f"Train/Valid {stage} History", fontsize = 20)

    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]

    plt.plot(
        range(0, len(result[trains][plot_from:])), 
        result[trains][plot_from:], 
        label = trains
        )

    plt.plot(
        range(0, len(result[valids][plot_from:])), 
        result[valids][plot_from:], 
        label = valids
        )

    plt.legend()
    if stage == "Loss":
        plt.yscale('log')
    plt.grid(True)
    plt.show()  
    
    
    
    
    
    
