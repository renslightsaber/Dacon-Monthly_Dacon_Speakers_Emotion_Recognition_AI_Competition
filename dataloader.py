import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


## MyDataset()
class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, mode = "train"):
        self.dataset = df
        self.max_length=  max_length
        self.tokenizer = tokenizer
        self.mode = mode
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Original
        # text = self.dataset['Utterance'][idx]  

        text = self.dataset.essay[idx]

        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            # padding='max_length', 
            max_length = self.max_length, 
            truncation=True, 
            )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
    
        if self.mode == "train":
            y = self.dataset['new_target'][idx]
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': y}
        else:
            return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],}
          
          
          
## prepare_loaders()          
def prepare_loader(train, 
                   fold,
                   tokenizer, 
                   max_length, 
                   bs,
                   collate_fn):
    
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = MyDataset(train_df, tokenizer, max_length, mode = "train")

    valid_ds = MyDataset(valid_df, tokenizer, max_length, mode = "train")
    
    # Dataset -> DataLoader
    train_loader = DataLoader(train_ds,
                              batch_size = bs, 
                              collate_fn=collate_fn, 
                              ## Dependency on device
                              # num_workers = 2, 
                              # pin_memory = True, 
                              shuffle = True, 
                              drop_last= True)

    valid_loader = DataLoader(valid_ds,
                              batch_size = bs],
                              collate_fn=collate_fn,
                              ## Dependency on device
                              # num_workers = 2,
                              # pin_memory = True,
                              shuffle = False, )
    
    print("Dataloader Completed")
    return train_loader, valid_loader
  
  
  
