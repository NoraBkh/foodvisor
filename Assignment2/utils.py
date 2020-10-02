import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

from DATASET_tomato import  TomatoDataset


CUDA = False


def get_dataset(batch_size, path, valid_size = .2):
    """
    This function loads the dataset and performs transformations on each
     image (listed in transform = ...`).
     
     ##params:
        # valid_size: means 20%of data for validation and 80% for training
    
    """
    
    ## First pre-process the data( reshape, tensor and normalization)
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       
                                       ])    
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      
                                      ])
    
    train_dataset = TomatoDataset(path, transform = train_transforms)
    val_dataset = TomatoDataset(path, transform = test_transforms)
    
    ## Split data into train and val dataset
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader
