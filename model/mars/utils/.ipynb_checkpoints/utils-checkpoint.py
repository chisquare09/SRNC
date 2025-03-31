'''
Created on Mar 22, 2019

@author: maria
'''
import torch
from torch.utils.data import DataLoader

from model.epoch_sampler.epoch_sampler import EpochSampler


def init_labeled_loader(data, val_split = 0.8):
    """Initialize loaders for train and validation sets. Class labels are used only
    for stratified sampling between train and validation set."""
    target = torch.tensor(list(data.y))
    uniq = torch.unique(target, sorted=True)
    # class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
    class_idxs = [idx[torch.randperm(len(idx))] for idx in class_idxs]
    train_idx = torch.cat([idx[:int(val_split*len(idx))] for idx in class_idxs if len(idx)>0])
    val_idx = torch.cat([idx[int(val_split*len(idx)):] for idx in class_idxs if len(idx)>0])
    train_loader = DataLoader(data, 
                              batch_sampler=EpochSampler(train_idx),
                              pin_memory=True)
    val_loader = DataLoader(data, 
                            batch_sampler=EpochSampler(val_idx),
                            pin_memory=True)
    return train_loader, val_loader


def init_loader(datasets, val_split = 0.8):
    train_loader_all = []
    val_loader_all = []
    
    for data in datasets:
        
        curr_load_tr, curr_load_val = init_labeled_loader(data, val_split)
        train_loader_all.append(curr_load_tr)
        val_loader_all.append(curr_load_val)
    
    if val_split==1:
        val_loader_all = None
        
    return train_loader_all, val_loader_all


def init_data_loaders(labeled_data, unlabeled_data, pretrain_data, pretrain_batch, val_split):
    """Initialize loaders for pretraing, training (labeled and unlabeled datasets) and validation. """
    train_loader, val_loader = init_loader(labeled_data, val_split)
    if not pretrain_data:
        pretrain_data = unlabeled_data
    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_data, shuffle=True,
                                                       batch_size=pretrain_batch if pretrain_batch!=None else len(unlabeled_data.x))        
    test_loader = DataLoader(unlabeled_data, 
                            batch_sampler=EpochSampler(torch.randperm(len(unlabeled_data.x))),
                            pin_memory=True) 
    test_loader,_ = init_loader([unlabeled_data], 1.0) # to reproduce results in the paper
    test_loader = test_loader[0]
    return train_loader, test_loader, pretrain_loader, val_loader
           
           
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

# def init_labeled_loader(data, val_split=0.8):
#     """Initialize loaders for train and validation sets. Class labels are used only
#     for stratified sampling between train and validation set."""
    
#     # Check if data.y exists and is not empty
#     if not hasattr(data, 'y') or len(data.y) == 0:
#         raise ValueError("The dataset does not contain valid labels in the 'y' attribute.")
    
#     target = torch.tensor(list(data.y))
    
#     # Debug: Print the target tensor
#     print(f"Target labels: {target}")
    
#     uniq = torch.unique(target, sorted=True)
    
#     # Debug: Print the unique classes
#     print(f"Unique classes: {uniq}")
    
#     class_idxs = list(map(lambda c: target.eq(c).nonzero(as_tuple=False), uniq))
    
#     # Add debug statements to print the sizes and contents
#     for i, idx in enumerate(class_idxs):
#         print(f"Class {uniq[i]}: {len(idx)} samples")
    
#     class_idxs = [idx[torch.randperm(len(idx))] for idx in class_idxs if len(idx) > 0]

#     # Debug: Print sizes of class_idxs after shuffling and filtering
#     print(f"Class indexes after filtering empty classes: {[len(idx) for idx in class_idxs]}")

#     # Ensure class_idxs is not empty
#     if not class_idxs:
#         raise ValueError("No non-empty class indexes found. Check your data.")

#     train_idx_list = [idx[:int(val_split * len(idx))] for idx in class_idxs if len(idx) > 0]
#     val_idx_list = [idx[int(val_split * len(idx)):] for idx in class_idxs if len(idx) > 0]

#     # Debug: Print sizes of train and validation index lists
#     print(f"Training index list sizes: {[len(idx) for idx in train_idx_list]}")
#     print(f"Validation index list sizes: {[len(idx) for idx in val_idx_list]}")

#     # Ensure the lists for torch.cat are not empty
#     if not train_idx_list or not any(train_idx_list):
#         raise ValueError("Training index list is empty. Check your data and val_split.")
#     if not val_idx_list or not any(val_idx_list):
#         raise ValueError("Validation index list is empty. Check your data and val_split.")

#     train_idx = torch.cat(train_idx_list)
#     val_idx = torch.cat(val_idx_list)

#     # Debug: Print sizes of train and validation indexes
#     print(f"Train indexes: {train_idx.shape}, Validation indexes: {val_idx.shape}")

#     train_loader = DataLoader(data, 
#                               batch_sampler=EpochSampler(train_idx),
#                               pin_memory=True)
#     val_loader = DataLoader(data, 
#                             batch_sampler=EpochSampler(val_idx),
#                             pin_memory=True)
#     return train_loader, val_loader

# def init_loader(datasets, val_split=0.8):
#     train_loader_all = []
#     val_loader_all = []

#     for data in datasets:
#         print(f"Initializing loader for dataset: {data}")
#         curr_load_tr, curr_load_val = init_labeled_loader(data, val_split)
#         train_loader_all.append(curr_load_tr)
#         val_loader_all.append(curr_load_val)

#     if val_split == 1:
#         val_loader_all = None

#     return train_loader_all, val_loader_all

# def init_data_loaders(labeled_data, unlabeled_data, pretrain_data, pretrain_batch, val_split):
#     """Initialize loaders for pretraing, training (labeled and unlabeled datasets) and validation."""
#     print("Initializing data loaders...")
#     train_loader, val_loader = init_loader(labeled_data, val_split)
#     if not pretrain_data:
#         pretrain_data = unlabeled_data
#     pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_data, shuffle=True,
#                                                   batch_size=pretrain_batch if pretrain_batch != None else len(unlabeled_data.x))
#     test_loader = DataLoader(unlabeled_data,
#                              batch_sampler=EpochSampler(torch.randperm(len(unlabeled_data.x))),
#                              pin_memory=True)
#     return train_loader, test_loader, pretrain_loader, val_loader

# def euclidean_dist(x, y):
#     '''
#     Compute euclidean distance between two tensors
#     '''
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     if d != y.size(1):
#         raise Exception

#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)

#     return torch.pow(x - y, 2).sum(2)

