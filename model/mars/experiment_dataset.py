# coding=utf-8
import pandas as pd 
import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms


'''
Class representing dataset for an single-cell experiment.
'''

IMG_CACHE = {}


class ExperimentDataset(data.Dataset):
    
    
    def __init__(self, x, cells, genes, metadata, y=[]):
        '''
        x: numpy array of gene expressions of cells (rows are cells)
        cells: cell IDs in the order of appearance
        genes: gene IDs in the order of appearance
        metadata: experiment identifier
        y: numeric labels of cells (empty list if unknown)
        '''
        super(ExperimentDataset, self).__init__()
        # self.transform = transforms.Compose([transforms.ToTensor()]) 
        
        self.nitems = x.shape[0]
        if len(y)>0:
            print("== Dataset: Found %d items " % x.shape[0])
            print("== Dataset: Found %d classes" % len(np.unique(y)))
                
        if type(x)==torch.Tensor:
            # self.x = x
            self.x = [x[i] for i in range(x.shape[0])]
        else:
            # shape = x.shape[1]
            # self.x = [torch.from_numpy(inst).view(shape).float() for inst in x]
             self.x = [torch.from_numpy(inst).float() for inst in x]
        if len(y)==0:
            y = np.zeros(len(self.x), dtype=np.int64)
        self.y = tuple(y.tolist())
        self.xIDs = cells.tolist() 
        self.yIDs = genes.tolist()
        self.metadata = metadata
        #print out the information
        # print("Gene Expressions (x):", self.x)
        # print("Cell IDs:", self.xIDs)
        # print("Gene IDs:", self.yIDs)
        # print("Metadata:", self.metadata)
        # print("Labels (y):", self.y)
    

    def __getitem__(self, idx):
        
        # return self.transform(self.x[idx]), self.transform(self.y[idx]),(sself.xIDs[idx])
        return self.x[idx].squeeze(), self.y[idx], self.xIDs[idx]

    def __len__(self):
        return self.nitems
    
    def get_dim(self):
        return self.x[0].shape[0]
        


