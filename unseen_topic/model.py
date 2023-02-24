import torch 
import torch.nn.functional as F
from torch_rechub.basic.layers import MLP, EmbeddingLayer
import torch.nn as nn


class MLP(torch.nn.Module):

    def __init__(self, num_classes = 91):
        super(MLP, self).__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(5, 256),
            torch.nn.BatchNorm1d(256), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),
        
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),
        )
        self.feature2 = torch.nn.Sequential(
            
            torch.nn.Linear(64, 256),
            torch.nn.BatchNorm1d(256), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),
        
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64), 
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(0.2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = self.feature2(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x