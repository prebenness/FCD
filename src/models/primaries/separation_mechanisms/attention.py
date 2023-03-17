'''
pytorch modules for using an attention mechanism and its complement for the
separation mechanism
'''
from torch import nn


class Attention(nn.Module):
    '''
    Attention module for split mechanisms
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        ...
