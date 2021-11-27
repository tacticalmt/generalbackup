import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self,n_channels, out_channels, hidden_channels=None,ksize=3, pad=1,
                 activation='relu', downsample=False):
        super(ResBlock,self).__init__()
        self.res=nn.Sequential
        