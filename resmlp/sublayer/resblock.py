import torch.nn as nn
from resmlp.sublayer.cross_channel import CrossChannelBlock
from resmlp.sublayer.cross_patch import CrossPatchBlock

class ResMlpBlock(nn.Module) :
    def __init__(self, seq_length, channel_dim):
        super().__init__()
        cc = CrossChannelBlock(channel_dim)
        cp = CrossPatchBlock(seq_length, channel_dim)
        self.enc = nn.Sequential(cc, cp)

    def forward(self, x):
        return self.enc(x)
