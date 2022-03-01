import torch.nn as nn
from resmlp.sublayer.affine import AffineTransformBlock

class CrossChannelBlock(nn.Module) :
    def __init__(self, channel_dim):
        super().__init__()
        self.aff1 = AffineTransformBlock(channel_dim)
        self.aff2 = AffineTransformBlock(channel_dim)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(channel_dim, channel_dim)
        self.fc2 = nn.Linear(channel_dim, channel_dim)

    def forward(self, z):
        projected = self.fc1(self.aff1(z))
        projected = self.fc2(self.act(projected))
        y = z + self.aff2(projected)
        return y
