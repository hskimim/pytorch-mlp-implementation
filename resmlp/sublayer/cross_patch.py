import torch.nn as nn
from resmlp.sublayer.affine import AffineTransformBlock

class CrossPatchBlock(nn.Module) :
    def __init__(self, seq_length, channel_dim):
        super().__init__()
        self.aff1 = AffineTransformBlock(seq_length)
        self.aff2 = AffineTransformBlock(channel_dim)
        self.fc = nn.Linear(seq_length, seq_length)

    def forward(self, x):
        transposed = x.permute(0,2,1).contiguous()
        projected = self.fc(self.aff1(transposed)).permute(0,1,2)
        z = x + self.aff2(projected)
        return z
