import torch.nn as nn
import torch

class AffineTransformBlock(nn.Module) :
    def __init__(self, transform_dim):
        super().__init__()
        self.a = nn.Parameter(torch.randn(transform_dim))
        self.b = nn.Parameter(torch.randn(transform_dim))

    def forward(self, x):
        return self.a * x + self.b