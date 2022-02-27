import torch.nn as nn

class GmlpBlock(nn.Module) :
    def __init__(self,
                 d_model,
                 d_ff,
                 seq_length) :
        super().__init__()

        self.ff1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU())
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU())
        self.fc = nn.Linear(d_ff, d_model)
        self.s = SGU(seq_length)

    def flip_spatial2channel(self, x):
        return x.permute(0, 2, 1).contiguous()

    def forward(self, x):

        z1 = self.ff1(x)
        z2 = self.ff2(x)

        z1 = self.flip_spatial2channel(z1)
        z2 = self.flip_spatial2channel(z2)

        z_prime = self.flip_spatial2channel(self.s(z1, z2))

        y = self.fc2(z_prime)

        return y

class SGU(nn.Module) :
    def __init__(self,
                 seq_length):
        super().__init__()
        self.fc = nn.Linear(seq_length, seq_length)
        self.ln = nn.LayerNorm(seq_length)

    def forward(self, z1, z2):
        return z1 * self.fc(self.ln(z2))
