from copy import deepcopy
from gmlp.input_embedding.patch_embedding import PatchEmbedding
from gmlp.layer.block import GmlpBlock
import torch.nn as nn

class Gmlp(nn.Module) :
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch_size,
                 seq_length,
                 d_model,
                 d_ff,
                 num_layers,
                 output_dim
                 ):
        super().__init__()

        self.pe = PatchEmbedding(height,
                                width,
                                channel,
                                patch_size,
                                d_model)

        block = GmlpBlock(d_model,
                                d_ff,
                                seq_length)

        self.encoders = nn.ModuleList([deepcopy(block) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        emb = self.pe(x)
        for enc in self.encoders:
            emb = enc(emb)
        return self.fc(emb[:,0,:])
