import torch.nn as nn
from resmlp.sublayer.resblock import ResMlpBlock
from mlp_mixer.input_embedding.patch_embedding import PatchEmbedding

class ResMlpClassifier(nn.Module) :
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 d_model,
                 num_layers,
                 output_dim
                 ):
        super().__init__()
        assert (height * width) % (patch ** 2) == 0, "h*w is not divisible with p**2"

        self.pe = PatchEmbedding(
            height,
            width,
            channel,
            patch,
            d_model
        )
        seq_len = (height * width)//(patch**2)
        enc = ResMlpBlock(seq_len, d_model)
        self.encoders = nn.ModuleList([enc for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        z = self.pe(x)

        for enc in self.encoders :
            z = enc(z)

        projected = self.fc(z.mean(1))

        return projected
