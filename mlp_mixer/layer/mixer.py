from copy import deepcopy
import torch.nn as nn
from mlp_mixer.input_embedding.patch_embedding import PatchEmbedding

class Mixer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)
        self.act = nn.GELU()

    def forward(self, x):
        normalized_x = self.ln(x)
        projected = self.w1(normalized_x)
        activated = self.act(projected)
        return x + self.w2(activated)


class MixerBlock(nn.Module):
    def __init__(self, channel_mixer, token_mixer):
        super().__init__()
        self.cm = channel_mixer
        self.tm = token_mixer

    def forward(self, x):
        u = self.cm(x)
        ut = u.permute(0, 2, 1).contiguous()
        y = self.tm(ut)
        return y.permute(0, 2, 1).contiguous()


class MM(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel,
                 num_layers,
                 patch_size,
                 C,
                 S,
                 D_c,
                 D_s,
                 output_dim):
        super().__init__()

        self.pe = PatchEmbedding(height,
                                 width,
                                 channel,
                                 patch_size,
                                 C)

        channel_mixer = Mixer(C, D_c)
        token_mixer = Mixer(S, D_s)
        mixer = MixerBlock(channel_mixer, token_mixer)

        self.encoders = nn.ModuleList([deepcopy(mixer) for _ in range(num_layers)])
        self.fc = nn.Linear(C, output_dim)

    def forward(self, x):
        emb = self.pe(x)
        for enc in self.encoders:
            emb = enc(emb)

        return self.fc(emb.mean(1))
