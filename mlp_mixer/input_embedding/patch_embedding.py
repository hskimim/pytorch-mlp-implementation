import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 C):
        super().__init__()
        self.C = C

        self.patch_size = patch ** 2
        img_size = height * width
        assert img_size % self.patch_size == 0, 'img is not divisible with patch'

        self.seq_length = img_size // self.patch_size
        input_dim = self.patch_size * channel
        self.patch_emb = nn.Linear(input_dim, C)

    def forward(self, img):
        N, C, H, W = img.shape

        splitted = img.view(N, C, -1).split(self.patch_size, -1)  # [N, C, H*W]
        stacked_tensor = torch.stack(splitted, dim=2)  # [N, C, (H*W)/(P**2), P**2]

        stacked_tensor = stacked_tensor.permute(0, 2, 1, 3).contiguous()  # [N, (H*W)/(P**2), C, P**2]
        stacked_tensor = stacked_tensor.view(N, stacked_tensor.shape[1], -1)  # [N, (H*W)/(P**2), C * P**2]
        # S(sequence length) : (H*W)/(P**2)

        embeddings = self.patch_emb(stacked_tensor)  # [N, S, C]
        return embeddings
