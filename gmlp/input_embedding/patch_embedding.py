import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    it is same with vision transformer's input embedding except for positional embedding
    """
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 d_model):
        super().__init__()
        self.d_model = d_model

        self.patch_size = patch ** 2
        img_size = height * width
        assert img_size % self.patch_size == 0, 'img is not divisible with patch'

        self.seq_length = img_size // self.patch_size
        input_dim = self.patch_size * channel
        self.cls_tok = nn.Parameter(torch.randn(1, d_model), requires_grad=True)
        self.patch_emb = nn.Linear(input_dim, d_model)

        self.ln = nn.LayerNorm(d_model)

    def forward(self, img):
        N, C, H, W = img.shape

        splitted = img.view(N, C, -1).split(self.patch_size, -1)  # [N, C, H*W]
        stacked_tensor = torch.stack(splitted, dim=2)  # [N, C, (H*W)/(P**2)`, P**2]

        stacked_tensor = stacked_tensor.permute(0, 2, 1, 3).contiguous()  # [N, (H*W)/(P**2), C, P**2]
        stacked_tensor = stacked_tensor.view(N, stacked_tensor.shape[1], -1)  # [N, (H*W)/(P**2), C * P**2]
        # sequence length : (H*W)/(P**2)

        cls_tok = self.cls_tok.unsqueeze(0).repeat(N, 1, 1)  # [N, 1, d_model]
        projected = self.patch_emb(stacked_tensor)  # [N, sequence length, d_model]
        cated = torch.cat([cls_tok, projected], dim=1)  # [N, sequence length + 1, d_model]

        return cated
