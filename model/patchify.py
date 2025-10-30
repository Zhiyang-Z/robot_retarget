import torch.nn as nn
from einops import rearrange

class PatchEmbed1D(nn.Module):
    def __init__(
        self,
        traj_length=64,
        patch_size=1,
        in_chans=29,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        assert traj_length % patch_size == 0, "traj_length must be divisible by patch_size"
        self.out_length = traj_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = rearrange(x, "B L C -> B C L")
        B, C, L = x.shape
        x = self.proj(x)
        assert x.shape[2] == self.out_length, f"Input length ({L}) doesn't match model ({self.out_length})."
        x = rearrange(x, "B D L -> B L D")
        x = self.norm(x)
        return x
