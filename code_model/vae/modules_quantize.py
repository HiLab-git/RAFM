"""Vector quantizer extracted from taming-transformers.

来源：taming/modules/vqvae/quantize.py
保留 VectorQuantizer2（VQGAN 默认使用它）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer2(nn.Module):
    """Improved vector quantizer.

    Args:
        n_e: number of embeddings
        e_dim: embedding dimension
        beta: commitment loss weight
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        remap: Optional[str] = None,
        sane_index_shape: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.sane_index_shape = sane_index_shape

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if remap is not None:
            raise NotImplementedError("remap is not included in vqgan_export for simplicity")

    def forward(self, z: torch.Tensor):
        """z: (B, C, H, W)"""
        z = z.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e: (z-e)^2 = z^2 + e^2 - 2 e.z
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # commitment + codebook loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # straight-through estimator
        z_q = z + (z_q - z).detach()

        # reshape back
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.sane_index_shape:
            idx = min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])
        else:
            idx = min_encoding_indices

        info = (None, None, idx)
        return z_q, loss, info

    def embed_code(self, code_b: torch.Tensor) -> torch.Tensor:
        """code_b: indices"""
        z_q = self.embedding(code_b)
        return z_q
