"""vqgan_export.vqvae_model

在 vqgan_export 同目录下提供一个纯 PyTorch 的 VQ-VAE（无判别器）最小实现，方便你按
VQGANModel 的写法接入自己的 BaseModel 框架。

特点
- 结构与 VQGAN 共用 Encoder/Decoder/VectorQuantizer2
- 只优化一套参数（encoder/decoder/quantizer/conv）
- Loss: L1 重建 + codebook loss

输入约定
- 输入 x: Tensor[B,C,H,W]，范围建议 [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .modules_diffusion import Encoder, Decoder
from .modules_quantize import VectorQuantizer2


@dataclass
class VQVAEConfig:
    # encoder/decoder
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 1
    out_ch: int = 1
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.0
    double_z: bool = False

    # codebook
    n_embed: int = 1024
    embed_dim: int = 256
    beta: float = 0.25

    # loss
    pixelloss_weight: float = 1.0
    codebook_weight: float = 1.0


class VQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg

        ddconfig = dict(
            double_z=cfg.double_z,
            z_channels=cfg.z_channels,
            resolution=cfg.resolution,
            in_channels=cfg.in_channels,
            out_ch=cfg.out_ch,
            ch=cfg.ch,
            ch_mult=list(cfg.ch_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=list(cfg.attn_resolutions),
            dropout=cfg.dropout,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = nn.Conv2d(cfg.z_channels, cfg.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.z_channels, 1)

        self.quantize = VectorQuantizer2(
            n_e=cfg.n_embed,
            e_dim=cfg.embed_dim,
            beta=cfg.beta,
            remap=None,
            sane_index_shape=False,
        )

        self.global_step = 0

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant: torch.Tensor):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, x: torch.Tensor):
        quant, qloss, _ = self.encode(x)
        xrec = self.decode(quant)
        return xrec, qloss

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """返回 total loss 以及日志。"""
        xrec, qloss = self(x)
        rec = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        loss = self.cfg.pixelloss_weight * rec + self.cfg.codebook_weight * qloss.mean()
        out = {
            "loss": loss.detach(),
            "rec_loss": rec.detach(),
            "q_loss": qloss.detach(),
            "xrec": xrec.detach(),
        }
        return loss, out

    def configure_optimizers(self, lr: float, betas=(0.5, 0.9)):
        opt = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=betas,
        )
        return opt
