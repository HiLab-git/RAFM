"""vqgan_export.model

纯 PyTorch 的 VQGAN 最小实现（便于集成到你自己的训练框架中）。

设计目标
- 不依赖 pytorch_lightning / omegaconf
- 不依赖本仓库的数据管线
- 只暴露两类 loss 计算：generator_loss / discriminator_loss

输入约定
- 输入图像 x: Tensor[B,C,H,W]，建议范围 [-1, 1]

用法（伪代码）
    model = VQGAN(...)
    opt_g, opt_d = model.configure_optimizers(lr=...)

    # step G
    loss_g, out_g = model.generator_loss(real_B)
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    # step D
    loss_d, out_d = model.discriminator_loss(real_B)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_diffusion import Encoder, Decoder
from .modules_quantize import VectorQuantizer2
from .modules_discriminator import NLayerDiscriminator, weights_init
from .lpips import LPIPS


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    return value if global_step < threshold else weight


@dataclass
class VQGANConfig:
    # encoder/decoder (ddconfig)
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

    # discriminator/loss
    disc_start: int = 0
    disc_in_channels: int = 1
    disc_num_layers: int = 1
    disc_ndf: int = 64
    disc_factor: float = 1.0
    disc_weight: float = 0.8
    disc_loss: str = "hinge"  # hinge|vanilla

    # loss weights
    codebook_weight: float = 1.0
    pixelloss_weight: float = 1.0
    perceptual_weight: float = 1.0

    # misc
    use_lpips: bool = False


class VQGAN(nn.Module):
    def __init__(self, cfg: VQGANConfig):
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

        self.discriminator = NLayerDiscriminator(
            input_nc=cfg.disc_in_channels,
            n_layers=cfg.disc_num_layers,
            use_actnorm=False,
            ndf=cfg.disc_ndf,
        ).apply(weights_init)

        self.lpips = LPIPS().eval() if cfg.use_lpips else None
        if self.lpips is not None:
            for p in self.lpips.parameters():
                p.requires_grad = False

        if cfg.disc_loss not in {"hinge", "vanilla"}:
            raise ValueError(f"Unknown disc_loss: {cfg.disc_loss}")
        self._disc_loss_fn = hinge_d_loss if cfg.disc_loss == "hinge" else vanilla_d_loss

        # tracked outside
        self.global_step = 0

    # ---------- core forward ----------
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

    def get_last_layer(self):
        # decoder 最后一层权重（用于自适应权重计算）
        return self.decoder.conv_out.weight

    # ---------- losses ----------
    def _reconstruction_loss(self, x: torch.Tensor, xrec: torch.Tensor):
        rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
        if self.lpips is not None and self.cfg.perceptual_weight > 0:
            p_loss = self.lpips(x.contiguous(), xrec.contiguous())
            rec_loss = rec_loss + self.cfg.perceptual_weight * p_loss
        else:
            p_loss = torch.zeros((), device=x.device)
        nll_loss = torch.mean(rec_loss)
        return nll_loss, torch.mean(rec_loss), torch.mean(p_loss)

    def _calculate_adaptive_weight(self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.cfg.disc_weight
        return d_weight

    def generator_loss(self, x: torch.Tensor, global_step: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """优化 encoder/decoder/quantizer（generator 分支）。"""
        if global_step is None:
            global_step = self.global_step

        xrec, qloss = self(x)
        nll_loss, rec_loss, p_loss = self._reconstruction_loss(x, xrec)

        logits_fake = self.discriminator(xrec.contiguous())
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self._calculate_adaptive_weight(nll_loss, g_loss, self.get_last_layer())
        except RuntimeError:
            d_weight = torch.zeros((), device=x.device)

        disc_factor = adopt_weight(self.cfg.disc_factor, global_step, threshold=self.cfg.disc_start)

        loss = (
            nll_loss
            + d_weight * disc_factor * g_loss
            + self.cfg.codebook_weight * qloss.mean()
        )

        out = {
            "loss_g": loss.detach(),
            "nll_loss": nll_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "p_loss": p_loss.detach(),
            "g_loss": g_loss.detach(),
            "d_weight": d_weight.detach(),
            "disc_factor": torch.tensor(disc_factor, device=x.device),
            "xrec": xrec.detach(),
            "qloss": qloss.detach(),
        }
        return loss, out

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        xrec, _ = self(x)
        return xrec

    def discriminator_loss(self, x: torch.Tensor, global_step: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """优化 discriminator。"""
        if global_step is None:
            global_step = self.global_step

        with torch.no_grad():
            xrec, _ = self(x)

        logits_real = self.discriminator(x.contiguous().detach())
        logits_fake = self.discriminator(xrec.contiguous().detach())

        disc_factor = adopt_weight(self.cfg.disc_factor, global_step, threshold=self.cfg.disc_start)
        d_loss = disc_factor * self._disc_loss_fn(logits_real, logits_fake)

        out = {
            "loss_d": d_loss.detach(),
            "logits_real": torch.mean(logits_real.detach()),
            "logits_fake": torch.mean(logits_fake.detach()),
            "disc_factor": torch.tensor(disc_factor, device=x.device),
        }
        return d_loss, out

    # ---------- optimizers helper (optional) ----------
    def configure_optimizers(self, lr: float, betas=(0.5, 0.9)):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantize.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=betas,
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return opt_g, opt_d
