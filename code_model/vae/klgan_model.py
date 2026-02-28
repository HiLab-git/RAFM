from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .klvae_model import KLVAE, KLVAEConfig
from .modules_discriminator import NLayerDiscriminator, weights_init
from .lpips import LPIPS
from .vqgan_model import hinge_d_loss, vanilla_d_loss, adopt_weight


@dataclass
class KLGANConfig(KLVAEConfig):
    # discriminator/loss
    disc_start: int = 0
    disc_in_channels: int = 1
    disc_num_layers: int = 1
    disc_ndf: int = 64
    disc_factor: float = 1.0
    disc_weight: float = 0.8
    disc_loss: str = "hinge"  # hinge|vanilla

    perceptual_weight: float = 1.0
    use_lpips: bool = False


class KLGAN(nn.Module):
    """
    标准训练范式：
    - 每次迭代先算一次 ae(x)->xrec,mu,logvar
    - G loss 用 xrec（需要梯度）
    - D loss 用 xrec.detach()（不需要梯度）
    这样避免在 discriminator_loss 内部重复跑 ae(x)。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ae = KLVAE(cfg)

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

        self.global_step = 0

    # -------------------------
    # forward helpers
    # -------------------------
    def forward(self, x: torch.Tensor):
        # 与你原来一致：forward 直接走 AE
        return self.ae(x)

    def forward_ae(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """显式 AE forward，便于训练步骤复用结果。"""
        return self.ae(x)

    def get_last_layer(self):
        return self.ae.decoder.conv_out.weight

    # -------------------------
    # losses
    # -------------------------
    def reconstruction_loss(self, x: torch.Tensor, xrec: torch.Tensor):
        # L1
        rec = torch.abs(x.contiguous() - xrec.contiguous())

        # LPIPS（通常吃 3 通道；如果你的 LPIPS 已经处理 1 通道就不用重复）
        if self.lpips is not None and self.cfg.perceptual_weight > 0:
            p = self.lpips(x.contiguous(), xrec.contiguous())
            rec = rec + self.cfg.perceptual_weight * p
        else:
            p = torch.zeros((), device=x.device)

        nll = rec.mean()
        return nll, rec.mean().detach(), p.mean().detach()

    def _calculate_adaptive_weight(
        self,
        nll_loss: torch.Tensor,
        g_loss: torch.Tensor,
        last_layer: torch.Tensor,
    ):
        # 这就是 VQGAN 标准做法（你原来没错）
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True, create_graph=False)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True, create_graph=False)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.cfg.disc_weight

    def generator_loss_from_outputs(
        self,
        x: torch.Tensor,
        xrec: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        标准：输入已经算好的 xrec/mu/logvar，不要再跑 ae(x)。
        """
        if global_step is None:
            global_step = self.global_step

        nll_loss, rec_loss, p_loss = self.reconstruction_loss(x, xrec)
        kl = self.ae.kl_divergence(mu, logvar)

        # GAN loss（对 fake）
        logits_fake = self.discriminator(xrec.contiguous())
        g_loss = -logits_fake.mean()

        disc_factor = adopt_weight(self.cfg.disc_factor, global_step, threshold=self.cfg.disc_start)

        # adaptive weight：通常只在 disc_start 之后才有意义
        if disc_factor > 0:
            try:
                d_weight = self._calculate_adaptive_weight(nll_loss, g_loss, self.get_last_layer())
            except RuntimeError:
                d_weight = torch.zeros((), device=x.device)
        else:
            d_weight = torch.zeros((), device=x.device)

        loss = nll_loss + self.cfg.kl_weight * kl + d_weight * disc_factor * g_loss

        out = {
            "loss_g": loss.detach(),
            "nll_loss": nll_loss.detach(),
            "rec_loss": rec_loss,
            "p_loss": p_loss,
            "kl_loss": kl.detach(),
            "g_loss": g_loss.detach(),
            "d_weight": d_weight.detach(),
            "disc_factor": torch.tensor(disc_factor, device=x.device),
            "xrec": xrec.detach(),  # 日志/可视化用
            "z": z.detach()
        }
        return loss, out

    def discriminator_loss_from_fake(
        self,
        real: torch.Tensor,
        fake_detached: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        标准：D loss 只吃 real + fake_detached，不要内部再算 AE。
        """
        if global_step is None:
            global_step = self.global_step

        logits_real = self.discriminator(real.contiguous().detach())
        logits_fake = self.discriminator(fake_detached.contiguous().detach())

        disc_factor = adopt_weight(self.cfg.disc_factor, global_step, threshold=self.cfg.disc_start)
        d_loss = disc_factor * self._disc_loss_fn(logits_real, logits_fake)

        out = {
            "loss_d": d_loss.detach(),
            "logits_real": logits_real.mean().detach(),
            "logits_fake": logits_fake.mean().detach(),
            "disc_factor": torch.tensor(disc_factor, device=real.device),
        }
        return d_loss, out

    # 保留你原来的 API：向后兼容
    def generator_loss(self, x: torch.Tensor, global_step: Optional[int] = None):
        xrec, mu, logvar, z = self.forward_ae(x)
        return self.generator_loss_from_outputs(x, xrec, mu, logvar, z, global_step)
    
    def generator_loss_pair(
        self,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Conditional reconstruction:
        xrec = AE(input_img)
        reconstruction loss against target_img
        KL from encoding(input_img)
        GAN loss from discriminator(xrec)

        Keeps backward-compatible: does NOT change generator_loss(x).
        """
        if global_step is None:
            global_step = self.global_step

        # 1) forward AE on input_img
        xrec, mu, logvar, z = self.forward_ae(input_img)

        # 2) compute losses using target_img as "x"
        #    - rec/LPIPS between target_img and xrec
        #    - kl from mu/logvar (input encoding)
        #    - g_loss from discriminator(xrec)
        loss, out = self.generator_loss_from_outputs(
            x=target_img,
            xrec=xrec,
            mu=mu,
            logvar=logvar,
            z=z,
            global_step=global_step,
        )

        # optional: log input/target for debugging
        out["input_img"] = input_img.detach()
        out["target_img"] = target_img.detach()
        out['mu'] = mu.detach()
        out['logvar'] = logvar.detach()
        return loss, out


    def discriminator_loss(self, x: torch.Tensor, global_step: Optional[int] = None):
        # 旧 API：仍然能用，但建议上层改成复用 fake
        with torch.no_grad():
            xrec, _, _, _= self.forward_ae(x)
        return self.discriminator_loss_from_fake(x, xrec, global_step)

    def configure_optimizers(self, lr: float, betas=(0.5, 0.9)):
        opt_g = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return opt_g, opt_d