from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules_diffusion import Encoder, Decoder

@dataclass
class KLVAEConfig:
    # --- 基础设置 ---
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 1
    out_ch: int = 1
    
    # --- 核心架构调整 ---
    ch: int = 64
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = ()
    use_mid_attn: bool = True
    dropout: float = 0.0

    # --- Latent Space ---
    embed_dim: int = 16  
    
    # loss
    pixelloss_weight: float = 1.0
    kl_weight: float = 1.0

    # --- [新增] 推理采样配置 ---
    # 默认模式: 'sample' (随机), 'deterministic' (取均值), 'gating' (方差门控)
    default_sample_mode: str = "sample"
    # 门控敏感度: 用于 gating 模式，值越大，对不确定性区域(伪影)抑制越强
    gating_sensitivity: float = 3.0


class KLVAE(nn.Module):
    """
    连续潜变量 VAE，支持多种推理采样策略。
    """
    def __init__(self, cfg: KLVAEConfig):
        super().__init__()
        self.cfg = cfg

        ddconfig = dict(
            double_z=True,
            z_channels=cfg.z_channels,
            resolution=cfg.resolution,
            in_channels=cfg.in_channels,
            out_ch=cfg.out_ch,
            ch=cfg.ch,
            ch_mult=list(cfg.ch_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=list(cfg.attn_resolutions),
            use_mid_attn = bool(cfg.use_mid_attn),
            dropout=cfg.dropout,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(
            double_z=False,
            z_channels=cfg.z_channels,
            resolution=cfg.resolution,
            in_channels=cfg.in_channels,
            out_ch=cfg.out_ch,
            ch=cfg.ch,
            ch_mult=list(cfg.ch_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=list(cfg.attn_resolutions),
            use_mid_attn = bool(cfg.use_mid_attn),
            dropout=cfg.dropout,
        )

        self.to_mu = nn.Conv2d(2 * cfg.z_channels, cfg.embed_dim, 1)
        self.to_logvar = nn.Conv2d(2 * cfg.z_channels, cfg.embed_dim, 1)
        self.from_z = nn.Conv2d(cfg.embed_dim, cfg.z_channels, 1)

        self.global_step = 0
        
        # [新增] 初始化采样模式
        self.sample_mode = cfg.default_sample_mode

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q(z|x)||N(0,1))
        return 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)

    def set_sample_mode(self, mode: str):
        """
        [接口] 修改采样模式
        Args:
            mode: 'sample' | 'deterministic' | 'energy_preserving'
        """
        valid_modes = ["sample", "deterministic", "energy_preserving", "gated_rescaling"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid sample mode: {mode}. Must be one of {valid_modes}")
        self.sample_mode = mode
        # print(f"[KLVAE] Switched sample mode to: {self.sample_mode}")

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        根据 self.sample_mode 执行不同的采样逻辑
        """
        if self.sample_mode == "sample":
            # 1. 标准 VAE 随机采样 (训练时必须用这个)
            # z = mu + sigma * epsilon
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        elif self.sample_mode == "deterministic":
            return mu 

        elif self.sample_mode == "energy_preserving":
            # 方案一：保持每个元素的能量 (模长) 不变
            std = torch.exp(0.5 * logvar)
            # 符号取 mu 的符号，幅度取 sqrt(mu^2 + std^2)
            # 1e-6 防止 mu 为 0 时符号计算出错
            z = torch.sign(mu) * torch.sqrt(mu**2 + std**2)
            return z
        
        elif self.sample_mode == "gated_rescaling":
            # 1. 基础能量补偿 (解决黑色空洞)
            # 你发现 2.0 倍能补回能量，那我们就以 2.0 为基准
            scale_factor = 2.0 
            
            # 2. 计算门控 (解决伪影敏感)
            # std 越大 (伪影)，tanh 越接近 1，gate 越接近 0 -> 抑制
            # std 越小 (骨头)，tanh 越接近 0，gate 越接近 1 -> 保持放大
            std = torch.exp(0.5 * logvar)
            
            # sensitivity 控制“刹车”的灵敏度
            # 建议从 3.0 开始调。如果伪影还是多，调大到 5.0；如果骨头黑了，调小到 2.0
            sensitivity = getattr(self.cfg, 'gating_sensitivity', 1.0)
            gate = 1.0 - torch.tanh(sensitivity * std)
            
            # 3. 组合
            # 逻辑：(mu * 2.0) * gate
            # 骨头区域：mu * 2.0 * 1.0 = 2.0倍 (清晰，无空洞)
            # 伪影区域：mu * 2.0 * 0.1 = 0.2倍 (被压制，不可见)
            z = (mu * scale_factor) * gate
            
            return z
        
        else:
            # Fallback
            return mu

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        
        # [修改] 使用成员方法 sample_z 代替静态方法
        z = self.sample_z(mu, logvar)
        
        return z, mu, logvar

    def decode(self, z: torch.Tensor):
        z = self.from_z(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        [修改] 返回 z，以便外部可视化或分析
        """
        z, mu, logvar = self.encode(x)
        xrec = self.decode(z)
        return xrec, mu, logvar, z  # 返回 z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 训练计算 Loss 时，为了保证 KL 散度计算的数学正确性，通常需要随机采样。
        # 这里临时强制切换为 'sample' 模式，计算完后再切回去（不破坏外部设置的状态）。
        
        prev_mode = self.sample_mode
        self.sample_mode = "sample" 
        
        xrec, mu, logvar, z = self(x)
        
        rec = torch.mean(torch.abs(x.contiguous() - xrec.contiguous()))
        kl = self.kl_divergence(mu, logvar)
        loss = self.cfg.pixelloss_weight * rec + self.cfg.kl_weight * kl
        
        # 恢复原有模式
        self.sample_mode = prev_mode

        out = {
            "loss": loss.detach(),
            "rec_loss": rec.detach(),
            "kl_loss": kl.detach(),
            "xrec": xrec.detach(),
            "z_mean_abs": torch.mean(torch.abs(z)).detach(), # 监控 latent 强度
        }
        return loss, out

    def configure_optimizers(self, lr: float, betas=(0.5, 0.9)):
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return opt