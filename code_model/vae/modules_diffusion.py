"""Encoder/Decoder extracted from taming-transformers.

来源：taming/modules/diffusionmodules/model.py

说明
- 该文件较长，但这是 VQGAN 的核心卷积骨干。
- 为了可直接拷贝集成，这里保留了必要实现（与原仓库结构保持一致）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def Normalize(in_channels):
    """
    修改说明：
    将原本的 GroupNorm 替换为 BatchNorm2d。
    这是实现 AdaBN (Adaptive BatchNorm) 或 TTA (Test-Time Adaptation) 的基础。
    
    参数说明：
    - num_features: 输入通道数
    - eps: 保持为 1e-6 以维持数值稳定性
    - affine: True 表示包含可学习的参数 (gamma, beta)
    - track_running_stats: 默认为 True，这对于 AdaBN 很关键
    """
    return torch.nn.BatchNorm2d(num_features=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (swap)
        h_ = torch.bmm(v, w_)  # b,c,hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_mid_attn=True,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_mid_attn = use_mid_attn

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # ---- debug info buffers ----
        self._attn_debug = bool(ignorekwargs.get("attn_debug", True))  # 默认 True：你要“初始化时输出”
        attn_resolutions = tuple(attn_resolutions) if isinstance(attn_resolutions, (list, tuple)) else (attn_resolutions,)
        self._attn_resolutions_cfg = attn_resolutions
        self._attn_inserted_levels = []  # list of dicts

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            # 这一层是否会插 attn（注意：curr_res 在 level 内不变，所以要么全插要么不插）
            use_attn_this_level = (curr_res in attn_resolutions)

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if use_attn_this_level:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            # 记录 debug
            self._attn_inserted_levels.append({
                "level": i_level,
                "curr_res": curr_res,
                "use_attn": use_attn_this_level,
                "num_attn_blocks": len(attn),
                "num_res_blocks": self.num_res_blocks,
                "channels": block_in,
            })

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        if self.use_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        else:
            self.mid.attn_1 = None
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

        # ---- print debug info (init-time) ----
        if self._attn_debug:
            # 实际会出现的各 level 分辨率序列（非常关键）
            actual_res_list = [d["curr_res"] for d in self._attn_inserted_levels]
            used_res_list = [d["curr_res"] for d in self._attn_inserted_levels if d["use_attn"]]

            total_down_attn = sum(d["num_attn_blocks"] for d in self._attn_inserted_levels)
            total_mid_attn = 1 if (self.use_mid_attn and self.mid.attn_1 is not None) else 0
            total_attn = total_down_attn + total_mid_attn

            print("\n" + "=" * 92)
            print("[Encoder] Attention usage summary")
            print("-" * 92)
            print(f"resolution={resolution}, ch_mult={tuple(ch_mult)}, num_res_blocks={num_res_blocks}")
            print(f"attn_resolutions(cfg)={attn_resolutions} | use_mid_attn={self.use_mid_attn}")
            print(f"actual level resolutions={tuple(actual_res_list)}")
            print(f"matched resolutions (down path)={tuple(used_res_list) if len(used_res_list)>0 else '()'}")
            print(f"attn blocks: down={total_down_attn}, mid={total_mid_attn}, total={total_attn}")
            print("-" * 92)
            for d in self._attn_inserted_levels:
                flag = "ON " if d["use_attn"] else "OFF"
                print(
                    f"  level={d['level']:>2}  res={d['curr_res']:>4}  attn={flag}  "
                    f"attn_blocks={d['num_attn_blocks']:>2}/{d['num_res_blocks']}  ch={d['channels']}"
                )
            if self.use_mid_attn:
                print(f"  mid-attn: ON  (ch={block_in})")
            else:
                print("  mid-attn: OFF")
            print("=" * 92 + "\n")


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        
        # [修改] 根据 use_mid_attn 决定是否执行
        if self.use_mid_attn:
            h = self.mid.attn_1(h)
            
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        use_mid_attn=True,  # [新增参数] 默认为 True
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.use_mid_attn = use_mid_attn  # 保存该标志位

        # compute in_ch
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # [修改] 根据 use_mid_attn 决定是否初始化 AttnBlock
        if self.use_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        else:
            self.mid.attn_1 = None

        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.act_out = torch.nn.Tanh()

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        
        # [修改] 根据 use_mid_attn 决定是否执行
        if self.use_mid_attn:
            h = self.mid.attn_1(h)
            
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.act_out(h)
        return h