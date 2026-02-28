import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop

def psnr(gt: np.ndarray, 
         pred: np.ndarray,
         mask: Optional[np.ndarray] = None,
         use_population_range: Optional[bool] = False,
         dynamic_range: Optional[tuple] = None) -> float:
    """
    Compute Peak Signal to Noise Ratio metric (PSNR)
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).
    use_population_range : bool, optional
        When a predefined population wide dynamic range should be used.
        gt and pred will also be clipped to these values.
    dynamic_range : tuple, optional
        Predefined dynamic range (min, max) for the data. Required if
        use_population_range is True.

    Returns
    -------
    psnr : float
        Peak signal to noise ratio.
    """
    if mask is None:
        mask = np.ones(gt.shape)
    else:
        # Binarize mask
        mask = np.where(mask > 0, 1., 0.)
        
    if use_population_range:
        if dynamic_range is None:
            raise ValueError("Dynamic range must be provided when use_population_range is True.")
        range_min, range_max = dynamic_range
        dynamic_range_value = range_max - range_min
        
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, range_min, range_max)
        pred = np.clip(pred, range_min, range_max)
    else:
        dynamic_range_value = gt.max() - gt.min()
        
    # Apply mask
    gt = gt[mask == 1]
    pred = pred[mask == 1]
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range_value)
    return float(psnr_value)
    
    
import numpy as np
from typing import Optional
from skimage.metrics import structural_similarity


def _adaptive_win_size(h: int, w: int, default: int = 7) -> int:
    """
    根据图像尺寸自适应选择 win_size：
      - 默认 7
      - 若 min(h,w) < 7，则选 <= min(h,w) 的最大奇数
    """
    min_hw = int(min(h, w))
    if min_hw <= 1:
        # 太小了几乎没法算 SSIM，这里给 win_size=1，后面 structural_similarity 会退化为逐点比较
        return 1
    if min_hw >= default:
        return default
    # 取 <= min_hw 的最大奇数
    if min_hw % 2 == 0:
        min_hw -= 1
    return max(min_hw, 1)


def _ssim_2d_core(gt_2d: np.ndarray,
                  pred_2d: np.ndarray,
                  data_range: float,
                  win_size: int):
    """对单张 2D 图像计算 SSIM（返回 value 和 map），带自适应 win_size。"""
    ssim_value_full, ssim_map = structural_similarity(
        gt_2d,
        pred_2d,
        data_range=data_range,
        win_size=win_size,
        full=True
    )
    return ssim_value_full, ssim_map


def ssim(gt: np.ndarray,
         pred: np.ndarray,
         mask: Optional[np.ndarray] = None,
         dynamic_range: Optional[tuple] = None) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM)

    支持：
      - 2D: (H, W)
      - 3D: (D, H, W)   （逐 slice 做 2D SSIM，再在 D 上取平均）

    mask 会在最后用来对 ssim_map 做加权平均：
      - 2D: mask 形状 (H,W)
      - 3D: mask 形状 (D,H,W)
    """
    # ---- 预处理 & 裁掉无效维度 ----
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    if mask is not None:
        mask = np.squeeze(mask)

    if dynamic_range is None:
        raise ValueError("Dynamic range must be provided.")
    range_min, range_max = dynamic_range

    # ---- clip 到指定动态范围 ----
    gt = np.clip(gt, range_min, range_max)
    pred = np.clip(pred, range_min, range_max)

    # ---- mask 阶段（只做二值掩膜，不再在这里改动 gt/pred 的形状）----
    if mask is not None:
        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
        # 如果 mask 全 0 直接返回 1（跟你原来逻辑一致）
        if np.sum(mask) == 0:
            return 1.0

    # ---- shift 到非负区间（skimage 要求）----
    if range_min < 0:
        gt = gt - range_min
        pred = pred - range_min

    dynamic_range_value = float(range_max - range_min)

    # ===================== 2D 情况 =====================
    if gt.ndim == 2:
        H, W = gt.shape
        win_size = _adaptive_win_size(H, W)
        ssim_value_full, ssim_map = _ssim_2d_core(
            gt, pred,
            data_range=dynamic_range_value,
            win_size=win_size
        )
        pad_h = pad_w = (win_size - 1) // 2

        if mask is not None:
            # 做一个安全的 crop（避免尺寸太小 pad 过大）
            pad_h_eff = min(pad_h, H // 2)
            pad_w_eff = min(pad_w, W // 2)

            if pad_h_eff > 0 and pad_w_eff > 0:
                cropped_ssim = ssim_map[
                    pad_h_eff:H - pad_h_eff,
                    pad_w_eff:W - pad_w_eff
                ]
                cropped_mask = mask[
                    pad_h_eff:H - pad_h_eff,
                    pad_w_eff:W - pad_w_eff
                ].astype(bool)
            else:
                cropped_ssim = ssim_map
                cropped_mask = mask.astype(bool)

            if np.any(cropped_mask):
                return float(cropped_ssim[cropped_mask].mean(dtype=np.float64))
            else:
                return 1.0
        else:
            return float(ssim_value_full)

    # ===================== 3D 情况：(D,H,W) =====================
    elif gt.ndim == 3:
        D, H, W = gt.shape
        win_size = _adaptive_win_size(H, W)
        pad_h = pad_w = (win_size - 1) // 2

        ssim_maps = []
        ssim_vals = []
        for z in range(D):
            v_z, m_z = _ssim_2d_core(
                gt[z], pred[z],
                data_range=dynamic_range_value,
                win_size=win_size
            )
            ssim_vals.append(v_z)
            ssim_maps.append(m_z)

        ssim_map_3d = np.stack(ssim_maps, axis=0)  # (D,H,W)
        ssim_value_full = float(np.mean(ssim_vals))

        if mask is not None:
            if mask.shape != ssim_map_3d.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} != ssim_map shape {ssim_map_3d.shape}"
                )

            pad_h_eff = min(pad_h, H // 2)
            pad_w_eff = min(pad_w, W // 2)

            if pad_h_eff > 0 and pad_w_eff > 0:
                cropped_ssim = ssim_map_3d[
                    :,
                    pad_h_eff:H - pad_h_eff,
                    pad_w_eff:W - pad_w_eff
                ]
                cropped_mask = mask[
                    :,
                    pad_h_eff:H - pad_h_eff,
                    pad_w_eff:W - pad_w_eff
                ].astype(bool)
            else:
                cropped_ssim = ssim_map_3d
                cropped_mask = mask.astype(bool)

            if np.any(cropped_mask):
                return float(cropped_ssim[cropped_mask].mean(dtype=np.float64))
            else:
                return 1.0
        else:
            return ssim_value_full

    else:
        raise ValueError(
            f"SSIM only supports 2D or 3D arrays after squeeze, got shape {gt.shape}"
        )
