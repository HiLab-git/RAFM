import torch
# from piq import LPIPS
from code_model.vae.lpips import LPIPS
from tqdm import tqdm

from .base_model import BaseModel
from code_network import define_network
try:
    from code_util.data.mask import generateSegMask
except ImportError:
    pass

from code_network.tools.memory_bank import SemanticRetriever

import scipy.stats as stats
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class TissueMaskExtractor(nn.Module):
    """
    从 CBCT/CT (-1~1) 自动提取组织区域 mask

    特性：
      - 背景≈-1
      - 组织区域是 1~2 个最大环状连通域
      - 输出 mask: 0/1

    参数：
      bg_threshold:
          背景阈值，x <= bg_threshold 认为是背景

      keep_components:
          保留最大几个连通组织区域（通常 1 或 2）

      fill_holes:
          是否填充环内部空洞（推荐 True）

      min_area:
          小于该面积的噪声区域直接删除
    """

    def __init__(
        self,
        bg_threshold: float = -0.5,
        keep_components: int = 2,
        fill_holes: bool = True,
        min_area: int = 200,
    ):
        super().__init__()

        self.bg_threshold = bg_threshold
        self.keep_components = keep_components
        self.fill_holes = fill_holes
        self.min_area = min_area

    # ---------------------------------------------------------
    # main forward
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x: [B,1,H,W] or [1,H,W]
        return: mask [B,1,H,W] (0/1)
        """

        if x.dim() == 3:
            x = x.unsqueeze(0)

        B, C, H, W = x.shape
        assert C == 1, "Only support single-channel CT/CBCT"

        # --- Step1: threshold ---
        mask = (x > self.bg_threshold).float()

        # --- Step2: remove small noise by morphology ---
        mask = self._morph_open(mask)

        # --- Step3: keep largest connected components ---
        mask = self._keep_largest_components(mask)

        # --- Step4: fill holes (ring -> solid tissue) ---
        if self.fill_holes:
            mask = self._fill_holes(mask)

        return mask

    # ---------------------------------------------------------
    # morphology open: erosion + dilation
    # ---------------------------------------------------------
    def _morph_open(self, mask):
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)

        # erosion
        eroded = F.conv2d(mask, kernel, padding=1)
        eroded = (eroded == 9).float()

        # dilation
        dilated = F.conv2d(eroded, kernel, padding=1)
        dilated = (dilated > 0).float()

        return dilated

    # ---------------------------------------------------------
    # keep largest 1~2 connected components
    # ---------------------------------------------------------
    def _keep_largest_components(self, mask):
        """
        简化版连通域：
        torch 内部无法直接做完整 floodfill，
        所以这里用 CPU skimage 处理（医学上最稳）
        """

        import numpy as np
        from skimage.measure import label

        out = torch.zeros_like(mask)

        for b in range(mask.shape[0]):
            m = mask[b, 0].detach().cpu().numpy().astype(np.uint8)

            labeled = label(m)  # connected components
            if labeled.max() == 0:
                continue

            areas = []
            for i in range(1, labeled.max() + 1):
                area = (labeled == i).sum()
                if area >= self.min_area:
                    areas.append((area, i))

            if len(areas) == 0:
                continue

            # sort by area
            areas.sort(reverse=True)

            # keep top-K
            keep_ids = [idx for _, idx in areas[: self.keep_components]]

            keep_mask = np.isin(labeled, keep_ids).astype(np.float32)
            out[b, 0] = torch.from_numpy(keep_mask).to(mask.device)

        return out

    # ---------------------------------------------------------
    # hole filling: ring -> solid
    # ---------------------------------------------------------
    def _fill_holes(self, mask):
        """
        Fill holes inside tissue ring (binary closing)
        """

        kernel = torch.ones((1, 1, 5, 5), device=mask.device)

        # dilation
        dilated = F.conv2d(mask, kernel, padding=2)
        dilated = (dilated > 0).float()

        # erosion
        eroded = F.conv2d(dilated, kernel, padding=2)
        eroded = (eroded == kernel.numel()).float()

        return eroded

# [Deleted] IntensityMapper class has been removed.

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')
    
# ==========================================
#  辅助函数：PyTorch 原生指数采样 (GPU加速版)
# ==========================================
def sample_t_torch(batch_size, device, a=2.0):
    """
    在 [0, 1] 范围内采样 t，分布集中在 0 和 1 两端。
    逻辑：
    1. 在 [0, 1] 区间进行截断指数分布采样 (Truncated Exponential)。
    2. 随机反转一半样本 (t -> 1-t)，形成 U 型分布。
    """
    # 1. 均匀分布 u ~ U[0, 1]
    u = torch.rand(batch_size, device=device)
    
    # 2. 截断指数分布的逆变换采样 (Inverse Transform Sampling)
    # PDF: p(x) ~ exp(ax)  (这里用 exp(ax) 使得倾向于大值，后续反转逻辑会处理)
    # 实际上为了简单，我们采样倾向于 0 的分布，然后随机翻转
    # PDF: p(x) ~ exp(-ax) on [0, 1]
    # CDF(x) = (1 - exp(-ax)) / (1 - exp(-a))
    # Inverse CDF: x = -1/a * ln(1 - u * (1 - exp(-a)))
    
    # 这里实现简单的 "靠近0的指数分布"
    # x ~ Exp(a), 截断在 [0, 1]
    # 由于我们还要做 cat([t, 1-t])，所以只需要生成靠近 0 的，然后一半翻转变成靠近 1 的即可
    
    # 逆变换采样：生成倾向于 0 的 t
    # normalization_const = 1 - torch.exp(torch.tensor(-a))
    # t = -torch.log(1 - u * normalization_const) / a
    
    # 为了复刻原 scipys 代码逻辑 (exp(ax))，它其实是倾向于大的数值。
    # 我们这里采用更直观的逻辑：生成两头密的分布。
    # 方法：生成 t ~ Beta(0.5, 0.5) 也是两头密，但用户指定了"指数"。
    
    # 采用原逻辑的 PyTorch 复刻：
    # 1. 生成倾向于 0 的指数分布
    t = -torch.log(1 - u * (1 - torch.exp(torch.tensor(-a, device=device)))) / a
    
    # 2. 随机翻转：50% 的概率 t = 1 - t
    # 这样 t 就会聚集在 0 和 1 附近
    mask = torch.rand(batch_size, device=device) > 0.5
    t[mask] = 1 - t[mask]
    
    # 3. 数值稳定性截断
    t_min = 1e-5
    t_max = 1 - 1e-5
    t = t * (t_max - t_min) + t_min
    
    return t

# def lpips_loss(x,y):
#     loss_lpips = LPIPS(replace_pooling=True, reduction="none")
#     # loss_lpips = torch.compile(loss_lpips)
#     return loss_lpips(x * 0.5 + 0.5, y * 0.5 + 0.5)

def l2_squared_loss(x, y):
    return torch.mean((x - y)**2, dim=(1, 2, 3))

def l2_loss(x, y):
    return torch.sqrt(torch.mean((x - y)**2, dim=(1, 2, 3)))

def l1_loss(x, y):
    return torch.mean(torch.abs(x - y), dim=(1, 2, 3))

def huber_loss(x, y):
    data_dim = x.shape[1] * x.shape[2] * x.shape[3]
    huber_c = 0.00054 * data_dim
    loss = torch.sum((x - y)**2, dim=(1, 2, 3))
    loss = torch.sqrt(loss + huber_c**2) - huber_c
    return loss / data_dim
    
class I2IRFModel(BaseModel):
    
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.config = config
        self.loss_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_diff', 'real_diff', 'interpoint', 'mask']
        self.model_names = ['G']
        self.metric_names = ['ssim']
        self.loss_type = self.config["model"].get("loss_type", "l1")
        self.lpips_divt = True 
        # self.compile = True
        self.phase = config["phase"]
        self.L1_loss = torch.nn.L1Loss()
        if "lpips" in self.loss_type:
            self.lpips_loss = LPIPS().to(self.device)
        else:
            pass

        self.netG = define_network(config, net_type="g")

        params = list(filter(lambda p: p.requires_grad, self.netG.parameters()))

        if config["phase"] == "train":
            # define loss functions
            lr = config["network"]["lr"]
            beta1 = config["network"]["beta1"] 
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)  
        
        # --- Config for Source Type ---
        self.source_type = self.config["model"].get("source_type", "image") 
        self.mix_ratio = self.config["model"].get("mix_ratio", 0.5)
        
        self.use_source_image_aid = self.config["model"].get("use_source_image_aid", False)
        self.use_fixed_t = self.config["model"].get("use_fixed_t", False)

        # --- Config for Time Sampling ---
        self.sample_type = self.config["model"].get("sample_type", "uniform")
        self.sample_param_a = self.config["model"].get("sample_param", 2.0)

        # ============================================================
        # [NEW] Semantic Retriever Configuration
        # ============================================================
        self.use_retriever = self.config["model"].get("use_retriever", False)
        
        # Option: Only start optimizing parameters after bank is full
        self.warmup_until_full = self.config["model"].get("warmup_until_full", True)
        self.is_warming_up = False # State flag
        
        # 决定 Bank 接收哪些数据作为检索库的来源
        # "real": 只存真实的 B (Standard Unpaired)
        # "mixed": 同时存 真实 B 和 Pseudo B (Reflow Triplet)
        self.bank_input_source = self.config["model"].get("bank_input_source", "mixed")

        if self.use_retriever and self.phase == "train":
            retriever_name = self.config["model"].get("retriever_name", "dinov3_vitb16") # or "dinov2_vits14"
            bank_size = self.config["model"].get("bank_size", 1024)
            img_shape = self.config["model"].get("img_shape", (1, 256, 256)) # Check your data shape!
            
            self.retriever = SemanticRetriever(
                extractor_name=retriever_name,
                bank_size=bank_size,
                img_shape=img_shape
            ).to(self.device)
            print(f"✅ SemanticRetriever initialized with {retriever_name}, bank size {bank_size}")

        # ============================================================
        # [NEW] Tissue mask configs
        # ============================================================
        # 是否启用 mask（总开关）：你原来有 use_mask，我们保留这个语义
        self.use_mask = bool(self.config["model"].get("use_mask", False))
        self.use_masked_loss = bool(self.config["model"].get("use_masked_loss", False))
        
        # 新增：若数据没给 mask，就从 real_A 自动提取
        self.use_auto_tissue_mask = bool(self.config["model"].get("use_auto_tissue_mask", False))

        if self.use_auto_tissue_mask:
            self.tissue_mask_extractor = TissueMaskExtractor(
                bg_threshold=float(self.config["model"].get("mask_bg_threshold", -0.5)),
                keep_components=int(self.config["model"].get("mask_keep_components", 2)),
                fill_holes=bool(self.config["model"].get("mask_fill_holes", True)),
                min_area=int(self.config["model"].get("mask_min_area", 200)),
            ).to(self.device)
        else:
            self.tissue_mask_extractor = None

    def _lpips_loss(self, x, y):
        return self.lpips_loss(x * 0.5 + 0.5, y * 0.5 + 0.5)

    def get_flow_source(self, real_A):
        """Generate flow source (state at t=1)"""
        if self.source_type == "image":
            return real_A
        elif self.source_type == "noise":
            return torch.randn_like(real_A)
        elif self.source_type == "mix":
            noise = torch.randn_like(real_A)
            return (1 - self.mix_ratio) * real_A + self.mix_ratio * noise
        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")

    @staticmethod
    def _apply_mask_fill_bg(img, mask, bg_val=-1.0):
        """
        img:  [B,1,H,W]
        mask: [B,1,H,W] (0/1)
        """
        return img * mask + (1 - mask) * bg_val

    def _maybe_build_mask_from_A(self):
        """
        如果启用了 auto mask 且当前没有 mask，则从 real_A 提取。
        """
        if not self.use_mask:
            return

        if self.mask is not None:
            # ensure binary 0/1
            self.mask = (self.mask > 0.5).float()
            return

        if self.use_auto_tissue_mask and self.tissue_mask_extractor is not None:
            with torch.no_grad():
                self.mask = self.tissue_mask_extractor(self.real_A)

    def _maybe_apply_mask_to_B(self):
        """
        统一把 mask 作用到当前 self.real_B 上（包含 random B 或 matched B）。
        """
        if not self.use_mask:
            return
        if self.mask is None:
            return
        self.real_B = self._apply_mask_fill_bg(self.real_B, self.mask, bg_val=-1.0)

    def forward(self):
        # print("Running forward pass...")
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.phase == "train":
            
            # 1. Determine Flow Source (Source, t=1)
            # Note: self.real_A is the source (e.g., CBCT)
            self.flow_src = self.get_flow_source(self.real_A)

            # 2. Sample time t
            batch_size = self.real_A.shape[0]
            device = self.real_A.device
            
            if self.sample_type == "uniform":
                self.t = torch.rand(batch_size, device=device) * (1 - 2e-5) + 1e-5
            elif self.sample_type == "exponential":
                self.t = sample_t_torch(batch_size, device, a=self.sample_param_a)
            else:
                raise ValueError(f"Unknown sample_type: {self.sample_type}")

            # 3. Construct Interpolation
            # Note: self.real_B is now the retrieved Target (CT)
            self.interpoint = (1-self.t).view(-1, 1, 1, 1) * self.real_B + self.t.view(-1, 1, 1, 1) * self.flow_src 
            
            # Ground Truth Difference
            self.real_diff = (self.flow_src - self.real_B)
            
            # 4. Construct Network Input
            if self.use_source_image_aid:
                self.input_G = torch.cat([self.interpoint, self.real_A], dim=1)
            else:
                self.input_G = self.interpoint
                
            if self.use_fixed_t:
                input_t = torch.ones_like(self.t) * 0.5
            else:
                input_t = self.t
            
            # 5. Network Prediction
            self.fake_diff = self.netG(self.input_G, input_t) 
            
            # 6. Recover Predicted Target
            self.fake_B = self.interpoint - self.fake_diff * self.t.view(-1, 1, 1, 1)

            self.fake_B_up = self.fake_B.repeat(1, 3, 1, 1)
            self.real_B_up = self.real_B.repeat(1, 3, 1, 1)

            if self.use_masked_loss == True:
                self.fake_B_up = self.fake_B_up * self.mask + (1 - self.mask) * -1.0
                self.real_B_up = self.real_B_up * self.mask + (1 - self.mask) * -1.0
                self.fake_B = self.fake_B * self.mask + (1 - self.mask) * -1.0
                self.real_B = self.real_B * self.mask + (1 - self.mask) * -1.0

        else:
            # === Inference Phase ===
            forward_times = self.config["model"].get("forward_times", 1)
            solver = self.config["model"].get("solver", 'euler')
            N = self.config["model"].get("N", 1)
            
            z1 = self.get_flow_source(self.real_A)
            condition_image = self.real_A if self.use_source_image_aid else None
            current_state = z1
            
            for _ in range(forward_times):
                _, traj_uncond_x0 = sample_ode_generative(
                    self.netG, 
                    z1=current_state, 
                    N=N, 
                    solver=solver, 
                    use_source_image_aid=self.use_source_image_aid, 
                    condition_image=condition_image,
                    use_fixed_t=self.use_fixed_t
                )
                output_B = traj_uncond_x0[-1]
                current_state = output_B 
            self.fake_B = output_B
            if self.use_masked_loss == True:
                # self.fake_B_up = self.fake_B_up * self.mask + (1 - self.mask) * -1.0
                # self.real_B_up = self.real_B_up * self.mask + (1 - self.mask) * -1.0
                self.fake_B = self.fake_B * self.mask + (1 - self.mask) * -1.0
                self.real_B = self.real_B * self.mask + (1 - self.mask) * -1.0

    def cal_loss_G(self):
        # ... (Your existing loss calculation logic, omitted for brevity) ...
        # Ensure you use self.real_B here, which has been replaced by the retrieved image
        if self.loss_type == 'lpips':
            if self.lpips_divt:
                loss = self._lpips_loss(self.fake_B_up, self.real_B_up) / self.t.squeeze()
            else:
                loss = self._lpips_loss(self.fake_B_up, self.real_B_up)
            loss = loss.mean()
        elif self.loss_type == 'lpips-huber':
            loss_huber = huber_loss(self.fake_diff,self.real_diff)
            loss_lp = self._lpips_loss(self.fake_B_up, self.real_B_up)
            if self.lpips_divt:
                loss = (1-(self.t).squeeze()) * loss_huber + loss_lp / self.t.squeeze()
            else:
                loss = (1-(self.t).squeeze()) * loss_huber + loss_lp
            loss = loss.mean()
        elif self.loss_type == 'l2-squared':
            loss = l2_squared_loss(self.fake_diff, self.real_diff)
        elif self.loss_type == 'l2':
            loss = l2_loss(self.fake_diff, self.real_diff)
        elif self.loss_type == "l1":
            loss = self.L1_loss(self.fake_B,self.real_B)
        elif self.loss_type == "lpips-l1-fake":
            loss_lpips = self._lpips_loss(self.fake_B_up, self.real_B_up).mean()
            loss_l1 = self.L1_loss(self.fake_B,self.real_B)
            loss = loss_lpips + loss_l1
        elif self.loss_type == "lpips-l1-diff":
            loss_lpips = self._lpips_loss(self.fake_diff, self.real_diff).mean()
            loss_l1 = self.L1_loss(self.fake_diff,self.real_diff)
            loss = loss_lpips + loss_l1
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        self.loss_G = loss
        self.loss_G_lambda = self.loss_G * self.config["network"]["lambda_L1"] 
       
    def set_input(self, input):
        """
        解包数据 +（可选）自动从 real_A 提 mask +（可选）对 real_B 做组织范围对齐 + 检索替换
        """
        self.real_A = input["A"]["data"].to(self.device)

        # dataset-provided mask (optional)
        self.mask = None
        if input.get("Mask", None) is not None:
            self.mask = input["Mask"]["data"].to(self.device)

        # unpack B sources
        if "Real_B" in input:
            self.random_real_B = input["Real_B"]["data"].to(self.device)
            b_path = input["Real_B"]["params"].get("path")
        else:
            self.random_real_B = input["B"]["data"].to(self.device)
            b_path = input["B"]["params"].get("path")

        self.pseudo_B = input["Pseudo_B"]["data"].to(self.device) if "Pseudo_B" in input else None

        self.image_paths = {
            "A_path": input["A"]["params"].get("path"),
            "B_path": b_path,
            "Mask_path": input.get("Mask", {}).get("params", {}).get("path") if input.get("Mask", None) is not None else None,
        }

        # ===========================
        # [NEW] build mask from A if needed
        # ===========================
        self._maybe_build_mask_from_A()

        # default optimization target
        self.real_B = self.random_real_B

        # ===========================
        # [NEW] apply A-mask to B (align tissue range)
        # ===========================
        self._maybe_apply_mask_to_B()

        # ===========================
        # Retriever logic (unchanged semantics)
        # ===========================
        if self.use_retriever and self.phase == "train":
            bank_candidates = []

            # (A) add random real B (do NOT mask for bank by default)
            if ("real" in self.bank_input_source) or ("mixed" in self.bank_input_source):
                bank_candidates.append(self.random_real_B)

            # (B) add pseudo B (optionally masked by existing mask for bank safety)
            if (self.pseudo_B is not None) and ("mixed" in self.bank_input_source):
                curr_pseudo_B = self.pseudo_B
                # 如果你希望 pseudo_B 只在组织区域可信：mask 外填 -1
                if self.mask is not None:
                    curr_pseudo_B = self._apply_mask_fill_bg(curr_pseudo_B, self.mask, bg_val=-1.0)
                bank_candidates.append(curr_pseudo_B)

            # update bank
            if len(bank_candidates) > 0:
                bank_input = torch.cat(bank_candidates, dim=0)
                self.retriever.update_bank(bank_input.detach())

            # warmup until full
            if self.warmup_until_full and (not self.retriever.bank.is_full):
                self.is_warming_up = True
                return
            self.is_warming_up = False

            # query & replace real_B
            if self.retriever.bank.curr_size > 0:
                matched_B, scores = self.retriever.query_target(self.real_A)
                self.real_B = matched_B

                # [NEW] re-apply A-mask to matched_B as well
                self._maybe_apply_mask_to_B()

    def optimize_parameters(self):
        if self.use_retriever and self.warmup_until_full and self.is_warming_up:
            # If we are in warm-up mode and bank is not full:
            # We skip forward pass and optimization.
            # set_input has already updated the bank.
            # We can optionally print a log every N steps
            if getattr(self, "global_step", 0) % 10 == 0:
                print(f"[Warmup] Bank filling... Current size: {self.retriever.bank.ptr.item()}/{self.retriever.bank.size}")
            return
        return super().optimize_parameters()

class EDMPrecondVel(torch.nn.Module):
    def __init__(self,
        model,
        sigma_min=0,
        sigma_max=float('inf'),
        sigma_data=0.5,
    ):
        super().__init__()
        self.label_dim = getattr(model, 'class_dim', 0)
        self.augment_dim = getattr(model, 'augment_dim', 0)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x_, sigma_, class_labels=None, augment_labels=None, **model_kwargs):
        x = x_
        sigma = sigma_

        if augment_labels is None and self.augment_dim > 0:
            augment_labels = torch.zeros([x.shape[0], self.augment_dim], device=x.device)

        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([x.shape[0], self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out  = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in   = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            c_in * x,
            noise_labels=c_noise.flatten(),
            class_labels=class_labels,
            augment_labels=augment_labels,
            **model_kwargs
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        v_t = (x_ - D_x) / sigma_.view(-1, 1, 1, 1)
        return v_t

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
def sample_ode_generative(model, z1, N, use_tqdm=False, solver='euler', label=None, inversion=False, time_schedule=None, sampler='default', use_source_image_aid=False, condition_image=None, use_fixed_t=False):
    assert solver in ['euler', 'heun']
    assert len(z1.shape) == 4
    if inversion:
        assert sampler == 'default'
    tq = tqdm if use_tqdm else lambda x: x

    if solver == 'heun' and N % 2 == 0:
        raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
        N = (N + 1) // 2
    traj = [] 
    x0hat_list = []
    z1 = z1.detach()
    z = z1.clone()
    batchsize = z.shape[0]
    
    if time_schedule is not None:
        time_schedule = time_schedule + [0]
    else:
        t_func = lambda i: i / N
        if inversion:
            time_schedule = [t_func(i) for i in range(0,N)] + [1]
            time_schedule[0] = 1e-3
        else:
            time_schedule = [t_func(i) for i in reversed(range(1,N+1))] + [0]
            time_schedule[0] = 1-1e-5

    cnt = 0

    traj.append(z.detach().clone())
    for i in tq(range(len(time_schedule[:-1]))):
        t = torch.ones((batchsize), device=z1.device) * time_schedule[i]
        t_next = torch.ones((batchsize), device=z1.device) * time_schedule[i+1]
        dt = t_next[0] - t[0]
        
        if use_source_image_aid:
            assert condition_image is not None
            model_input = torch.cat([z, condition_image], dim=1)
        else:
            model_input = z
            
        if use_fixed_t:
            input_t = torch.ones_like(t) * 0.5
        else:
            input_t = t
            
        vt = model(model_input, input_t, label)
        
        if inversion:
            x0hat = z + vt * (1-t).view(-1,1,1,1) 
        else:
            x0hat = z - vt * t.view(-1,1,1,1) 

        # Heun's correction
        if solver == 'heun' and cnt < N - 1:
            assert use_source_image_aid == False
            if sampler == 'default' or inversion:
                z_next = z.detach().clone() + vt * dt
            elif sampler == 'new':
                z_next = (1 - t_next.view(-1,1,1,1)) * x0hat + t_next.view(-1,1,1,1) * z1
            else:
                raise NotImplementedError(f"Sampler {sampler} not implemented.")

            vt_next = model(z_next, t_next, label)
            vt = (vt + vt_next) / 2

            if inversion:
                x0hat = z_next + vt_next * (1-t_next).view(-1,1,1,1) 
            else:
                x0hat = z_next - vt_next * t_next.view(-1,1,1,1) 
    
        x0hat_list.append(x0hat)
        
        if inversion:
            x0hat = z + vt * (1-t).view(-1,1,1,1)
        else:
            x0hat = z - vt * t.view(-1,1,1,1)
        
        if sampler == 'default' or inversion:
            z = z.detach().clone() + vt * dt
        elif sampler == 'new':
            z = (1 - t_next.view(-1,1,1,1)) * x0hat + t_next.view(-1,1,1,1) * z1
        else:
            raise NotImplementedError(f"Sampler {sampler} not implemented.")
        cnt += 1
        traj.append(z.detach().clone())

    return traj, x0hat_list