import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设这些 Adapter 在你的项目中已经定义好 (如 user 提供的代码所示)
# from code_network.dinov3.tools.dinov1_adapter import Dinov1Adapter
# from code_network.dinov3.tools.dinov2_adapter import Dinov2Adapter
from code_network.dinov3.tools.dinov3_adapter import Dinov3Adapter
# from code_network.sam.sam_adapter import SAMAdapter
# from code_network.clip.clip_adapter import CLIPAdapter
# from torchvision.models import vgg19

class UniversalFeatureExtractor(nn.Module):
    def __init__(self, model_name, **kwargs):
        """
        通用特征提取器：只负责加载模型并提取特征，不计算 Loss。
        Args:
            model_name (str): 模型名称 (e.g., 'dinov3_vits14', 'sam_vit_b', 'vgg')
            **kwargs: 传递给具体 Adapter 的参数
        """
        super().__init__()
        self.model_name = str(model_name).lower()
        self.backbone = self._load_backbone(**kwargs)
        
        # 冻结参数，只做提取
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _load_backbone(self, **kwargs):
        name = self.model_name
        
        # 1. DINO v3
        if name.startswith("dinov3"):
            print(f"Loading Feature Extractor: DINOv3 ({name})")
            return Dinov3Adapter(model_name=name, **kwargs)
            
        # 2. DINO v2
        elif name.startswith("dinov2"):
            print(f"Loading Feature Extractor: DINOv2 ({name})")
            # return Dinov2Adapter(model_name=name, **kwargs)
            raise NotImplementedError("Please import your Dinov2Adapter here")
        
        # 3. DINO v1
        elif name.startswith("dino_"):
            print(f"Loading Feature Extractor: DINOv1 ({name})")
            # return Dinov1Adapter(model_name=name, **kwargs)
            raise NotImplementedError("Please import your Dinov1Adapter here")

        # 4. SAM
        elif name.startswith("sam"):
            print(f"Loading Feature Extractor: SAM ({name})")
            # return SAMAdapter(model_name=name, **kwargs)
            raise NotImplementedError("Please import your SAMAdapter here")

        # 5. CLIP
        elif name.startswith("clip"):
            print(f"Loading Feature Extractor: CLIP ({name})")
            # return CLIPAdapter(model_name=name, **kwargs)
            raise NotImplementedError("Please import your CLIPAdapter here")

        # 6. VGG
        elif name == "vgg":
            print("Loading Feature Extractor: VGG19")
            from torchvision.models import vgg19, VGG19_Weights
            # VGG 提取器通常取中间层，这里为了简化，我们取 features 部分
            # 实际使用建议封装一个只返回特定层的 VGG 类
            vgg = vgg19(weights=VGG19_Weights.DEFAULT)
            return vgg.features

        else:
            raise ValueError(f"Unknown feature extractor name: {name}")

    def forward(self, x):
        return self.backbone.extract_cls_token(x)