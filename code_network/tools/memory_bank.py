import torch
import torch.nn.functional as F
import torch.nn as nn

from code_network.tools.fundation_models import UniversalFeatureExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, size, feature_dim, img_shape=(1, 256, 256)):
        super().__init__()
        self.size = size
        self.feature_dim = feature_dim
        self.img_shape = img_shape
        
        # 使用 register_buffer 确保这些状态会随模型保存到 ckpt 中
        # features: 存储特征向量
        self.register_buffer("features", torch.randn(size, feature_dim))
        # images: 存储原图
        self.register_buffer("images", torch.zeros(size, *img_shape))
        # ptr: 当前写入指针
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long)) 
        # n_tracked: 累计已存储的样本总数
        self.register_buffer("n_tracked", torch.zeros(1, dtype=torch.long)) 

        # 初始化 features 为单位向量，防止初始查询除零
        self.features = F.normalize(self.features, dim=1)

    @property
    def is_full(self):
        """
        判断 Bank 是否已满
        外部调用 self.retriever.bank.is_full 时会触发此函数
        """
        return self.n_tracked[0] >= self.size

    @property
    def curr_size(self):
        """
        获取当前 Bank 中实际有效的数据量
        """
        return min(self.n_tracked[0].item(), self.size)

    def update(self, new_features, new_images):
        """
        Args:
            new_features: (B, Dim)
            new_images: (B, C, H, W)
        """
        batch_size = new_features.shape[0]
        ptr = int(self.ptr)
        
        # 归一化特征
        new_features = F.normalize(new_features, dim=1)

        # 队列写入逻辑 (处理环形覆盖)
        assert batch_size <= self.size
        if ptr + batch_size <= self.size:
            self.features[ptr:ptr+batch_size] = new_features
            self.images[ptr:ptr+batch_size] = new_images
            new_ptr = ptr + batch_size
        else:
            # 分两段写
            tail = self.size - ptr
            rem = batch_size - tail
            # 第一段：填满尾部
            self.features[ptr:] = new_features[:tail]
            self.images[ptr:] = new_images[:tail]
            # 第二段：从头覆盖
            self.features[:rem] = new_features[tail:]
            self.images[:rem] = new_images[tail:]
            new_ptr = rem
            
        self.ptr[0] = new_ptr if new_ptr < self.size else 0
        self.n_tracked[0] += batch_size # 持续累加，用于判断是否 full

    def query(self, query_features):
        """
        Args:
            query_features: (B, Dim)
        Returns:
            retrieved_imgs: (B, C, H, W) 最相似的图片
            sim_scores: (B,) 相似度分数
        """
        # 1. 归一化查询向量
        q = F.normalize(query_features, dim=1) # (B, Dim)
        
        # 2. 获取当前有效的 Bank 数据
        # 必须截取有效部分，否则未填充区域(随机噪声或0)会干扰检索
        valid_size = self.curr_size
        k_features = self.features[:valid_size] # (N, Dim)
        k_images = self.images[:valid_size]     # (N, C, H, W)

        # 3. 计算相似度
        # (B, Dim) @ (Dim, N) -> (B, N)
        sim_matrix = torch.mm(q, k_features.transpose(0, 1))

        # 4. 找到最大值
        sim_scores, indices = torch.max(sim_matrix, dim=1) # (B,)

        # 5. 取出对应的图片
        retrieved_imgs = torch.index_select(k_images, 0, indices)
        
        return retrieved_imgs, sim_scores

class SemanticRetriever(nn.Module):
    def __init__(self, 
                 extractor_name, 
                 bank_size=1024, 
                 feature_dim=None, # 需要手动指定，或者通过 dummy input 自动推断
                 img_shape=(1, 256, 256),
                 **kwargs):
        """
        语义检索器：
        1. 指定大明星 (extractor_name) 提取特征
        2. 维护 Target Images 的 Bank
        3. 通过 Source Images 查询 Bank
        """
        super().__init__()
        
        # 1. 加载提取器
        self.extractor = UniversalFeatureExtractor(extractor_name, **kwargs)
        
        # 2. 自动推断特征维度 (如果未指定)
        if feature_dim is None:
            feature_dim = self._infer_dimension(img_shape)
            print(f"Auto-inferred feature dimension: {feature_dim}")

        # 3. 初始化 Bank
        self.bank = MemoryBank(bank_size, feature_dim, img_shape)

    def _infer_dimension(self, img_shape):
        """跑一次假数据来确定特征维度"""
        dummy = torch.randn(1, *img_shape)
        with torch.no_grad():
            out = self.extractor(dummy)
            flat_out = self._flatten_feature(out)
        return flat_out.shape[1]

    def _flatten_feature(self, feats):
        """
        关键函数：将各种模型输出的特征 (List, Tuple, Map) 统一转为 (B, Dim) 向量
        用于存入 Memory Bank。
        """
        # A. 如果输出是 list/tuple (DINO/SAM 经常返回多层特征)
        if isinstance(feats, (list, tuple)):
            # 策略：通常取最后一层，或者最深层的语义特征
            feat = feats[-1]
        else:
            feat = feats

        # B. 如果是 Feature Map (B, C, H, W) -> Global Average Pooling -> (B, C)
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.flatten(1)
            
        # C. 如果是 (B, N, C) (ViT 的 Sequence output) -> 取 CLS token 或 Mean
        elif feat.dim() == 3:
            # 假设 dim 1 是 sequence length (patch + cls)
            # 简单的策略是取 mean
            feat = feat.mean(dim=1) 
        
        return feat

    def update_bank(self, target_images):
        """
        输入 Target 图像 Batch -> 提取特征 -> 存入 Bank
        Args:
            target_images: (B, C, H, W)
        """
        with torch.no_grad():
            # 1. 提取原始特征
            raw_feats = self.extractor(target_images)
            # 2. 压扁成向量 (B, Dim)
            vec_feats = self._flatten_feature(raw_feats)
            # 3. 存入 Bank
            self.bank.update(vec_feats, target_images)

    def query_target(self, source_images):
        """
        输入 Source 图像 Batch -> 提取特征 -> 在 Bank 中找最像的 Target
        Args:
            source_images: (B, C, H, W)
        Returns:
            matched_targets: (B, C, H, W)
            scores: (B,)
        """
        with torch.no_grad():
            # 1. 提取 Query 特征
            raw_feats = self.extractor(source_images)
            # 2. 压扁
            vec_feats = self._flatten_feature(raw_feats)
            # 3. 查询
            matched_targets, scores = self.bank.query(vec_feats)
            
        return matched_targets, scores