"""
perception/enhanced_multimodal.py - 增强多模态感知融合

在原有多模态融合基础上增强：
- 跨模态注意力机制
- 自适应模态选择
- 多模态对齐与同步
- 模态质量评估
- 渐进式融合策略
- 上下文感知的模态加权

使用方式：
    from agi_agent.perception import EnhancedMultimodalFusion
    
    fusion = EnhancedMultimodalFusion(feature_dim=512)
    fused = fusion.fuse({
        'text': text_features,
        'image': image_features,
        'audio': audio_features
    })
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from .multimodal_fusion import MultimodalFusion, FusionStrategy


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


class ModalityQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class CrossModalAttention:
    """
    跨模态注意力机制
    
    计算不同模态之间的注意力权重，实现信息交互和融合。
    """
    
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self.attention_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def compute_attention(self, modalities: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        计算跨模态注意力权重
        
        Args:
            modalities: 模态特征字典
            
        Returns:
            注意力权重矩阵
        """
        modality_keys = list(modalities.keys())
        
        for i, key1 in enumerate(modality_keys):
            for j, key2 in enumerate(modality_keys):
                if i == j:
                    continue
                
                feat1 = modalities[key1]
                feat2 = modalities[key2]
                
                attention = self._compute_single_attention(feat1, feat2)
                self.attention_weights[key1][key2] = attention
        
        return dict(self.attention_weights)
    
    def _compute_single_attention(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        计算两个模态之间的注意力权重
        
        Args:
            feat1: 模态1特征
            feat2: 模态2特征
        
        Returns:
            注意力权重
        """
        feat1_flat = feat1.flatten()
        feat2_flat = feat2.flatten()
        
        min_len = min(len(feat1_flat), len(feat2_flat))
        if min_len == 0:
            return 0.0
        
        feat1_norm = feat1_flat[:min_len] / (np.linalg.norm(feat1_flat[:min_len]) + 1e-8)
        feat2_norm = feat2_flat[:min_len] / (np.linalg.norm(feat2_flat[:min_len]) + 1e-8)
        
        similarity = np.dot(feat1_norm, feat2_norm)
        attention = (similarity + 1) / 2
        
        return float(attention)
    
    def apply_attention(self, modalities: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        应用注意力权重到模态特征
        
        Args:
            modalities: 模态特征字典
            
        Returns:
            增强后的模态特征
        """
        self.compute_attention(modalities)
        
        enhanced = {}
        for key, feat in modalities.items():
            attention_sum = sum(self.attention_weights[key].values())
            if attention_sum > 0:
                enhanced_feat = feat.copy()
                for other_key, weight in self.attention_weights[key].items():
                    other_feat = modalities[other_key]
                    scaled_weight = weight / attention_sum
                    enhanced_feat += scaled_weight * self._resize_feature(other_feat, feat.shape)
                enhanced[key] = enhanced_feat
            else:
                enhanced[key] = feat
        
        return enhanced
    
    def _resize_feature(self, feat: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """调整特征形状以匹配目标形状"""
        if feat.shape == target_shape:
            return feat
        
        target_size = np.prod(target_shape)
        feat_flat = feat.flatten()
        
        if len(feat_flat) >= target_size:
            resized = feat_flat[:target_size].reshape(target_shape)
        else:
            resized = np.zeros(target_size)
            resized[:len(feat_flat)] = feat_flat
            resized = resized.reshape(target_shape)
        
        return resized


class ModalityQualityAssessor:
    """
    模态质量评估器
    
    评估各模态输入的质量，用于自适应融合策略。
    """
    
    def __init__(self):
        self.quality_history: Dict[str, List[ModalityQuality]] = defaultdict(list)
    
    def assess_text_quality(self, text: str) -> ModalityQuality:
        """
        评估文本质量
        
        Args:
            text: 文本内容
        
        Returns:
            质量等级
        """
        if not text or len(text.strip()) == 0:
            return ModalityQuality.POOR
        
        length = len(text.strip())
        
        if length > 200:
            return ModalityQuality.EXCELLENT
        elif length > 50:
            return ModalityQuality.GOOD
        elif length > 10:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def assess_image_quality(self, image_features: np.ndarray) -> ModalityQuality:
        """
        评估图像质量
        
        Args:
            image_features: 图像特征
        
        Returns:
            质量等级
        """
        if image_features is None or image_features.size == 0:
            return ModalityQuality.POOR
        
        variance = np.var(image_features)
        
        if variance > 0.5:
            return ModalityQuality.EXCELLENT
        elif variance > 0.2:
            return ModalityQuality.GOOD
        elif variance > 0.05:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def assess_audio_quality(self, audio_features: np.ndarray) -> ModalityQuality:
        """
        评估音频质量
        
        Args:
            audio_features: 音频特征
        
        Returns:
            质量等级
        """
        if audio_features is None or audio_features.size == 0:
            return ModalityQuality.POOR
        
        energy = np.mean(audio_features ** 2)
        
        if energy > 0.3:
            return ModalityQuality.EXCELLENT
        elif energy > 0.1:
            return ModalityQuality.GOOD
        elif energy > 0.02:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def assess_modality(self, modality: str, data: Any) -> ModalityQuality:
        """
        评估模态质量（通用接口）
        
        Args:
            modality: 模态类型
            data: 模态数据
        
        Returns:
            质量等级
        """
        modality = modality.lower()
        
        if modality == "text":
            return self.assess_text_quality(str(data))
        elif modality == "image":
            return self.assess_image_quality(data)
        elif modality == "audio":
            return self.assess_audio_quality(data)
        else:
            return ModalityQuality.FAIR
    
    def get_quality_score(self, quality: ModalityQuality) -> float:
        """
        获取质量分数
        
        Args:
            quality: 质量等级
        
        Returns:
            质量分数 (0-1)
        """
        score_map = {
            ModalityQuality.EXCELLENT: 1.0,
            ModalityQuality.GOOD: 0.75,
            ModalityQuality.FAIR: 0.5,
            ModalityQuality.POOR: 0.25,
        }
        return score_map.get(quality, 0.5)


class AdaptiveFusionStrategy:
    """
    自适应融合策略
    
    根据模态质量和任务需求动态选择最优融合策略。
    """
    
    def __init__(self, default_strategy: str = "attention"):
        self.default_strategy = default_strategy
        self.quality_assessor = ModalityQualityAssessor()
    
    def select_strategy(self, modalities: Dict[str, Any], 
                       task_type: str = "general") -> str:
        """
        选择融合策略
        
        Args:
            modalities: 模态数据字典
            task_type: 任务类型 (general | classification | generation | retrieval)
        
        Returns:
            融合策略名称
        """
        qualities = {}
        total_score = 0.0
        valid_modalities = 0
        
        for modality, data in modalities.items():
            quality = self.quality_assessor.assess_modality(modality, data)
            qualities[modality] = quality
            score = self.quality_assessor.get_quality_score(quality)
            if score > 0.25:
                total_score += score
                valid_modalities += 1
        
        if valid_modalities == 0:
            return self.default_strategy
        
        avg_quality = total_score / valid_modalities
        
        if task_type == "retrieval":
            return "concat"
        elif task_type == "generation":
            return "attention"
        elif task_type == "classification":
            if avg_quality > 0.7:
                return "attention"
            else:
                return "weighted"
        else:
            if avg_quality > 0.7:
                return "attention"
            elif avg_quality > 0.4:
                return "weighted"
            else:
                return "concat"
    
    def compute_weights(self, modalities: Dict[str, Any]) -> Dict[str, float]:
        """
        计算模态权重
        
        Args:
            modalities: 模态数据字典
            
        Returns:
            权重字典
        """
        weights = {}
        total_weight = 0.0
        
        for modality, data in modalities.items():
            quality = self.quality_assessor.assess_modality(modality, data)
            weight = self.quality_assessor.get_quality_score(quality)
            weights[modality] = weight
            total_weight += weight
        
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights


class EnhancedMultimodalFusion(MultimodalFusion):
    """
    增强多模态融合引擎
    
    在原有基础上增强：
    - 跨模态注意力机制
    - 自适应模态选择
    - 模态质量评估
    - 渐进式融合策略
    - 上下文感知的模态加权
    
    Attributes:
        cross_attention: 跨模态注意力机制
        quality_assessor: 模态质量评估器
        adaptive_strategy: 自适应融合策略
        feature_dim: 特征维度
        history: 融合历史记录
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__(strategy=FusionStrategy.ATTENTION)
        
        self.feature_dim = feature_dim
        self.cross_attention = CrossModalAttention(feature_dim=feature_dim)
        self.quality_assessor = ModalityQualityAssessor()
        self.adaptive_strategy = AdaptiveFusionStrategy()
        self.history: List[Dict[str, Any]] = []
    
    def fuse(self, modalities: Dict[str, Any], 
            task_type: str = "general",
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        增强多模态融合
        
        Args:
            modalities: 模态数据字典
            task_type: 任务类型
            context: 上下文信息
            
        Returns:
            融合结果
        """
        strategy = self.adaptive_strategy.select_strategy(modalities, task_type)
        weights = self.adaptive_strategy.compute_weights(modalities)
        
        quality_scores = {}
        for modality, data in modalities.items():
            quality = self.quality_assessor.assess_modality(modality, data)
            quality_scores[modality] = {
                "quality": quality.value,
                "score": self.quality_assessor.get_quality_score(quality),
            }
        
        processed_modalities = {}
        for modality, data in modalities.items():
            if isinstance(data, np.ndarray):
                processed_modalities[modality] = self._normalize_feature(data)
            else:
                processed_modalities[modality] = data
        
        attention_enhanced = self.cross_attention.apply_attention(processed_modalities)
        
        fusion_input = {}
        for modality in ['text', 'image', 'audio', 'video', 'sensor']:
            if modality in attention_enhanced:
                fusion_input[modality] = attention_enhanced[modality]
        
        base_result = super().fuse(fusion_input)
        
        result = {
            **base_result,
            "strategy": strategy,
            "weights": weights,
            "quality_scores": quality_scores,
            "task_type": task_type,
            "num_modalities": len(modalities),
        }
        
        self.history.append(result)
        
        return result
    
    def progressive_fuse(self, modalities: Dict[str, Any], 
                        stages: int = 3) -> Dict[str, Any]:
        """
        渐进式融合
        
        分阶段逐步融合模态，从高质量模态开始。
        
        Args:
            modalities: 模态数据字典
            stages: 融合阶段数
            
        Returns:
            融合结果
        """
        weights = self.adaptive_strategy.compute_weights(modalities)
        
        sorted_modalities = sorted(weights.items(), key=lambda x: -x[1])
        
        current_result = None
        
        for i in range(min(stages, len(sorted_modalities))):
            modality, weight = sorted_modalities[i]
            data = modalities[modality]
            
            if isinstance(data, np.ndarray):
                data = self._normalize_feature(data)
            
            if current_result is None:
                current_result = data
            else:
                current_result = self._blend_features(current_result, data, weight)
        
        return {
            "fused_feature": current_result,
            "progressive_order": [m[0] for m in sorted_modalities[:stages]],
            "stages": stages,
        }
    
    def _normalize_feature(self, feat: np.ndarray) -> np.ndarray:
        """归一化特征"""
        if feat.size == 0:
            return feat
        
        norm = np.linalg.norm(feat)
        if norm > 0:
            return feat / norm
        return feat
    
    def _blend_features(self, feat1: np.ndarray, feat2: np.ndarray, 
                        weight: float) -> np.ndarray:
        """混合两个特征"""
        target_shape = feat1.shape
        feat2_resized = self._resize_feature(feat2, target_shape)
        
        blended = (1 - weight) * feat1 + weight * feat2_resized
        return blended
    
    def _resize_feature(self, feat: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """调整特征形状"""
        if feat.shape == target_shape:
            return feat
        
        target_size = np.prod(target_shape)
        feat_flat = feat.flatten()
        
        if len(feat_flat) >= target_size:
            resized = feat_flat[:target_size].reshape(target_shape)
        else:
            resized = np.zeros(target_size)
            resized[:len(feat_flat)] = feat_flat
            resized = resized.reshape(target_shape)
        
        return resized
    
    def get_fusion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取融合历史记录
        
        Args:
            limit: 返回数量限制
            
        Returns:
            历史记录列表
        """
        return self.history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取融合统计信息
        
        Returns:
            统计信息字典
        """
        if not self.history:
            return {"error": "No fusion history"}
        
        strategy_counts = defaultdict(int)
        modality_counts = defaultdict(int)
        total_quality = 0.0
        
        for record in self.history:
            strategy_counts[record["strategy"]] += 1
            for modality in record.get("quality_scores", {}):
                modality_counts[modality] += 1
            for quality_info in record.get("quality_scores", {}).values():
                total_quality += quality_info.get("score", 0.0)
        
        avg_quality = total_quality / max(sum(modality_counts.values()), 1)
        
        return {
            "total_fusions": len(self.history),
            "strategy_distribution": dict(strategy_counts),
            "modality_distribution": dict(modality_counts),
            "average_quality": avg_quality,
        }


def get_enhanced_multimodal_fusion(feature_dim: int = 256) -> EnhancedMultimodalFusion:
    """
    获取增强多模态融合引擎实例
    
    Args:
        feature_dim: 特征维度
    
    Returns:
        EnhancedMultimodalFusion 实例
    """
    return EnhancedMultimodalFusion(feature_dim=feature_dim)