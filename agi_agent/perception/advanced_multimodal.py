import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import torch
import torch.nn as nn

from .enhanced_multimodal import (
    ModalityType, ModalityQuality, CrossModalAttention,
    ModalityQualityAssessor, AdaptiveFusionStrategy, EnhancedMultimodalFusion
)


class FusionStage(Enum):
    FEATURE_LEVEL = "feature_level"
    ATTENTION_LEVEL = "attention_level"
    SEMANTIC_LEVEL = "semantic_level"
    DECISION_LEVEL = "decision_level"


class ContextType(Enum):
    GENERAL = "general"
    QUESTION_ANSWERING = "qa"
    IMAGE_CAPTIONING = "captioning"
    SPEECH_RECOGNITION = "speech"
    TEXT_CLASSIFICATION = "classification"
    MULTIMODAL_RETRIEVAL = "retrieval"


class AdvancedCrossModalAttention:
    def __init__(self, feature_dim: int = 256, num_heads: int = 4):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.intra_modality_weights: Dict[str, np.ndarray] = {}
        self.inter_modality_weights: Dict[str, Dict[str, np.ndarray]] = {}
        
        self.scaling_factor = np.sqrt(self.head_dim)
    
    def compute_intra_modality_attention(self, modality_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        enhanced = {}
        
        for modality, features in modality_features.items():
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            num_tokens = features.shape[0]
            attention_matrix = self._compute_attention_matrix(features)
            
            enhanced_features = attention_matrix @ features
            enhanced[modality] = enhanced_features
        
        self.intra_modality_weights = {k: v for k, v in enhanced.items()}
        return enhanced
    
    def compute_inter_modality_attention(self, modality_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        modalities = list(modality_features.keys())
        enhanced = {m: modality_features[m].copy() for m in modalities}
        
        for i, source_modality in enumerate(modalities):
            for j, target_modality in enumerate(modalities):
                if i == j:
                    continue
                
                source_feat = modality_features[source_modality]
                target_feat = modality_features[target_modality]
                
                attention = self._compute_cross_modality_attention(source_feat, target_feat)
                
                if source_modality not in self.inter_modality_weights:
                    self.inter_modality_weights[source_modality] = {}
                self.inter_modality_weights[source_modality][target_modality] = attention
                
                enhanced[target_modality] = enhanced[target_modality] + attention * self._resize_feature(source_feat, target_feat.shape)
        
        return enhanced
    
    def _compute_attention_matrix(self, features: np.ndarray) -> np.ndarray:
        Q = features
        K = features
        V = features
        
        attention_scores = (Q @ K.T) / self.scaling_factor
        attention_matrix = self._softmax(attention_scores)
        
        return attention_matrix
    
    def _compute_cross_modality_attention(self, source: np.ndarray, target: np.ndarray) -> float:
        source_flat = source.flatten()
        target_flat = target.flatten()
        
        min_len = min(len(source_flat), len(target_flat))
        if min_len == 0:
            return 0.0
        
        source_norm = source_flat[:min_len] / (np.linalg.norm(source_flat[:min_len]) + 1e-8)
        target_norm = target_flat[:min_len] / (np.linalg.norm(target_flat[:min_len]) + 1e-8)
        
        similarity = np.dot(source_norm, target_norm)
        return float((similarity + 1) / 2)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _resize_feature(self, feat: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        if feat.shape == target_shape:
            return feat
        
        target_size = np.prod(target_shape)
        feat_flat = feat.flatten()
        
        if len(feat_flat) >= target_size:
            return feat_flat[:target_size].reshape(target_shape)
        else:
            resized = np.zeros(target_size)
            resized[:len(feat_flat)] = feat_flat
            return resized.reshape(target_shape)
    
    def joint_attention(self, modality_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        intra_enhanced = self.compute_intra_modality_attention(modality_features)
        inter_enhanced = self.compute_inter_modality_attention(intra_enhanced)
        
        return inter_enhanced
    
    def get_attention_summary(self) -> Dict[str, Any]:
        summary = {
            "intra_modality": {},
            "inter_modality": {}
        }
        
        for modality, weights in self.intra_modality_weights.items():
            summary["intra_modality"][modality] = {
                "shape": weights.shape,
                "mean_attention": float(np.mean(weights))
            }
        
        for source, targets in self.inter_modality_weights.items():
            summary["inter_modality"][source] = {}
            for target, weight in targets.items():
                summary["inter_modality"][source][target] = weight
        
        return summary


class AdvancedModalityQualityAssessor(ModalityQualityAssessor):
    def __init__(self):
        super().__init__()
        self.anomaly_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.thresholds = {
            "text": {
                "excellent": 200,
                "good": 50,
                "fair": 10,
                "poor": 0
            },
            "image": {
                "excellent": 0.5,
                "good": 0.2,
                "fair": 0.05,
                "poor": 0
            },
            "audio": {
                "excellent": 0.3,
                "good": 0.1,
                "fair": 0.02,
                "poor": 0
            }
        }
    
    def assess_text_quality(self, text: str) -> ModalityQuality:
        if not text or len(text.strip()) == 0:
            return ModalityQuality.POOR
        
        length = len(text.strip())
        has_alpha = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        has_punctuation = any(c in '.,!?;:()[]{}' for c in text)
        coherence_score = sum([has_alpha, has_number, has_punctuation]) / 3.0
        
        if length > self.thresholds["text"]["excellent"] and coherence_score > 0.6:
            return ModalityQuality.EXCELLENT
        elif length > self.thresholds["text"]["good"]:
            return ModalityQuality.GOOD
        elif length > self.thresholds["text"]["fair"]:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def assess_image_quality(self, image_features: np.ndarray) -> ModalityQuality:
        if image_features is None or image_features.size == 0:
            return ModalityQuality.POOR
        
        variance = np.var(image_features)
        entropy = self._compute_entropy(image_features)
        mean_activation = np.mean(np.abs(image_features))
        
        quality_score = 0.4 * variance + 0.3 * entropy + 0.3 * mean_activation
        
        if quality_score > self.thresholds["image"]["excellent"]:
            return ModalityQuality.EXCELLENT
        elif quality_score > self.thresholds["image"]["good"]:
            return ModalityQuality.GOOD
        elif quality_score > self.thresholds["image"]["fair"]:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def assess_audio_quality(self, audio_features: np.ndarray) -> ModalityQuality:
        if audio_features is None or audio_features.size == 0:
            return ModalityQuality.POOR
        
        energy = np.mean(audio_features ** 2)
        spectral_flatness = self._compute_spectral_flatness(audio_features)
        zero_crossing_rate = self._compute_zero_crossing_rate(audio_features)
        
        quality_score = 0.5 * energy + 0.3 * spectral_flatness + 0.2 * zero_crossing_rate
        
        if quality_score > self.thresholds["audio"]["excellent"]:
            return ModalityQuality.EXCELLENT
        elif quality_score > self.thresholds["audio"]["good"]:
            return ModalityQuality.GOOD
        elif quality_score > self.thresholds["audio"]["fair"]:
            return ModalityQuality.FAIR
        else:
            return ModalityQuality.POOR
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        hist, _ = np.histogram(data, bins=32, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return float(min(entropy / 5, 1.0))
    
    def _compute_spectral_flatness(self, audio: np.ndarray) -> float:
        if len(audio) < 2:
            return 0.0
        
        fft_vals = np.abs(np.fft.fft(audio))[:len(audio)//2]
        if np.sum(fft_vals) == 0:
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(fft_vals + 1e-10)))
        arithmetic_mean = np.mean(fft_vals)
        return float(min(geometric_mean / (arithmetic_mean + 1e-10), 1.0))
    
    def _compute_zero_crossing_rate(self, audio: np.ndarray) -> float:
        if len(audio) < 2:
            return 0.0
        
        crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        return float(min(crossings / len(audio), 1.0))
    
    def detect_anomaly(self, modality: str, data: Any, 
                      window_size: int = 10) -> Dict[str, Any]:
        quality = self.assess_modality(modality, data)
        score = self.get_quality_score(quality)
        
        self.quality_history[modality].append(quality)
        
        if len(self.quality_history[modality]) >= window_size:
            recent_scores = [self.get_quality_score(q) for q in self.quality_history[modality][-window_size:]]
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            is_anomaly = score < mean_score - 2 * std_score
            
            anomaly_info = {
                "modality": modality,
                "quality": quality.value,
                "score": score,
                "is_anomaly": is_anomaly,
                "mean_score": mean_score,
                "std_score": std_score,
                "timestamp": np.datetime64('now')
            }
            
            self.anomaly_history[modality].append(anomaly_info)
            return anomaly_info
        
        return {
            "modality": modality,
            "quality": quality.value,
            "score": score,
            "is_anomaly": False,
            "message": "Insufficient history for anomaly detection"
        }
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        summary = {}
        
        for modality, anomalies in self.anomaly_history.items():
            total_anomalies = sum(1 for a in anomalies if a["is_anomaly"])
            summary[modality] = {
                "total_quality_checks": len(anomalies),
                "total_anomalies": total_anomalies,
                "anomaly_rate": total_anomalies / max(len(anomalies), 1),
                "latest_anomalies": [a for a in anomalies[-5:] if a["is_anomaly"]]
            }
        
        return summary


class ContextAwareFusionStrategy(AdaptiveFusionStrategy):
    def __init__(self):
        super().__init__()
        self.context_weights: Dict[str, Dict[str, float]] = {}
        self.task_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.performance_window: int = 20
    
    def select_strategy(self, modalities: Dict[str, Any], 
                       task_type: str = "general",
                       context: Optional[Dict[str, Any]] = None) -> str:
        strategy = super().select_strategy(modalities, task_type)
        
        if context:
            context_type = context.get("type", "general")
            strategy = self._adjust_strategy_by_context(strategy, context_type, modalities)
        
        return strategy
    
    def compute_context_aware_weights(self, modalities: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        base_weights = super().compute_weights(modalities)
        
        if not context:
            return base_weights
        
        context_type = context.get("type", "general")
        context_strength = context.get("strength", 1.0)
        
        context_bias = self._get_context_bias(context_type)
        
        adjusted_weights = {}
        for modality, weight in base_weights.items():
            bias = context_bias.get(modality, 1.0)
            adjusted_weights[modality] = weight * (1 + (bias - 1) * context_strength)
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        self.context_weights[context_type] = adjusted_weights
        
        return adjusted_weights
    
    def _adjust_strategy_by_context(self, strategy: str, 
                                   context_type: str,
                                   modalities: Dict[str, Any]) -> str:
        context_preferences = {
            "qa": {"text": 0.8, "image": 0.2},
            "captioning": {"image": 0.7, "text": 0.3},
            "speech": {"audio": 0.9, "text": 0.1},
            "classification": {"text": 0.6, "image": 0.4},
            "retrieval": {"text": 0.5, "image": 0.5}
        }
        
        preference = context_preferences.get(context_type)
        if preference:
            has_primary = any(m in modalities for m in preference.keys())
            if has_primary and strategy == "concat":
                return "attention"
        
        return strategy
    
    def _get_context_bias(self, context_type: str) -> Dict[str, float]:
        bias_map = {
            "qa": {"text": 1.3, "image": 0.8, "audio": 0.7},
            "captioning": {"image": 1.3, "text": 1.1, "audio": 0.6},
            "speech": {"audio": 1.4, "text": 0.9, "image": 0.5},
            "classification": {"text": 1.2, "image": 1.1, "audio": 0.8},
            "retrieval": {"text": 1.1, "image": 1.1, "audio": 0.9}
        }
        
        return bias_map.get(context_type, {"text": 1.0, "image": 1.0, "audio": 1.0})
    
    def update_performance(self, task_type: str, strategy: str, score: float):
        self.task_performance[task_type][strategy] = (
            self.task_performance[task_type][strategy] * (self.performance_window - 1) + score
        ) / self.performance_window
    
    def get_best_strategy(self, task_type: str) -> str:
        if task_type not in self.task_performance:
            return self.default_strategy
        
        strategies = self.task_performance[task_type]
        if not strategies:
            return self.default_strategy
        
        return max(strategies, key=strategies.get)


class ProgressiveMultimodalFusion(EnhancedMultimodalFusion):
    def __init__(self, feature_dim: int = 256):
        super().__init__(feature_dim)
        self.stage_results: List[Dict[str, Any]] = []
        self.cross_attention = AdvancedCrossModalAttention(feature_dim)
    
    def progressive_fuse(self, modalities: Dict[str, Any], 
                        stages: int = 4,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        weights = self.adaptive_strategy.compute_context_aware_weights(modalities, context)
        sorted_modalities = sorted(weights.items(), key=lambda x: -x[1])
        
        stage_outputs = []
        current_result = None
        
        for stage_idx in range(min(stages, len(sorted_modalities))):
            modality, weight = sorted_modalities[stage_idx]
            data = modalities[modality]
            
            if isinstance(data, np.ndarray):
                data = self._normalize_feature(data)
            
            stage_name = self._get_stage_name(stage_idx)
            
            if current_result is None:
                current_result = data
            else:
                current_result = self._blend_features(current_result, data, weight)
            
            stage_outputs.append({
                "stage": stage_name,
                "modality": modality,
                "weight": weight,
                "feature_shape": current_result.shape
            })
        
        self.stage_results.append({
            "timestamp": np.datetime64('now'),
            "stages": stage_outputs,
            "final_shape": current_result.shape
        })
        
        return {
            "fused_feature": current_result,
            "progressive_order": [m[0] for m in sorted_modalities[:stages]],
            "stages": stage_outputs,
            "num_stages": stages,
            "context": context
        }
    
    def _get_stage_name(self, stage_idx: int) -> str:
        stage_names = [
            FusionStage.FEATURE_LEVEL.value,
            FusionStage.ATTENTION_LEVEL.value,
            FusionStage.SEMANTIC_LEVEL.value,
            FusionStage.DECISION_LEVEL.value
        ]
        
        return stage_names[stage_idx] if stage_idx < len(stage_names) else f"stage_{stage_idx}"
    
    def get_intermediate_results(self) -> List[Dict[str, Any]]:
        return self.stage_results
    
    def visualize_progressive_fusion(self) -> Dict[str, Any]:
        if not self.stage_results:
            return {"error": "No fusion history"}
        
        latest = self.stage_results[-1]
        
        visualization_data = {
            "stages": [],
            "modality_contribution": {}
        }
        
        for stage in latest["stages"]:
            visualization_data["stages"].append({
                "name": stage["stage"],
                "modality": stage["modality"],
                "weight": stage["weight"]
            })
            
            visualization_data["modality_contribution"][stage["modality"]] = stage["weight"]
        
        return visualization_data


class AdvancedMultimodalFusionEngine:
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self.cross_attention = AdvancedCrossModalAttention(feature_dim)
        self.quality_assessor = AdvancedModalityQualityAssessor()
        self.fusion_strategy = ContextAwareFusionStrategy()
        self.progressive_fusion = ProgressiveMultimodalFusion(feature_dim)
        self.fusion_history: List[Dict[str, Any]] = []
    
    def fuse(self, modalities: Dict[str, Any],
            task_type: str = "general",
            context: Optional[Dict[str, Any]] = None,
            use_progressive: bool = False) -> Dict[str, Any]:
        quality_scores = {}
        for modality, data in modalities.items():
            quality = self.quality_assessor.assess_modality(modality, data)
            anomaly = self.quality_assessor.detect_anomaly(modality, data)
            quality_scores[modality] = {
                "quality": quality.value,
                "score": self.quality_assessor.get_quality_score(quality),
                "anomaly": anomaly
            }
        
        valid_modalities = {k: v for k, v in modalities.items() 
                           if quality_scores[k]["score"] > 0.25}
        
        if not valid_modalities:
            return {
                "error": "No valid modalities available",
                "quality_scores": quality_scores
            }
        
        attention_enhanced = self.cross_attention.joint_attention(valid_modalities)
        
        weights = self.fusion_strategy.compute_context_aware_weights(valid_modalities, context)
        strategy = self.fusion_strategy.select_strategy(valid_modalities, task_type, context)
        
        if use_progressive:
            result = self.progressive_fusion.progressive_fuse(attention_enhanced, 
                                                             stages=4, 
                                                             context=context)
        else:
            fusion_input = {}
            for modality in ['text', 'image', 'audio', 'video', 'sensor']:
                if modality in attention_enhanced:
                    fusion_input[modality] = attention_enhanced[modality]
            
            base_result = self.progressive_fusion.fuse(fusion_input, task_type, context)
            result = {
                "fused_feature": base_result.get("fused_feature"),
                "strategy": strategy,
                "weights": weights
            }
        
        result.update({
            "quality_scores": quality_scores,
            "attention_summary": self.cross_attention.get_attention_summary(),
            "context": context,
            "task_type": task_type,
            "timestamp": np.datetime64('now')
        })
        
        self.fusion_history.append(result)
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "total_fusions": len(self.fusion_history),
            "quality_assessment": self.quality_assessor.get_anomaly_summary(),
            "attention_stats": self.cross_attention.get_attention_summary(),
            "strategy_performance": dict(self.fusion_strategy.task_performance),
            "progressive_stages": self.progressive_fusion.get_intermediate_results()[-5:] if self.progressive_fusion.stage_results else []
        }
    
    def update_strategy_performance(self, task_type: str, strategy: str, score: float):
        self.fusion_strategy.update_performance(task_type, strategy, score)


def get_advanced_multimodal_fusion(feature_dim: int = 256) -> AdvancedMultimodalFusionEngine:
    return AdvancedMultimodalFusionEngine(feature_dim=feature_dim)
