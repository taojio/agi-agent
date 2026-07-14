import time
import hashlib
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class DataQualityLevel(Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    CONTAMINATED = "contaminated"


class DataSource(Enum):
    ENDOGENOUS = "endogenous"
    INTERACTIVE = "interactive"
    EXTERNAL = "external"
    SIMULATED = "simulated"


@dataclass
class TrainingData:
    data_id: str
    content: Any
    source: DataSource
    quality_level: DataQualityLevel = DataQualityLevel.BRONZE
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    features: Optional[np.ndarray] = None
    labels: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    used_count: int = 0
    last_used: Optional[float] = None
    quality_checks: Dict[str, bool] = field(default_factory=dict)

    def calculate_hash(self) -> str:
        content_str = str(self.content) if not isinstance(self.content, bytes) else self.content.hex()
        return hashlib.md5(content_str.encode()).hexdigest()


@dataclass
class DataQualityMetrics:
    completeness: float = 1.0
    consistency: float = 1.0
    diversity: float = 0.0
    freshness: float = 1.0
    overall_score: float = 0.0

    def compute_overall(self) -> float:
        self.overall_score = (
            0.3 * self.completeness +
            0.3 * self.consistency +
            0.2 * self.diversity +
            0.2 * self.freshness
        )
        return self.overall_score


class DataPipeline:
    def __init__(self, max_buffer_size: int = 10000):
        self.data_buffer: deque = deque(maxlen=max_buffer_size)
        self.quality_history: deque = deque(maxlen=1000)
        self.processed_count = 0
        self.rejected_count = 0
        self.duplicate_count = 0
        self._seen_hashes: set = set()
        self._quality_thresholds = {
            DataQualityLevel.GOLD: 0.8,
            DataQualityLevel.SILVER: 0.6,
            DataQualityLevel.BRONZE: 0.3,
            DataQualityLevel.CONTAMINATED: 0.0
        }
        self.data_ratios = {
            "basic": 0.4,
            "domain": 0.3,
            "exploration": 0.2,
            "adversarial": 0.1
        }

    def add_data(self, data: TrainingData) -> bool:
        data_hash = data.calculate_hash()
        if data_hash in self._seen_hashes:
            self.duplicate_count += 1
            return False
        self._seen_hashes.add(data_hash)

        quality_metrics = self._assess_quality(data)
        data.quality_level = self._classify_quality(quality_metrics.overall_score)
        data.metadata["quality_metrics"] = {
            "completeness": quality_metrics.completeness,
            "consistency": quality_metrics.consistency,
            "diversity": quality_metrics.diversity,
            "freshness": quality_metrics.freshness,
            "overall": quality_metrics.overall_score
        }

        if data.quality_level == DataQualityLevel.CONTAMINATED:
            self.rejected_count += 1
            return False

        self.data_buffer.append(data)
        self.processed_count += 1
        self.quality_history.append({
            "timestamp": time.time(),
            "level": data.quality_level.value,
            "score": quality_metrics.overall_score
        })
        return True

    def _assess_quality(self, data: TrainingData) -> DataQualityMetrics:
        metrics = DataQualityMetrics()

        metrics.completeness = self._check_completeness(data)
        metrics.consistency = self._check_consistency(data)
        metrics.diversity = self._check_diversity(data)
        metrics.freshness = self._check_freshness(data)
        metrics.compute_overall()

        return metrics

    def _check_completeness(self, data: TrainingData) -> float:
        if data.content is None:
            return 0.0
        if isinstance(data.content, (list, np.ndarray)):
            arr = np.array(data.content)
            missing_ratio = np.sum(arr == 0) / max(arr.size, 1) if arr.size > 0 else 0
            return max(0.0, 1.0 - missing_ratio)
        if isinstance(data.content, dict):
            non_empty = sum(1 for v in data.content.values() if v is not None and v != "")
            total = max(len(data.content), 1)
            return non_empty / total
        return data.confidence

    def _check_consistency(self, data: TrainingData) -> float:
        if data.features is not None:
            feat = data.features
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                return 0.3
            variance = np.var(feat) if feat.size > 1 else 0
            if variance < 1e-8:
                return 0.5
        return min(1.0, max(0.3, data.confidence))

    def _check_diversity(self, data: TrainingData) -> float:
        if len(self.data_buffer) < 10:
            return 0.8
        recent_features = []
        for d in list(self.data_buffer)[-50:]:
            if d.features is not None:
                recent_features.append(d.features.flatten()[:16])
        if not recent_features or data.features is None:
            return 0.5
        try:
            data_feat = data.features.flatten()[:16]
            similarities = []
            for rf in recent_features:
                min_len = min(len(data_feat), len(rf))
                if min_len == 0:
                    similarities.append(0.0)
                    continue
                a = data_feat[:min_len]
                b = rf[:min_len]
                dot = np.dot(a, b)
                norm_a = np.linalg.norm(a) + 1e-8
                norm_b = np.linalg.norm(b) + 1e-8
                cos_sim = dot / (norm_a * norm_b)
                similarities.append(abs(cos_sim))
            avg_sim = np.mean(similarities) if similarities else 0.5
            return max(0.0, 1.0 - avg_sim)
        except Exception:
            return 0.5

    def _check_freshness(self, data: TrainingData) -> float:
        age_hours = (time.time() - data.timestamp) / 3600
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.9
        elif age_hours < 168:
            return 0.7
        elif age_hours < 720:
            return 0.5
        else:
            return 0.3

    def _classify_quality(self, score: float) -> DataQualityLevel:
        if score >= self._quality_thresholds[DataQualityLevel.GOLD]:
            return DataQualityLevel.GOLD
        elif score >= self._quality_thresholds[DataQualityLevel.SILVER]:
            return DataQualityLevel.SILVER
        elif score >= self._quality_thresholds[DataQualityLevel.BRONZE]:
            return DataQualityLevel.BRONZE
        else:
            return DataQualityLevel.CONTAMINATED

    def preprocess_perception_data(self, raw_data: np.ndarray) -> np.ndarray:
        data = raw_data.copy()

        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        if np.any(np.isinf(data)):
            data = np.clip(data, -1e6, 1e6)

        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-8:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        noise = np.random.normal(0, 0.01, data.shape) * 0.1
        data = data + noise
        data = np.clip(data, 0.0, 1.0)

        return data

    def augment_data(self, data: TrainingData, augment_type: str = "gaussian_noise") -> Optional[TrainingData]:
        if data.features is None:
            return None

        augmented_features = data.features.copy()

        if augment_type == "gaussian_noise":
            noise = np.random.normal(0, 0.05, augmented_features.shape)
            augmented_features = augmented_features + noise
        elif augment_type == "masking":
            mask = np.random.binomial(1, 0.9, augmented_features.shape)
            augmented_features = augmented_features * mask
        elif augment_type == "scaling":
            scale = np.random.uniform(0.9, 1.1)
            augmented_features = augmented_features * scale

        return TrainingData(
            data_id=f"{data.data_id}_aug_{int(time.time()*1000)}",
            content=data.content,
            source=data.source,
            quality_level=DataQualityLevel.BRONZE,
            confidence=data.confidence * 0.8,
            features=augmented_features,
            labels=data.labels,
            metadata={**data.metadata, "augmented": True, "augment_type": augment_type}
        )

    def get_batch(self, batch_size: int, quality_min: DataQualityLevel = DataQualityLevel.BRONZE) -> List[TrainingData]:
        min_score = self._quality_thresholds[quality_min]
        eligible = [d for d in self.data_buffer
                     if d.metadata.get("quality_metrics", {}).get("overall", 0) >= min_score]

        if len(eligible) < batch_size:
            eligible = list(self.data_buffer)

        if not eligible:
            return []

        indices = np.random.choice(len(eligible), size=min(batch_size, len(eligible)), replace=False)
        batch = [eligible[i] for i in indices]

        for data in batch:
            data.used_count += 1
            data.last_used = time.time()

        return batch

    def get_quality_stats(self) -> Dict[str, Any]:
        level_counts = {level.value: 0 for level in DataQualityLevel}
        for data in self.data_buffer:
            level_counts[data.quality_level.value] += 1

        recent_scores = [q["score"] for q in self.quality_history]
        avg_quality = np.mean(recent_scores) if recent_scores else 0.0

        return {
            "total_processed": self.processed_count,
            "total_rejected": self.rejected_count,
            "duplicate_count": self.duplicate_count,
            "buffer_size": len(self.data_buffer),
            "quality_distribution": level_counts,
            "average_quality_score": avg_quality,
            "acceptance_rate": self.processed_count / max(1, self.processed_count + self.rejected_count)
        }

    def set_data_ratios(self, ratios: Dict[str, float]):
        total = sum(ratios.values())
        if abs(total - 1.0) > 0.01:
            ratios = {k: v / total for k, v in ratios.items()}
        self.data_ratios = ratios

    def cleanup_old_data(self, max_age_days: int = 30):
        max_age_seconds = max_age_days * 86400
        current_time = time.time()
        removed = 0
        new_buffer = deque(maxlen=self.data_buffer.maxlen)
        for data in self.data_buffer:
            if current_time - data.timestamp <= max_age_seconds:
                new_buffer.append(data)
            else:
                removed += 1
                data_hash = data.calculate_hash()
                if data_hash in self._seen_hashes:
                    self._seen_hashes.discard(data_hash)
        self.data_buffer = new_buffer
        return removed
