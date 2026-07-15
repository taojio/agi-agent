import numpy as np
from collections import deque
from .spiking_core import SpikingCore
from ..utils.numpy_utils import cosine_similarity


class PatternMatcher:
    def __init__(self, feature_dim=16, similarity_threshold=0.7):
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.patterns = {}
        self.pattern_features = {}
        self.pattern_predictions = {}
        self.pattern_confidence = {}
        self.match_history = deque(maxlen=100)
        self.snn = SpikingCore(input_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)

    def add_pattern(self, pattern_id, feature_vector, prediction=None, confidence=0.9):
        self.patterns[pattern_id] = {
            "feature": np.array(feature_vector).flatten(),
            "prediction": prediction,
            "confidence": confidence,
            "match_count": 0,
            "success_count": 0
        }
        self.pattern_features[pattern_id] = np.array(feature_vector).flatten()

    def match(self, input_vector):
        input_vec = np.array(input_vector).flatten()
        
        snn_output = self.snn.forward(input_vec)
        anomaly_score, _ = self.snn.detect_anomaly(input_vec)
        
        best_match = None
        best_similarity = 0.0
        best_prediction = None
        
        for pattern_id, pattern_data in self.patterns.items():
            similarity = cosine_similarity(input_vec, pattern_data["feature"])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern_id
                best_prediction = pattern_data["prediction"]
        
        is_known = best_similarity >= self.similarity_threshold
        is_anomalous = anomaly_score > 2.0
        
        if best_match and is_known:
            self.patterns[best_match]["match_count"] += 1
        
        self.match_history.append({
            "pattern_id": best_match,
            "similarity": best_similarity,
            "is_known": is_known,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous
        })
        
        return {
            "pattern_id": best_match,
            "similarity": best_similarity,
            "confidence": self.patterns[best_match]["confidence"] if best_match else 0.0,
            "prediction": best_prediction,
            "is_known": is_known,
            "is_anomalous": is_anomalous,
            "anomaly_score": anomaly_score
        }

    def batch_match(self, input_vectors):
        results = []
        for vec in input_vectors:
            results.append(self.match(vec))
        return results

    def update_pattern_confidence(self, pattern_id, success):
        if pattern_id in self.patterns:
            self.patterns[pattern_id]["match_count"] += 1
            if success:
                self.patterns[pattern_id]["success_count"] += 1
            total = self.patterns[pattern_id]["match_count"]
            self.patterns[pattern_id]["confidence"] = self.patterns[pattern_id]["success_count"] / total

    def prune_low_confidence_patterns(self, min_confidence=0.3, min_matches=5):
        pruned = []
        for pattern_id in list(self.patterns.keys()):
            pattern = self.patterns[pattern_id]
            if pattern["match_count"] >= min_matches and pattern["confidence"] < min_confidence:
                pruned.append(pattern_id)
                del self.patterns[pattern_id]
                del self.pattern_features[pattern_id]
        return pruned

    def get_pattern_stats(self):
        stats = {}
        for pattern_id, pattern in self.patterns.items():
            stats[pattern_id] = {
                "match_count": pattern["match_count"],
                "success_count": pattern["success_count"],
                "confidence": pattern["confidence"]
            }
        return stats

    def learn_from_history(self):
        if len(self.match_history) < 10:
            return
        
        recent_matches = list(self.match_history)[-10:]
        known_matches = [m for m in recent_matches if m["is_known"] and m["pattern_id"]]
        
        if known_matches:
            avg_similarity = np.mean([m["similarity"] for m in known_matches])
            self.similarity_threshold = max(0.5, min(0.9, avg_similarity * 0.9))

    def get_snn_activity(self):
        return self.snn.get_activity_level()