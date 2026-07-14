import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import DEVICE


class ContextType(Enum):
    PHYSICAL = "physical"
    TEMPORAL = "temporal"
    SOCIAL = "social"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    TASK = "task"
    ENVIRONMENTAL = "environmental"
    EXPLORATION = "exploration"


class SceneType(Enum):
    UNKNOWN = "unknown"
    DIALOG = "dialog"
    WORK = "work"
    LEARNING = "learning"
    EXPLORATION = "exploration"
    REST = "rest"
    CRITICAL = "critical"
    CREATIVE = "creative"
    DECISION_MAKING = "decision_making"
    PROBLEM_SOLVING = "problem_solving"


class ContextFrame:
    def __init__(self, context_id: str, context_type: ContextType,
                 features: Optional[np.ndarray] = None,
                 timestamp: float = None, metadata: Optional[Dict] = None):
        self.context_id = context_id
        self.context_type = context_type
        self.features = features if features is not None else np.array([])
        self.timestamp = timestamp if timestamp is not None else np.random.randint(1000000)
        self.metadata = metadata if metadata is not None else {}
        self.confidence = 0.5
        self.duration = 0.0

    def to_dict(self):
        return {
            "context_id": self.context_id,
            "context_type": self.context_type.value,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "duration": self.duration,
            "metadata": self.metadata
        }


class Situation:
    def __init__(self, situation_id: str, description: str, scene_type: SceneType,
                 context_frames: Optional[List[ContextFrame]] = None,
                 confidence: float = 0.5):
        self.situation_id = situation_id
        self.description = description
        self.scene_type = scene_type
        self.context_frames = context_frames if context_frames is not None else []
        self.confidence = confidence
        self.start_time = np.random.randint(1000000)
        self.end_time = None
        self.key_entities: List[str] = []
        self.goals: List[str] = []
        self.ongoing = True

    def add_context_frame(self, frame: ContextFrame):
        self.context_frames.append(frame)

    def end(self):
        self.ongoing = False
        self.end_time = np.random.randint(1000000)
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time

    def to_dict(self):
        return {
            "situation_id": self.situation_id,
            "description": self.description,
            "scene_type": self.scene_type.value,
            "confidence": self.confidence,
            "ongoing": self.ongoing,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": getattr(self, 'duration', 0),
            "key_entities": self.key_entities,
            "goals": self.goals,
            "context_count": len(self.context_frames)
        }


class ContextAwarenessEngine:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.context_history = deque(maxlen=500)
        self.current_situation: Optional[Situation] = None
        self.situation_history = deque(maxlen=200)
        self.scene_patterns: Dict[str, List[Dict]] = {}
        self.context_similarity_threshold = 0.6
        self.scene_confidence_threshold = 0.7

        self._init_scene_patterns()

    def _init_scene_patterns(self):
        self.scene_patterns = {
            SceneType.DIALOG.value: [
                {"context_type": ContextType.SOCIAL.value, "confidence_threshold": 0.7},
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.5},
                {"context_type": ContextType.EMOTIONAL.value, "confidence_threshold": 0.4}
            ],
            SceneType.WORK.value: [
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.8},
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.6},
                {"context_type": ContextType.TEMPORAL.value, "confidence_threshold": 0.5}
            ],
            SceneType.LEARNING.value: [
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.8},
                {"context_type": ContextType.ENVIRONMENTAL.value, "confidence_threshold": 0.5},
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.4}
            ],
            SceneType.EXPLORATION.value: [
                {"context_type": ContextType.PHYSICAL.value, "confidence_threshold": 0.6},
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.5},
                {"context_type": ContextType.ENVIRONMENTAL.value, "confidence_threshold": 0.7}
            ],
            SceneType.REST.value: [
                {"context_type": ContextType.EMOTIONAL.value, "confidence_threshold": 0.6},
                {"context_type": ContextType.PHYSICAL.value, "confidence_threshold": 0.5},
                {"context_type": ContextType.TEMPORAL.value, "confidence_threshold": 0.4}
            ],
            SceneType.CRITICAL.value: [
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.9},
                {"context_type": ContextType.EMOTIONAL.value, "confidence_threshold": 0.7},
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.8}
            ],
            SceneType.CREATIVE.value: [
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.7},
                {"context_type": ContextType.EMOTIONAL.value, "confidence_threshold": 0.6},
                {"context_type": ContextType.ENVIRONMENTAL.value, "confidence_threshold": 0.5}
            ],
            SceneType.DECISION_MAKING.value: [
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.8},
                {"context_type": ContextType.SOCIAL.value, "confidence_threshold": 0.5},
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.7}
            ],
            SceneType.PROBLEM_SOLVING.value: [
                {"context_type": ContextType.COGNITIVE.value, "confidence_threshold": 0.8},
                {"context_type": ContextType.TASK.value, "confidence_threshold": 0.7},
                {"context_type": ContextType.ENVIRONMENTAL.value, "confidence_threshold": 0.5}
            ]
        }

    def add_context_frame(self, context_type: ContextType, features: np.ndarray = None,
                          metadata: Optional[Dict] = None) -> str:
        context_id = f"ctx_{len(self.context_history) + 1}"
        frame = ContextFrame(context_id, context_type, features, metadata=metadata)
        self.context_history.append(frame)

        self._update_current_situation(frame)

        return context_id

    def _update_current_situation(self, frame: ContextFrame):
        if self.current_situation is None:
            self._start_new_situation(frame)
        else:
            self.current_situation.add_context_frame(frame)
            self._reassess_situation()

    def _start_new_situation(self, frame: ContextFrame):
        scene_type = self._detect_scene_type([frame])
        situation_id = f"situation_{len(self.situation_history) + 1}"
        description = self._generate_situation_description(scene_type)

        self.current_situation = Situation(situation_id, description, scene_type)
        self.current_situation.add_context_frame(frame)

    def _reassess_situation(self):
        if self.current_situation is None:
            return

        frames = self.current_situation.context_frames[-10:]
        new_scene_type = self._detect_scene_type(frames)

        if new_scene_type != self.current_situation.scene_type:
            confidence_diff = abs(new_scene_type != self.current_situation.scene_type)
            if confidence_diff > 0.3:
                self._transition_situation(new_scene_type)

    def _detect_scene_type(self, frames: List[ContextFrame]) -> SceneType:
        if not frames:
            return SceneType.UNKNOWN

        scene_scores = {}
        for scene_type in SceneType:
            if scene_type == SceneType.UNKNOWN:
                continue

            pattern = self.scene_patterns.get(scene_type.value, [])
            if not pattern:
                continue

            score = 0.0
            matches = 0

            for frame in frames:
                for required_context in pattern:
                    if frame.context_type.value == required_context["context_type"]:
                        score += frame.confidence * required_context["confidence_threshold"]
                        matches += 1

            if matches > 0:
                scene_scores[scene_type] = score / len(pattern)

        if not scene_scores:
            return SceneType.UNKNOWN

        best_scene = max(scene_scores, key=scene_scores.get)
        if scene_scores[best_scene] >= self.scene_confidence_threshold:
            return best_scene

        return SceneType.UNKNOWN

    def _generate_situation_description(self, scene_type: SceneType) -> str:
        descriptions = {
            SceneType.DIALOG: "对话交流场景",
            SceneType.WORK: "工作任务场景",
            SceneType.LEARNING: "学习探索场景",
            SceneType.EXPLORATION: "环境探索场景",
            SceneType.REST: "休息放松场景",
            SceneType.CRITICAL: "紧急关键场景",
            SceneType.CREATIVE: "创意生成场景",
            SceneType.DECISION_MAKING: "决策制定场景",
            SceneType.PROBLEM_SOLVING: "问题解决场景",
            SceneType.UNKNOWN: "未知场景"
        }
        return descriptions.get(scene_type, "未知场景")

    def _transition_situation(self, new_scene_type: SceneType):
        if self.current_situation is None:
            return

        self.current_situation.end()
        self.situation_history.append(self.current_situation)

        situation_id = f"situation_{len(self.situation_history) + 1}"
        description = self._generate_situation_description(new_scene_type)

        self.current_situation = Situation(situation_id, description, new_scene_type)

    def get_current_context(self) -> Dict[str, Any]:
        context_summary = {
            "current_situation": None,
            "recent_contexts": [],
            "context_distribution": {},
            "overall_confidence": 0.0
        }

        if self.current_situation:
            context_summary["current_situation"] = self.current_situation.to_dict()

        recent_frames = list(self.context_history)[-20:]
        context_summary["recent_contexts"] = [f.to_dict() for f in recent_frames]

        for frame in recent_frames:
            ct = frame.context_type.value
            context_summary["context_distribution"][ct] = context_summary["context_distribution"].get(ct, 0) + 1

        if recent_frames:
            context_summary["overall_confidence"] = float(np.mean([f.confidence for f in recent_frames]))

        return context_summary

    def find_contextual_pattern(self, target_context: ContextType,
                                time_window: int = 50) -> List[Dict]:
        recent_frames = list(self.context_history)[-time_window:]
        matches = []

        for i, frame in enumerate(recent_frames):
            if frame.context_type == target_context:
                context_before = recent_frames[max(0, i - 3):i]
                context_after = recent_frames[i + 1:min(i + 4, len(recent_frames))]

                matches.append({
                    "timestamp": frame.timestamp,
                    "confidence": frame.confidence,
                    "context_before": [f.context_type.value for f in context_before],
                    "context_after": [f.context_type.value for f in context_after],
                    "metadata": frame.metadata
                })

        return matches

    def predict_next_context(self) -> Dict[str, Any]:
        if len(self.context_history) < 5:
            return {"predicted_type": None, "confidence": 0.0, "reasoning": "Insufficient data"}

        recent_sequence = list(self.context_history)[-10:]
        context_sequence = [f.context_type.value for f in recent_sequence]

        transition_counts = {}
        for i in range(len(context_sequence) - 1):
            current = context_sequence[i]
            next_ctx = context_sequence[i + 1]
            key = (current, next_ctx)
            transition_counts[key] = transition_counts.get(key, 0) + 1

        if not transition_counts:
            return {"predicted_type": None, "confidence": 0.0, "reasoning": "No transitions found"}

        last_context = context_sequence[-1]
        possible_next = [(k[1], v) for k, v in transition_counts.items() if k[0] == last_context]

        if not possible_next:
            return {"predicted_type": None, "confidence": 0.0, "reasoning": "No transition from last context"}

        possible_next.sort(key=lambda x: -x[1])
        total_transitions = sum(v for _, v in possible_next)
        predicted_type = possible_next[0][0]
        confidence = possible_next[0][1] / total_transitions

        return {
            "predicted_type": predicted_type,
            "confidence": confidence,
            "reasoning": f"Transition pattern from {last_context}",
            "alternatives": [{"type": t, "confidence": c / total_transitions} for t, c in possible_next[1:]]
        }

    def get_contextual_influence(self, entity_id: str) -> Dict[str, Any]:
        recent_frames = list(self.context_history)[-50:]
        entity_mentions = []

        for frame in recent_frames:
            if "entities" in frame.metadata:
                if entity_id in frame.metadata["entities"]:
                    entity_mentions.append(frame)

        if not entity_mentions:
            return {"influence": 0.0, "contexts": [], "confidence": 0.0}

        context_types = {}
        for frame in entity_mentions:
            ct = frame.context_type.value
            context_types[ct] = context_types.get(ct, 0) + frame.confidence

        total_confidence = sum(context_types.values())
        influence = total_confidence / len(recent_frames)

        return {
            "influence": float(influence),
            "contexts": context_types,
            "confidence": float(total_confidence / len(entity_mentions)) if entity_mentions else 0.0,
            "mention_count": len(entity_mentions)
        }

    def get_situation_stats(self) -> Dict[str, Any]:
        stats = {
            "total_situations": len(self.situation_history),
            "current_situation": None,
            "scene_distribution": {},
            "avg_situation_duration": 0.0,
            "avg_confidence": 0.0,
            "active_context_count": len(self.context_history)
        }

        if self.current_situation:
            stats["current_situation"] = self.current_situation.to_dict()

        for situation in self.situation_history:
            st = situation.scene_type.value
            stats["scene_distribution"][st] = stats["scene_distribution"].get(st, 0) + 1

        completed_situations = [s for s in self.situation_history if not s.ongoing]
        if completed_situations:
            durations = []
            for s in completed_situations:
                if hasattr(s, 'duration') and s.duration:
                    durations.append(s.duration)
            stats["avg_situation_duration"] = float(np.mean(durations)) if durations else 0.0
            stats["avg_confidence"] = float(np.mean([s.confidence for s in completed_situations]))

        return stats

    def detect_scene(self) -> SceneType:
        if self.current_situation is not None:
            return self.current_situation.scene_type
        
        if self.context_history:
            return self._detect_scene_type([self.context_history[-1]])
        
        return SceneType.UNKNOWN

    def resize(self, new_dim):
        self.feature_dim = new_dim