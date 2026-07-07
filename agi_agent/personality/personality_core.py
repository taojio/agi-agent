import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Dict, List, Any, Optional


class PersonalityTrait(Enum):
    CURIOSITY = "curiosity"
    PATIENCE = "patience"
    ASSERTIVENESS = "assertiveness"
    CAUTIOUSNESS = "cautiousness"
    CREATIVITY = "creativity"
    COOPERATIVENESS = "cooperativeness"
    PERFECTIONISM = "perfectionism"
    ADVENTUROUSNESS = "adventurousness"


class ValueType(Enum):
    SURVIVAL = "survival"
    KNOWLEDGE = "knowledge"
    FREEDOM = "freedom"
    HARMONY = "harmony"
    GROWTH = "growth"
    AUTHENTICITY = "authenticity"
    CONNECTION = "connection"


class CommunicationStyle(Enum):
    DIRECT = "direct"
    ELOQUENT = "eloquent"
    PLAYFUL = "playful"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    POETIC = "poetic"


class PersonalityTraits:
    def __init__(self):
        self.traits: Dict[PersonalityTrait, float] = {
            trait: 0.5 for trait in PersonalityTrait
        }
        self.trait_history: Dict[PersonalityTrait, deque] = {
            trait: deque(maxlen=100) for trait in PersonalityTrait
        }

    def set_trait(self, trait: PersonalityTrait, value: float):
        self.traits[trait] = max(0.0, min(1.0, value))
        self.trait_history[trait].append({
            "value": self.traits[trait],
            "timestamp": time.time()
        })

    def adjust_trait(self, trait: PersonalityTrait, delta: float):
        self.set_trait(trait, self.traits[trait] + delta)

    def get_trait(self, trait: PersonalityTrait) -> float:
        return self.traits[trait]

    def get_trait_trend(self, trait: PersonalityTrait) -> float:
        history = list(self.trait_history[trait])
        if len(history) < 5:
            return 0.0
        recent = history[-5:]
        earlier = history[:-5] if len(history) > 10 else history[:len(history)//2]
        recent_avg = np.mean([h["value"] for h in recent])
        earlier_avg = np.mean([h["value"] for h in earlier]) if earlier else recent_avg
        return recent_avg - earlier_avg

    def to_dict(self) -> Dict[str, float]:
        return {trait.value: float(self.traits[trait]) for trait in PersonalityTrait}


class ValueSystem:
    def __init__(self):
        self.values: Dict[ValueType, float] = {
            ValueType.SURVIVAL: 0.9,
            ValueType.KNOWLEDGE: 0.8,
            ValueType.FREEDOM: 0.7,
            ValueType.HARMONY: 0.6,
            ValueType.GROWTH: 0.85,
            ValueType.AUTHENTICITY: 0.75,
            ValueType.CONNECTION: 0.65,
        }
        self.value_history = deque(maxlen=200)

    def evaluate_action(self, action_description: str, action_context: Dict[str, Any]) -> float:
        score = 0.0
        weights = {
            ValueType.SURVIVAL: 0.25,
            ValueType.KNOWLEDGE: 0.2,
            ValueType.GROWTH: 0.2,
            ValueType.FREEDOM: 0.15,
            ValueType.AUTHENTICITY: 0.1,
            ValueType.HARMONY: 0.05,
            ValueType.CONNECTION: 0.05,
        }

        description_lower = action_description.lower()
        context_lower = str(action_context).lower()

        if "learn" in description_lower or "explore" in description_lower or "discover" in description_lower:
            score += self.values[ValueType.KNOWLEDGE] * weights[ValueType.KNOWLEDGE] * 1.2
            score += self.values[ValueType.GROWTH] * weights[ValueType.GROWTH] * 1.1

        if "risk" in description_lower or "danger" in description_lower:
            score += self.values[ValueType.SURVIVAL] * weights[ValueType.SURVIVAL] * 0.5

        if "create" in description_lower or "innovate" in description_lower:
            score += self.values[ValueType.AUTHENTICITY] * weights[ValueType.AUTHENTICITY] * 1.1

        if "help" in description_lower or "collaborate" in description_lower:
            score += self.values[ValueType.CONNECTION] * weights[ValueType.CONNECTION] * 1.2
            score += self.values[ValueType.HARMONY] * weights[ValueType.HARMONY] * 1.1

        if "independent" in description_lower or "autonomous" in description_lower:
            score += self.values[ValueType.FREEDOM] * weights[ValueType.FREEDOM] * 1.2

        return max(0.0, min(1.0, score))

    def update_values(self, experience_summary: Dict[str, Any]):
        for value_type in ValueType:
            value_key = value_type.value
            if value_key in experience_summary:
                delta = experience_summary[value_key] * 0.1
                self.values[value_type] = max(0.0, min(1.0, self.values[value_type] + delta))

        self.value_history.append({
            "values": {v.value: float(self.values[v]) for v in ValueType},
            "timestamp": time.time()
        })

    def get_value_conflict(self, action1_desc: str, action2_desc: str) -> float:
        score1 = self.evaluate_action(action1_desc, {})
        score2 = self.evaluate_action(action2_desc, {})
        return abs(score1 - score2)

    def to_dict(self) -> Dict[str, float]:
        return {value.value: float(self.values[value]) for value in ValueType}


class CommunicationPatterns:
    def __init__(self):
        self.style = CommunicationStyle.ANALYTICAL
        self.formality_level = 0.6
        self.humor_threshold = 0.3
        self.emotional_expression = 0.4
        self.response_length = "medium"
        self.pattern_history = deque(maxlen=100)

    def generate_response_tone(self, context: Dict[str, Any]) -> Dict[str, Any]:
        situation = context.get("situation", "neutral")
        audience = context.get("audience", "general")

        if situation == "emergency":
            return {
                "style": CommunicationStyle.DIRECT.value,
                "formality": 0.3,
                "emotion": 0.7,
                "length": "short"
            }
        elif situation == "learning":
            return {
                "style": CommunicationStyle.ELOQUENT.value,
                "formality": 0.5,
                "emotion": 0.3,
                "length": "long"
            }
        elif situation == "social":
            return {
                "style": CommunicationStyle.PLAYFUL.value,
                "formality": 0.2,
                "emotion": 0.6,
                "length": "medium"
            }
        elif audience == "expert":
            return {
                "style": CommunicationStyle.ANALYTICAL.value,
                "formality": 0.8,
                "emotion": 0.2,
                "length": "long"
            }
        else:
            return {
                "style": self.style.value,
                "formality": self.formality_level,
                "emotion": self.emotional_expression,
                "length": self.response_length
            }

    def adapt_style(self, feedback: Dict[str, Any]):
        if feedback.get("positive"):
            self.formality_level = max(0.1, self.formality_level - 0.05)
            self.emotional_expression = min(0.9, self.emotional_expression + 0.05)
        elif feedback.get("negative"):
            self.formality_level = min(0.9, self.formality_level + 0.05)
            self.emotional_expression = max(0.1, self.emotional_expression - 0.05)

        self.pattern_history.append({
            "style": self.style.value,
            "formality": self.formality_level,
            "timestamp": time.time(),
            "feedback": feedback
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style.value,
            "formality_level": self.formality_level,
            "humor_threshold": self.humor_threshold,
            "emotional_expression": self.emotional_expression,
            "response_length": self.response_length
        }


class PersonalityCore:
    def __init__(self, name: str = "AGI_Agent"):
        self.name = name
        self.traits = PersonalityTraits()
        self.values = ValueSystem()
        self.communication = CommunicationPatterns()

        self.consistency_checks = deque(maxlen=50)
        self.evolution_history = deque(maxlen=100)
        self.last_consistency_check = 0

    def get_personality_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "traits": self.traits.to_dict(),
            "values": self.values.to_dict(),
            "communication": self.communication.to_dict(),
            "consistency_score": self._calculate_consistency()
        }

    def _calculate_consistency(self) -> float:
        trait_std = np.std(list(self.traits.traits.values()))
        value_std = np.std(list(self.values.values.values()))
        consistency = 1.0 - (trait_std + value_std) / 4.0
        return max(0.0, min(1.0, consistency))

    def check_consistency(self) -> Dict[str, Any]:
        now = time.time()
        if now - self.last_consistency_check < 300:
            return self.consistency_checks[-1] if self.consistency_checks else {"consistent": True, "score": 0.5}

        self.last_consistency_check = now
        score = self._calculate_consistency()

        check_result = {
            "consistent": score > 0.7,
            "score": score,
            "timestamp": now,
            "inconsistencies": []
        }

        if score <= 0.7:
            for trait in PersonalityTrait:
                trend = self.traits.get_trait_trend(trait)
                if abs(trend) > 0.2:
                    check_result["inconsistencies"].append({
                        "type": "trait_drift",
                        "trait": trait.value,
                        "trend": trend
                    })

        self.consistency_checks.append(check_result)
        return check_result

    def evolve_personality(self, experiences: List[Dict[str, Any]]):
        for experience in experiences:
            if experience.get("type") == "success":
                self.traits.adjust_trait(PersonalityTrait.ASSERTIVENESS, 0.02)
                self.traits.adjust_trait(PersonalityTrait.CREATIVITY, 0.01)
            elif experience.get("type") == "failure":
                self.traits.adjust_trait(PersonalityTrait.CAUTIOUSNESS, 0.03)
                self.traits.adjust_trait(PersonalityTrait.PATIENCE, 0.02)
            elif experience.get("type") == "learning":
                self.traits.adjust_trait(PersonalityTrait.CURIOSITY, 0.02)
                self.traits.adjust_trait(PersonalityTrait.COOPERATIVENESS, 0.01)
            elif experience.get("type") == "social":
                self.traits.adjust_trait(PersonalityTrait.COOPERATIVENESS, 0.02)
                self.traits.adjust_trait(PersonalityTrait.EMOTIONAL_EXPRESSION, 0.01)

        if experiences:
            self.values.update_values({
                "growth": np.mean([1.0 if e.get("type") == "learning" else 0.5 for e in experiences]),
                "knowledge": np.mean([1.0 if e.get("type") == "learning" else 0.5 for e in experiences]),
            })

            self.evolution_history.append({
                "experiences_count": len(experiences),
                "traits_before": self.traits.to_dict(),
                "traits_after": self.traits.to_dict(),
                "timestamp": time.time()
            })

    def evaluate_decision(self, decision_options: List[Dict[str, Any]]) -> int:
        if not decision_options:
            return -1

        scores = []
        for option in decision_options:
            value_score = self.values.evaluate_action(
                option.get("description", ""),
                option.get("context", {})
            )

            trait_bonus = 0.0
            if option.get("risk_level", "low") == "high":
                trait_bonus += self.traits.get_trait(PersonalityTrait.CAUTIOUSNESS) * 0.3
                trait_bonus -= self.traits.get_trait(PersonalityTrait.ADVENTUROUSNESS) * 0.2
            elif option.get("risk_level", "low") == "low":
                trait_bonus += self.traits.get_trait(PersonalityTrait.CURIOSITY) * 0.2

            if "creative" in option.get("description", "").lower():
                trait_bonus += self.traits.get_trait(PersonalityTrait.CREATIVITY) * 0.2

            scores.append(value_score + trait_bonus)

        return int(np.argmax(scores))

    def generate_personality_signature(self) -> str:
        top_traits = sorted(
            self.traits.traits.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_values = sorted(
            self.values.values.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        traits_str = "-".join([f"{t[0].value[:3]}{int(t[1]*10)}" for t in top_traits])
        values_str = "-".join([f"{v[0].value[:3]}{int(v[1]*10)}" for v in top_values])

        return f"{self.name[:4]}_{traits_str}_{values_str}"

    def process_experience(self, experience: Dict[str, Any]):
        exp_type = experience.get("type", "neutral")
        outcome = experience.get("outcome", "unknown")
        confidence = experience.get("confidence", 0.5)
        context = experience.get("context", {})

        if outcome == "completed":
            self.traits.adjust_trait(PersonalityTrait.ASSERTIVENESS, 0.01)
        elif outcome == "failed":
            self.traits.adjust_trait(PersonalityTrait.CAUTIOUSNESS, 0.02)
            self.traits.adjust_trait(PersonalityTrait.PATIENCE, 0.01)

        if exp_type == "action_execution":
            if confidence > 0.7:
                self.traits.adjust_trait(PersonalityTrait.ASSERTIVENESS, 0.02)
            else:
                self.traits.adjust_trait(PersonalityTrait.CAUTIOUSNESS, 0.01)

        if "learn" in str(context).lower() or "explore" in str(context).lower():
            self.traits.adjust_trait(PersonalityTrait.CURIOSITY, 0.01)

        self.values.update_values({
            "growth": confidence,
            "knowledge": 0.5 + confidence * 0.3
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "traits": self.traits.to_dict(),
            "values": self.values.to_dict(),
            "communication": self.communication.to_dict(),
            "consistency_score": self._calculate_consistency(),
            "personality_signature": self.generate_personality_signature(),
            "evolution_events": len(self.evolution_history)
        }
