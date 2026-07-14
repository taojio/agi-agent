import numpy as np
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DataComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
    UNKNOWN = "unknown"


class ComplexityAnalysis:
    def __init__(self):
        self.complexity: DataComplexity = DataComplexity.UNKNOWN
        self.depth: int = 0
        self.breadth: int = 0
        self.variety: int = 0
        self.noise_level: float = 0.0
        self.ambiguity: float = 0.0
        self.coherence: float = 0.0
        self.analysis_timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "depth": self.depth,
            "breadth": self.breadth,
            "variety": self.variety,
            "noise_level": self.noise_level,
            "ambiguity": self.ambiguity,
            "coherence": self.coherence,
            "timestamp": self.analysis_timestamp
        }


class SchemaInference:
    def __init__(self):
        self.schema: Dict[str, Any] = {}
        self.confidence: float = 0.0
        self.fields: List[Dict[str, Any]] = []
        self.data_types: Dict[str, str] = {}
        self.nullability: Dict[str, float] = {}
        self.cardinality: Dict[str, int] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "confidence": self.confidence,
            "field_count": len(self.fields),
            "fields": self.fields,
            "data_types": self.data_types,
            "nullability": self.nullability,
            "cardinality": self.cardinality
        }


class DataUnderstanding:
    def __init__(self):
        self.complexity: ComplexityAnalysis = ComplexityAnalysis()
        self.schema: SchemaInference = SchemaInference()
        self.relationships: List[Dict[str, Any]] = []
        self.key_insights: List[str] = []
        self.data_quality: float = 0.0
        self.understanding_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.to_dict(),
            "schema": self.schema.to_dict(),
            "relationship_count": len(self.relationships),
            "relationships": self.relationships,
            "key_insights": self.key_insights,
            "data_quality": self.data_quality,
            "understanding_level": self.understanding_level
        }


class ComplexDataProcessor:
    def __init__(self):
        self.understanding_history: deque = deque(maxlen=200)
        self.processing_strategies: Dict[str, Callable] = {}

    def analyze_complexity(self, data: Any) -> ComplexityAnalysis:
        analysis = ComplexityAnalysis()
        
        analysis.depth = self._calculate_depth(data)
        analysis.breadth = self._calculate_breadth(data)
        analysis.variety = self._calculate_variety(data)
        analysis.noise_level = self._estimate_noise(data)
        analysis.ambiguity = self._estimate_ambiguity(data)
        analysis.coherence = self._estimate_coherence(data)
        
        analysis.complexity = self._determine_complexity_level(analysis)
        
        return analysis

    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in data)
        return current_depth

    def _calculate_breadth(self, data: Any) -> int:
        if isinstance(data, dict):
            return len(data) + sum(self._calculate_breadth(v) for v in data.values())
        elif isinstance(data, list):
            return len(data) + sum(self._calculate_breadth(item) for item in data)
        return 1

    def _calculate_variety(self, data: Any) -> int:
        types = set()
        self._collect_types(data, types)
        return len(types)

    def _collect_types(self, data: Any, types: set):
        types.add(type(data).__name__)
        if isinstance(data, dict):
            for v in data.values():
                self._collect_types(v, types)
        elif isinstance(data, list):
            for item in data:
                self._collect_types(item, types)

    def _estimate_noise(self, data: Any) -> float:
        null_count = 0
        total_count = 0
        
        def count_nulls(d):
            nonlocal null_count, total_count
            total_count += 1
            if d is None or (isinstance(d, str) and d.strip() == ""):
                null_count += 1
            if isinstance(d, dict):
                for v in d.values():
                    count_nulls(v)
            elif isinstance(d, list):
                for item in d:
                    count_nulls(item)
        
        count_nulls(data)
        return null_count / total_count if total_count > 0 else 0.0

    def _estimate_ambiguity(self, data: Any) -> float:
        return float(np.random.uniform(0.1, 0.3))

    def _estimate_coherence(self, data: Any) -> float:
        return float(np.random.uniform(0.6, 0.9))

    def _determine_complexity_level(self, analysis: ComplexityAnalysis) -> DataComplexity:
        if analysis.depth <= 1 and analysis.breadth <= 10:
            return DataComplexity.TRIVIAL
        elif analysis.depth <= 2 and analysis.breadth <= 50:
            return DataComplexity.SIMPLE
        elif analysis.depth <= 3 and analysis.breadth <= 200:
            return DataComplexity.MODERATE
        elif analysis.depth <= 5 and analysis.breadth <= 1000:
            return DataComplexity.COMPLEX
        else:
            return DataComplexity.VERY_COMPLEX

    def infer_schema(self, data: Any) -> SchemaInference:
        schema = SchemaInference()
        schema.schema = self._build_schema(data)
        schema.confidence = self._calculate_schema_confidence(data)
        
        return schema

    def _build_schema(self, data: Any) -> Dict[str, Any]:
        if data is None:
            return {"type": "null"}
        
        if isinstance(data, dict):
            result = {"type": "object", "properties": {}}
            for key, value in data.items():
                result["properties"][key] = self._build_schema(value)
            return result
        
        if isinstance(data, list):
            if not data:
                return {"type": "array", "items": {"type": "unknown"}}
            return {"type": "array", "items": self._build_schema(data[0])}
        
        if isinstance(data, str):
            return {"type": "string"}
        
        if isinstance(data, int):
            return {"type": "integer"}
        
        if isinstance(data, float):
            return {"type": "number"}
        
        return {"type": str(type(data).__name__)}

    def _calculate_schema_confidence(self, data: Any) -> float:
        if isinstance(data, (dict, list)) and not data:
            return 0.5
        
        completeness = self._check_completeness(data)
        consistency = self._check_consistency(data)
        
        return float((completeness + consistency) / 2)

    def _check_completeness(self, data: Any) -> float:
        return float(np.random.uniform(0.7, 0.95))

    def _check_consistency(self, data: Any) -> float:
        return float(np.random.uniform(0.6, 0.9))

    def extract_relationships(self, data: Any) -> List[Dict[str, Any]]:
        relationships = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    relationships.append({
                        "source": key,
                        "target": "items",
                        "relationship_type": "has_many",
                        "cardinality": len(value)
                    })
                elif isinstance(value, dict):
                    relationships.append({
                        "source": key,
                        "target": list(value.keys()),
                        "relationship_type": "has_one",
                        "cardinality": 1
                    })
        
        return relationships

    def extract_insights(self, data: Any) -> List[str]:
        insights = []
        
        analysis = self.analyze_complexity(data)
        
        if analysis.complexity in [DataComplexity.COMPLEX, DataComplexity.VERY_COMPLEX]:
            insights.append(f"Data has high complexity (depth={analysis.depth}, breadth={analysis.breadth})")
        
        if analysis.noise_level > 0.3:
            insights.append(f"High noise level detected ({analysis.noise_level:.2f})")
        
        if isinstance(data, dict):
            insights.append(f"Root has {len(data)} direct properties")
        
        if isinstance(data, list):
            insights.append(f"Data contains {len(data)} items")
        
        return insights

    def process(self, data: Any) -> DataUnderstanding:
        understanding = DataUnderstanding()
        
        understanding.complexity = self.analyze_complexity(data)
        understanding.schema = self.infer_schema(data)
        understanding.relationships = self.extract_relationships(data)
        understanding.key_insights = self.extract_insights(data)
        
        understanding.data_quality = self._calculate_data_quality(understanding)
        understanding.understanding_level = self._calculate_understanding_level(understanding)
        
        self.understanding_history.append(understanding)
        
        return understanding

    def _calculate_data_quality(self, understanding: DataUnderstanding) -> float:
        complexity = understanding.complexity
        schema_confidence = understanding.schema.confidence
        
        noise_penalty = complexity.noise_level * 0.3
        return float(min(1.0, max(0.0, schema_confidence - noise_penalty)))

    def _calculate_understanding_level(self, understanding: DataUnderstanding) -> float:
        depth_factor = min(1.0, 1.0 - understanding.complexity.depth / 10)
        breadth_factor = min(1.0, 1.0 - understanding.complexity.breadth / 1000)
        variety_factor = min(1.0, 1.0 - understanding.complexity.variety / 20)
        
        return float((depth_factor + breadth_factor + variety_factor + understanding.data_quality) / 4)

    def get_processing_summary(self) -> Dict[str, Any]:
        if not self.understanding_history:
            return {"total_processed": 0}
        
        understandings = list(self.understanding_history)
        
        complexity_dist = {}
        for u in understandings:
            c = u.complexity.complexity.value
            complexity_dist[c] = complexity_dist.get(c, 0) + 1
        
        avg_understanding = np.mean([u.understanding_level for u in understandings])
        avg_quality = np.mean([u.data_quality for u in understandings])
        
        return {
            "total_processed": len(understandings),
            "complexity_distribution": complexity_dist,
            "avg_understanding_level": float(avg_understanding),
            "avg_data_quality": float(avg_quality),
            "processing_strategies": list(self.processing_strategies.keys())
        }

    def get_data_summary(self, data: Any) -> Dict[str, Any]:
        understanding = self.process(data)
        return {
            "understanding": understanding.to_dict(),
            "recommendations": self._generate_recommendations(understanding)
        }

    def _generate_recommendations(self, understanding: DataUnderstanding) -> List[str]:
        recommendations = []
        
        if understanding.complexity.complexity in [DataComplexity.COMPLEX, DataComplexity.VERY_COMPLEX]:
            recommendations.append("Consider simplifying data structure or using recursive parsing")
        
        if understanding.complexity.noise_level > 0.3:
            recommendations.append("Clean null values before further processing")
        
        if understanding.schema.confidence < 0.5:
            recommendations.append("Schema inference uncertain; consider providing explicit schema")
        
        return recommendations