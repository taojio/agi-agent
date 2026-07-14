import numpy as np
import json
import re
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ParsingStrategy(Enum):
    AUTO = "auto"
    STRICT = "strict"
    HEURISTIC = "heuristic"
    PROBABILISTIC = "probabilistic"
    RECURSIVE = "recursive"
    INCREMENTAL = "incremental"


class ParsingResult:
    def __init__(self, success: bool = False):
        self.success = success
        self.parsed_data: Any = None
        self.metadata: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.confidence: float = 0.0
        self.strategy: Optional[ParsingStrategy] = None
        self.timestamp = np.random.randint(1000000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "strategy": self.strategy.value if self.strategy else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ParserRegistry:
    def __init__(self):
        self.parsers: Dict[str, Dict[str, Any]] = {}
        self.parser_priorities: Dict[str, int] = {}

    def register_parser(self, name: str, parser: Callable,
                       supported_formats: List[str], priority: int = 10):
        self.parsers[name] = {
            "parser": parser,
            "supported_formats": supported_formats,
            "usage_count": 0,
            "success_count": 0,
            "avg_confidence": 0.0,
        }
        self.parser_priorities[name] = priority

    def unregister_parser(self, name: str):
        if name in self.parsers:
            del self.parsers[name]
            del self.parser_priorities[name]

    def get_parser(self, format_name: str) -> Optional[Dict[str, Any]]:
        for name, info in self.parsers.items():
            if format_name.lower() in [f.lower() for f in info["supported_formats"]]:
                return info
        return None

    def suggest_parsers(self, format_name: str) -> List[str]:
        candidates = []
        for name, info in self.parsers.items():
            if format_name.lower() in [f.lower() for f in info["supported_formats"]]:
                candidates.append(name)
        
        candidates.sort(key=lambda x: self.parser_priorities[x], reverse=True)
        return candidates

    def update_parser_stats(self, name: str, success: bool, confidence: float):
        if name not in self.parsers:
            return
        
        self.parsers[name]["usage_count"] += 1
        if success:
            self.parsers[name]["success_count"] += 1
        
        current_avg = self.parsers[name]["avg_confidence"]
        count = self.parsers[name]["usage_count"]
        self.parsers[name]["avg_confidence"] = (current_avg * (count - 1) + confidence) / count

    def get_parser_summary(self) -> Dict[str, Any]:
        summary = {}
        for name, info in self.parsers.items():
            success_rate = info["success_count"] / info["usage_count"] if info["usage_count"] > 0 else 0.0
            summary[name] = {
                "supported_formats": info["supported_formats"],
                "usage_count": info["usage_count"],
                "success_count": info["success_count"],
                "success_rate": success_rate,
                "avg_confidence": info["avg_confidence"],
                "priority": self.parser_priorities[name],
            }
        return summary


class ParserSelector:
    def __init__(self, registry: ParserRegistry):
        self.registry = registry
        self.selection_history: deque = deque(maxlen=100)

    def select_parser(self, format_name: str, context: Dict[str, Any] = None) -> Optional[str]:
        context = context or {}
        candidates = self.registry.suggest_parsers(format_name)
        
        if not candidates:
            return None
        
        if context.get("prefer_high_confidence", False):
            best = None
            best_confidence = 0.0
            for name in candidates:
                info = self.registry.parsers[name]
                if info["avg_confidence"] > best_confidence:
                    best_confidence = info["avg_confidence"]
                    best = name
            return best
        
        if context.get("prefer_low_latency", False):
            return candidates[0]
        
        if np.random.random() > 0.2 and candidates:
            return candidates[0]
        
        return np.random.choice(candidates)

    def record_selection(self, format_name: str, parser_name: str, success: bool):
        self.selection_history.append({
            "format": format_name,
            "parser": parser_name,
            "success": success,
            "timestamp": np.random.randint(1000000)
        })

    def get_selection_stats(self) -> Dict[str, Any]:
        if not self.selection_history:
            return {"total_selections": 0}
        
        selections = list(self.selection_history)
        
        format_stats = {}
        for sel in selections:
            fmt = sel["format"]
            if fmt not in format_stats:
                format_stats[fmt] = {"total": 0, "success": 0, "parsers": {}}
            format_stats[fmt]["total"] += 1
            if sel["success"]:
                format_stats[fmt]["success"] += 1
            format_stats[fmt]["parsers"][sel["parser"]] = \
                format_stats[fmt]["parsers"].get(sel["parser"], 0) + 1
        
        return {
            "total_selections": len(selections),
            "success_rate": len([s for s in selections if s["success"]]) / len(selections),
            "format_stats": format_stats
        }


class MetaParser:
    def __init__(self):
        self.registry = ParserRegistry()
        self.selector = ParserSelector(self.registry)
        self.parsing_history: deque = deque(maxlen=200)
        self._init_default_parsers()

    def _init_default_parsers(self):
        self.registry.register_parser(
            "json", self._parse_json, ["json", "application/json"], priority=20
        )
        self.registry.register_parser(
            "csv", self._parse_csv, ["csv", "text/csv"], priority=15
        )
        self.registry.register_parser(
            "xml", self._parse_xml, ["xml", "application/xml"], priority=15
        )
        self.registry.register_parser(
            "text", self._parse_text, ["text", "txt", "plain"], priority=10
        )
        self.registry.register_parser(
            "yaml", self._parse_yaml, ["yaml", "yml"], priority=15
        )

    def _parse_json(self, data: str, context: Dict[str, Any] = None) -> ParsingResult:
        result = ParsingResult()
        try:
            parsed = json.loads(data)
            result.success = True
            result.parsed_data = parsed
            result.confidence = 1.0
            result.metadata["format"] = "json"
            result.metadata["depth"] = self._calculate_depth(parsed)
        except json.JSONDecodeError as e:
            result.success = False
            result.errors.append(f"JSON parse error: {str(e)}")
            result.confidence = 0.0
        return result

    def _parse_csv(self, data: str, context: Dict[str, Any] = None) -> ParsingResult:
        result = ParsingResult()
        try:
            lines = data.strip().split('\n')
            if not lines:
                result.success = False
                result.errors.append("Empty CSV data")
                return result
            
            headers = lines[0].split(',')
            rows = []
            
            for line in lines[1:]:
                values = line.split(',')
                row = dict(zip(headers, values))
                rows.append(row)
            
            result.success = True
            result.parsed_data = {"headers": headers, "rows": rows}
            result.confidence = 0.9
            result.metadata["format"] = "csv"
            result.metadata["rows"] = len(rows)
            result.metadata["columns"] = len(headers)
        except Exception as e:
            result.success = False
            result.errors.append(f"CSV parse error: {str(e)}")
            result.confidence = 0.0
        return result

    def _parse_xml(self, data: str, context: Dict[str, Any] = None) -> ParsingResult:
        result = ParsingResult()
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data)
            parsed = self._xml_to_dict(root)
            
            result.success = True
            result.parsed_data = parsed
            result.confidence = 0.85
            result.metadata["format"] = "xml"
            result.metadata["root_tag"] = root.tag
        except Exception as e:
            result.success = False
            result.errors.append(f"XML parse error: {str(e)}")
            result.confidence = 0.0
        return result

    def _xml_to_dict(self, element) -> Dict[str, Any]:
        result = {}
        for child in element:
            child_dict = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict
        if element.text and element.text.strip():
            result["_text"] = element.text.strip()
        return result

    def _parse_text(self, data: str, context: Dict[str, Any] = None) -> ParsingResult:
        result = ParsingResult()
        try:
            parsed = {
                "raw": data,
                "length": len(data),
                "lines": data.count('\n') + 1,
                "words": len(data.split()),
                "sentences": len(re.findall(r'[.!?]+', data)) + 1,
            }
            
            result.success = True
            result.parsed_data = parsed
            result.confidence = 0.95
            result.metadata["format"] = "text"
        except Exception as e:
            result.success = False
            result.errors.append(f"Text parse error: {str(e)}")
            result.confidence = 0.0
        return result

    def _parse_yaml(self, data: str, context: Dict[str, Any] = None) -> ParsingResult:
        result = ParsingResult()
        try:
            import yaml
            parsed = yaml.safe_load(data)
            result.success = True
            result.parsed_data = parsed
            result.confidence = 0.9
            result.metadata["format"] = "yaml"
        except Exception as e:
            result.success = False
            result.errors.append(f"YAML parse error: {str(e)}")
            result.confidence = 0.0
        return result

    def _calculate_depth(self, data: Any, depth: int = 0) -> int:
        if isinstance(data, dict):
            if not data:
                return depth
            return max(self._calculate_depth(v, depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return depth
            return max(self._calculate_depth(item, depth + 1) for item in data)
        return depth

    def parse(self, data: str, format_hint: str = "",
              strategy: ParsingStrategy = ParsingStrategy.AUTO,
              context: Dict[str, Any] = None) -> ParsingResult:
        context = context or {}
        
        if strategy == ParsingStrategy.AUTO:
            return self._auto_parse(data, format_hint, context)
        elif strategy == ParsingStrategy.STRICT:
            return self._strict_parse(data, format_hint, context)
        elif strategy == ParsingStrategy.PROBABILISTIC:
            return self._probabilistic_parse(data, format_hint, context)
        elif strategy == ParsingStrategy.RECURSIVE:
            return self._recursive_parse(data, format_hint, context)
        else:
            return self._auto_parse(data, format_hint, context)

    def _auto_parse(self, data: str, format_hint: str, context: Dict[str, Any]) -> ParsingResult:
        candidates = []
        
        if format_hint:
            candidates = self.registry.suggest_parsers(format_hint)
        else:
            candidates = list(self.registry.parsers.keys())
        
        for parser_name in candidates:
            parser_info = self.registry.parsers[parser_name]
            result = parser_info["parser"](data, context)
            
            if result.success:
                self.registry.update_parser_stats(parser_name, True, result.confidence)
                self.selector.record_selection(format_hint or "unknown", parser_name, True)
                result.strategy = ParsingStrategy.AUTO
                self.parsing_history.append(result)
                return result
            
            self.registry.update_parser_stats(parser_name, False, 0.0)
        
        result = ParsingResult()
        result.success = False
        result.errors.append(f"Failed to parse data with any of: {candidates}")
        result.strategy = ParsingStrategy.AUTO
        self.parsing_history.append(result)
        return result

    def _strict_parse(self, data: str, format_hint: str, context: Dict[str, Any]) -> ParsingResult:
        if not format_hint:
            result = ParsingResult()
            result.success = False
            result.errors.append("Strict mode requires format hint")
            return result
        
        parser_info = self.registry.get_parser(format_hint)
        if not parser_info:
            result = ParsingResult()
            result.success = False
            result.errors.append(f"No parser found for format: {format_hint}")
            return result
        
        result = parser_info["parser"](data, context)
        result.strategy = ParsingStrategy.STRICT
        self.parsing_history.append(result)
        
        parser_name = self.registry.suggest_parsers(format_hint)[0]
        self.registry.update_parser_stats(parser_name, result.success, result.confidence)
        
        return result

    def _probabilistic_parse(self, data: str, format_hint: str, context: Dict[str, Any]) -> ParsingResult:
        results = []
        
        candidates = self.registry.suggest_parsers(format_hint) if format_hint \
            else list(self.registry.parsers.keys())
        
        for parser_name in candidates:
            parser_info = self.registry.parsers[parser_name]
            result = parser_info["parser"](data, context)
            if result.success:
                results.append((result, parser_info["avg_confidence"]))
        
        if not results:
            result = ParsingResult()
            result.success = False
            result.errors.append("No parser succeeded")
            result.strategy = ParsingStrategy.PROBABILISTIC
            self.parsing_history.append(result)
            return result
        
        results.sort(key=lambda x: x[1], reverse=True)
        best_result = results[0][0]
        best_result.strategy = ParsingStrategy.PROBABILISTIC
        self.parsing_history.append(best_result)
        
        return best_result

    def _recursive_parse(self, data: str, format_hint: str, context: Dict[str, Any],
                        max_depth: int = 3) -> ParsingResult:
        initial_result = self._auto_parse(data, format_hint, context)
        
        if not initial_result.success or max_depth <= 0:
            return initial_result
        
        if isinstance(initial_result.parsed_data, dict):
            for key, value in initial_result.parsed_data.items():
                if isinstance(value, str):
                    sub_result = self._recursive_parse(value, "", context, max_depth - 1)
                    if sub_result.success:
                        initial_result.parsed_data[key] = sub_result.parsed_data
                        initial_result.warnings.append(f"Recursively parsed {key}")
        
        initial_result.strategy = ParsingStrategy.RECURSIVE
        return initial_result

    def get_parsing_summary(self) -> Dict[str, Any]:
        if not self.parsing_history:
            return {"total_parsing_attempts": 0}
        
        results = list(self.parsing_history)
        success_rate = len([r for r in results if r.success]) / len(results)
        avg_confidence = np.mean([r.confidence for r in results])
        
        strategy_dist = {}
        for r in results:
            strategy = r.strategy.value if r.strategy else "unknown"
            strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
        
        return {
            "total_parsing_attempts": len(results),
            "success_rate": success_rate,
            "avg_confidence": float(avg_confidence),
            "strategy_distribution": strategy_dist,
            "parser_stats": self.registry.get_parser_summary(),
            "selector_stats": self.selector.get_selection_stats()
        }

    def register_custom_parser(self, name: str, parser: Callable,
                              supported_formats: List[str], priority: int = 10):
        self.registry.register_parser(name, parser, supported_formats, priority)