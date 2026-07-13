import numpy as np
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .meta_parser import MetaParser, ParsingStrategy, ParsingResult
from .data_transformer import DataTransformer, TransformationType, TransformationResult
from .complex_data_processor import ComplexDataProcessor, DataUnderstanding


class ParsingOrchestrator:
    def __init__(self):
        self.parser = MetaParser()
        self.transformer = DataTransformer()
        self.complex_processor = ComplexDataProcessor()
        self._processing_pipelines: Dict[str, List[str]] = {}
        self._orchestration_history: deque = deque(maxlen=200)

    def parse_and_understand(self, data: str, format_hint: str = "",
                            strategy: ParsingStrategy = ParsingStrategy.AUTO) -> Dict[str, Any]:
        parsing_result = self.parser.parse(data, format_hint, strategy)
        
        if not parsing_result.success:
            return {
                "success": False,
                "stage": "parsing",
                "error": parsing_result.errors,
                "parsing_result": parsing_result.to_dict()
            }
        
        understanding = self.complex_processor.process(parsing_result.parsed_data)
        
        return {
            "success": True,
            "stage": "understanding",
            "parsing_result": parsing_result.to_dict(),
            "understanding": understanding.to_dict()
        }

    def parse_transform_and_understand(self, data: str, format_hint: str = "",
                                       strategy: ParsingStrategy = ParsingStrategy.AUTO,
                                       transformation_rules: List[str] = None) -> Dict[str, Any]:
        parsing_result = self.parser.parse(data, format_hint, strategy)
        
        if not parsing_result.success:
            return {
                "success": False,
                "stage": "parsing",
                "error": parsing_result.errors,
                "parsing_result": parsing_result.to_dict()
            }
        
        if transformation_rules:
            transform_result = self.transformer.transform(
                parsing_result.parsed_data, transformation_rules
            )
        else:
            transform_result = self.transformer.auto_transform(parsing_result.parsed_data)
        
        understanding = self.complex_processor.process(transform_result.transformed_data)
        
        self._orchestration_history.append({
            "timestamp": np.random.randint(1000000),
            "format": format_hint,
            "parsing_success": parsing_result.success,
            "transformation_success": transform_result.success,
            "understanding_level": understanding.understanding_level,
            "data_quality": understanding.data_quality
        })
        
        return {
            "success": True,
            "stage": "complete",
            "parsing_result": parsing_result.to_dict(),
            "transformation_result": transform_result.to_dict(),
            "understanding": understanding.to_dict()
        }

    def create_pipeline(self, pipeline_id: str, steps: List[Dict[str, Any]]):
        self._processing_pipelines[pipeline_id] = steps

    def execute_pipeline(self, pipeline_id: str, data: str) -> Dict[str, Any]:
        if pipeline_id not in self._processing_pipelines:
            return {"success": False, "error": f"Pipeline {pipeline_id} not found"}
        
        steps = self._processing_pipelines[pipeline_id]
        results = []
        current_data = data
        
        for i, step in enumerate(steps):
            step_type = step.get("type")
            
            if step_type == "parse":
                result = self.parser.parse(
                    current_data,
                    step.get("format_hint", ""),
                    step.get("strategy", ParsingStrategy.AUTO)
                )
                current_data = result.parsed_data if result.success else current_data
                results.append({"step": i, "type": "parse", "result": result.to_dict()})
                
                if not result.success:
                    return {"success": False, "stage": f"step_{i}", "results": results}
            
            elif step_type == "transform":
                rules = step.get("rules", [])
                chain_id = step.get("chain_id")
                
                if isinstance(current_data, str):
                    parse_result = self.parser.parse(current_data)
                    if parse_result.success:
                        current_data = parse_result.parsed_data
                
                result = self.transformer.transform(current_data, rules, chain_id)
                current_data = result.transformed_data if result.success else current_data
                results.append({"step": i, "type": "transform", "result": result.to_dict()})
            
            elif step_type == "process":
                understanding = self.complex_processor.process(current_data)
                results.append({"step": i, "type": "process", "result": understanding.to_dict()})
        
        return {"success": True, "results": results, "final_data": current_data}

    def auto_process(self, data: str) -> Dict[str, Any]:
        understanding = self.complex_processor.process({"raw": data})
        
        if understanding.complexity.complexity.value in ["complex", "very_complex"]:
            return self.parse_transform_and_understand(data, "", ParsingStrategy.RECURSIVE)
        
        return self.parse_and_understand(data)

    def batch_process(self, data_list: List[str], format_hint: str = "") -> List[Dict[str, Any]]:
        results = []
        for data in data_list:
            result = self.parse_and_understand(data, format_hint)
            results.append(result)
        return results

    def get_overview(self) -> Dict[str, Any]:
        return {
            "parser": self.parser.get_parsing_summary(),
            "transformer": self.transformer.get_transformation_summary(),
            "complex_processor": self.complex_processor.get_processing_summary(),
            "pipelines": list(self._processing_pipelines.keys()),
            "total_orchestrations": len(self._orchestration_history)
        }

    def register_custom_parser(self, name: str, parser: Callable,
                              supported_formats: List[str], priority: int = 10):
        self.parser.register_custom_parser(name, parser, supported_formats, priority)

    def register_transformation_rule(self, rule_id: str, transformation_type: TransformationType,
                                     condition: str = None, action: Callable = None,
                                     params: Dict[str, Any] = None):
        self.transformer.create_rule(rule_id, transformation_type, condition, action, params)

    def get_recent_processing(self, limit: int = 10) -> List[Dict[str, Any]]:
        recent = list(self._orchestration_history)[-limit:]
        return recent[::-1]