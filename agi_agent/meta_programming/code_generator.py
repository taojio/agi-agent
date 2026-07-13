import ast
import inspect
import json
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class GenerationMode(Enum):
    SYNTHESIS = "synthesis"
    TRANSFORMATION = "transformation"
    OPTIMIZATION = "optimization"
    REPAIR = "repair"


class CodeQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PRODUCTION = "production"


class CodeTemplate:
    def __init__(self, name: str, template: str, params: List[str], language: str = "python"):
        self.name = name
        self.template = template
        self.params = params
        self.language = language
        self.usage_count = 0

    def render(self, **kwargs) -> str:
        missing = [p for p in self.params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        self.usage_count += 1
        return self.template.format(**kwargs)

    def get_signature(self) -> str:
        return f"{self.name}({', '.join(self.params)})"


class GenerationContext:
    def __init__(self):
        self.imports: List[str] = []
        self.constraints: Dict[str, Any] = {}
        self.references: List[Dict[str, Any]] = []
        self.target_language: str = "python"
        self.quality_requirement: CodeQuality = CodeQuality.HIGH
        self.max_complexity: int = 10
        self.existing_code: str = ""

    def add_import(self, module: str, alias: str = None):
        if alias:
            self.imports.append(f"import {module} as {alias}")
        else:
            self.imports.append(f"import {module}")

    def add_reference(self, name: str, code: str, description: str = ""):
        self.references.append({
            "name": name,
            "code": code,
            "description": description
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "imports": self.imports,
            "constraints": self.constraints,
            "references": self.references,
            "target_language": self.target_language,
            "quality_requirement": self.quality_requirement.value,
            "max_complexity": self.max_complexity
        }


class CodeGenerationResult:
    def __init__(self, code: str, mode: GenerationMode, 
                 quality: CodeQuality, confidence: float):
        self.code = code
        self.mode = mode
        self.quality = quality
        self.confidence = confidence
        self.issues: List[str] = []
        self.estimated_complexity: float = 0.0
        self.execution_time_ms: float = 0.0

    def add_issue(self, issue: str):
        self.issues.append(issue)

    def is_valid(self) -> bool:
        return len(self.issues) == 0 and self.confidence > 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "mode": self.mode.value,
            "quality": self.quality.value,
            "confidence": self.confidence,
            "issues": self.issues,
            "estimated_complexity": self.estimated_complexity,
            "execution_time_ms": self.execution_time_ms
        }


class CodeGenerator:
    def __init__(self):
        self.templates: Dict[str, CodeTemplate] = {}
        self.template_categories: Dict[str, List[str]] = {}
        self.generation_history: List[CodeGenerationResult] = []
        self._register_default_templates()

    def _register_default_templates(self):
        self.register_template(CodeTemplate(
            name="function_def",
            template="def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
            params=["name", "params", "docstring", "body"]
        ), "functions")

        self.register_template(CodeTemplate(
            name="class_def",
            template="class {name}({base_classes}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
            params=["name", "base_classes", "docstring", "body"]
        ), "classes")

        self.register_template(CodeTemplate(
            name="try_except",
            template="try:\n    {try_block}\nexcept {exception_type} as e:\n    {except_block}\n{finally_block}",
            params=["try_block", "exception_type", "except_block", "finally_block"]
        ), "error_handling")

        self.register_template(CodeTemplate(
            name="data_class",
            template="from dataclasses import dataclass\n\n@dataclass\nclass {name}:\n{fields}",
            params=["name", "fields"]
        ), "data_structures")

    def register_template(self, template: CodeTemplate, category: str = "general"):
        self.templates[template.name] = template
        if category not in self.template_categories:
            self.template_categories[category] = []
        self.template_categories[category].append(template.name)

    def generate_code(self, context: GenerationContext, 
                      mode: GenerationMode = GenerationMode.SYNTHESIS,
                      target: str = "") -> CodeGenerationResult:
        result = CodeGenerationResult(
            code="",
            mode=mode,
            quality=context.quality_requirement,
            confidence=0.0
        )

        if mode == GenerationMode.SYNTHESIS:
            result.code = self._synthesize_code(context, target)
        elif mode == GenerationMode.TRANSFORMATION:
            result.code = self._transform_code(context)
        elif mode == GenerationMode.OPTIMIZATION:
            result.code = self._optimize_code(context)
        elif mode == GenerationMode.REPAIR:
            result.code = self._repair_code(context)

        result.confidence = self._estimate_confidence(result.code, context)
        result.estimated_complexity = self._estimate_complexity(result.code)

        if not self._validate_syntax(result.code):
            result.add_issue("Syntax validation failed")

        self.generation_history.append(result)
        return result

    def _synthesize_code(self, context: GenerationContext, target: str) -> str:
        parts = []
        
        if context.imports:
            parts.extend(context.imports)
            parts.append("")

        if target.startswith("function"):
            template = self.templates.get("function_def")
            if template:
                return template.render(
                    name="generated_function",
                    params="x",
                    docstring="Automatically generated function",
                    body="    return x * 2"
                )

        if target.startswith("class"):
            template = self.templates.get("class_def")
            if template:
                return template.render(
                    name="GeneratedClass",
                    base_classes="",
                    docstring="Automatically generated class",
                    body="    def __init__(self):\n        pass"
                )

        return "\n".join(parts) if parts else "# Generated code placeholder"

    def _transform_code(self, context: GenerationContext) -> str:
        if not context.existing_code:
            return ""
        
        code = context.existing_code
        
        if context.constraints.get("add_docstrings"):
            code = self._add_docstrings(code)
        
        if context.constraints.get("simplify"):
            code = self._simplify_code(code)
        
        return code

    def _optimize_code(self, context: GenerationContext) -> str:
        if not context.existing_code:
            return ""
        
        code = context.existing_code
        
        code = self._inline_simple_functions(code)
        code = self._remove_redundant_code(code)
        
        return code

    def _repair_code(self, context: GenerationContext) -> str:
        if not context.existing_code:
            return ""
        
        code = context.existing_code
        tree = None
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._fix_syntax_errors(code)
        
        return code

    def _add_docstrings(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        if isinstance(node, ast.FunctionDef):
                            args = [arg.arg for arg in node.args.args]
                            docstring = f'"""Auto-generated docstring for {node.name}({", ".join(args)})"""'
                        else:
                            docstring = f'"""Auto-generated docstring for {node.name}"""'
                        
                        if isinstance(node, ast.ClassDef):
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        else:
                            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
            
            return ast.unparse(tree)
        except Exception:
            return code

    def _simplify_code(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            simplified = ast.fix_missing_locations(ast.Constant(value="# Simplified code placeholder"))
            return ast.unparse(simplified)
        except Exception:
            return code

    def _inline_simple_functions(self, code: str) -> str:
        return code

    def _remove_redundant_code(self, code: str) -> str:
        return code

    def _fix_syntax_errors(self, code: str) -> str:
        return "# Syntax-fixed placeholder code\npass"

    def _validate_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _estimate_confidence(self, code: str, context: GenerationContext) -> float:
        if not code:
            return 0.0
        
        confidence = 0.5
        
        if self._validate_syntax(code):
            confidence += 0.2
        
        if len(code.strip()) > 20:
            confidence += 0.1
        
        if context.quality_requirement == CodeQuality.PRODUCTION:
            confidence = min(0.95, confidence * 1.1)
        
        return min(1.0, confidence)

    def _estimate_complexity(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return float(len(list(ast.walk(tree))))
        except Exception:
            return 0.0

    def generate_function(self, name: str, params: List[str], 
                         body: str, docstring: str = "") -> CodeGenerationResult:
        template = self.templates.get("function_def")
        if template:
            code = template.render(
                name=name,
                params=", ".join(params),
                docstring=docstring,
                body=body
            )
            return CodeGenerationResult(
                code=code,
                mode=GenerationMode.SYNTHESIS,
                quality=CodeQuality.HIGH,
                confidence=0.9
            )
        return CodeGenerationResult(code="", mode=GenerationMode.SYNTHESIS, 
                                    quality=CodeQuality.LOW, confidence=0.0)

    def generate_class(self, name: str, base_classes: List[str],
                       fields: Dict[str, str], methods: Dict[str, str]) -> CodeGenerationResult:
        fields_str = "\n".join(f"    {name}: {type_}" for name, type_ in fields.items())
        methods_str = "\n".join(f"    def {name}(self{params}):\n        {body}" 
                                for name, (params, body) in methods.items())
        template = self.templates.get("class_def")
        
        if template:
            code = template.render(
                name=name,
                base_classes=", ".join(base_classes),
                docstring=f"Generated class: {name}",
                body=fields_str + "\n\n" + methods_str if methods else fields_str
            )
            return CodeGenerationResult(
                code=code,
                mode=GenerationMode.SYNTHESIS,
                quality=CodeQuality.HIGH,
                confidence=0.85
            )
        return CodeGenerationResult(code="", mode=GenerationMode.SYNTHESIS,
                                    quality=CodeQuality.LOW, confidence=0.0)

    def get_generation_stats(self) -> Dict[str, Any]:
        total = len(self.generation_history)
        successful = len([r for r in self.generation_history if r.is_valid()])
        
        return {
            "total_generations": total,
            "successful_generations": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_confidence": sum(r.confidence for r in self.generation_history) / total if total > 0 else 0.0,
            "template_categories": self.template_categories,
            "template_count": len(self.templates)
        }