import ast
import re
import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class LanguageType(Enum):
    PYTHON = "python"
    C = "c"
    ASSEMBLY = "assembly"


class CodeQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class CodeIssue:
    def __init__(self, severity: str, message: str, line: int = 0, 
                 column: int = 0, code: str = "", category: str = ""):
        self.severity = severity
        self.message = message
        self.line = line
        self.column = column
        self.code = code
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "category": self.category
        }


class ComplexityMetrics:
    def __init__(self):
        self.cyclomatic_complexity: int = 0
        self.cognitive_complexity: int = 0
        self.loc: int = 0
        self.functions_count: int = 0
        self.classes_count: int = 0
        self.nested_depth: int = 0
        self.average_method_length: float = 0.0
        self.max_method_length: int = 0
        self.coupling_score: float = 0.0
        self.cohesion_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "loc": self.loc,
            "functions_count": self.functions_count,
            "classes_count": self.classes_count,
            "nested_depth": self.nested_depth,
            "average_method_length": self.average_method_length,
            "max_method_length": self.max_method_length,
            "coupling_score": self.coupling_score,
            "cohesion_score": self.cohesion_score
        }


class OptimizationPriority(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    NONE = 1


class OptimizationSuggestion:
    def __init__(self, priority: OptimizationPriority, category: str, 
                 description: str, code_change: str = "", 
                 expected_improvement: float = 0.0):
        self.priority = priority
        self.category = category
        self.description = description
        self.code_change = code_change
        self.expected_improvement = expected_improvement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority.value,
            "priority_label": self.priority.name,
            "category": self.category,
            "description": self.description,
            "code_change": self.code_change,
            "expected_improvement": self.expected_improvement
        }


class CodeAnalysisResult:
    def __init__(self):
        self.language: LanguageType = LanguageType.PYTHON
        self.complexity: ComplexityMetrics = ComplexityMetrics()
        self.issues: List[CodeIssue] = []
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        self.quality_score: float = 0.0
        self.quality_level: CodeQualityLevel = CodeQualityLevel.FAIR
        self.structure: Dict[str, Any] = {}
        self.dependencies: Dict[str, Any] = {"nodes": [], "edges": [], "imports": []}
        self.raw_analysis: Dict[str, Any] = {}

    def add_issue(self, severity: str, message: str, line: int = 0, 
                  column: int = 0, code: str = "", category: str = ""):
        self.issues.append(CodeIssue(severity, message, line, column, code, category))

    def add_suggestion(self, priority: OptimizationPriority, category: str, 
                       description: str, code_change: str = "", 
                       expected_improvement: float = 0.0):
        self.optimization_suggestions.append(
            OptimizationSuggestion(priority, category, description, code_change, expected_improvement)
        )

    def is_clean(self) -> bool:
        return len([i for i in self.issues if i.severity in ("error", "critical")]) == 0

    def get_severity_counts(self) -> Dict[str, int]:
        counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language.value,
            "complexity": self.complexity.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "optimization_suggestions": [s.to_dict() for s in self.optimization_suggestions],
            "quality_score": self.quality_score,
            "quality_level": self.quality_level.value,
            "structure": self.structure,
            "dependencies": self.dependencies,
            "is_clean": self.is_clean(),
            "severity_counts": self.get_severity_counts()
        }


class PythonAnalyzer:
    def analyze(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        result.language = LanguageType.PYTHON

        try:
            tree = ast.parse(code)
            result.raw_analysis["ast_valid"] = True

            self._analyze_structure(tree, code, result)
            self._calculate_complexity(tree, code, result)
            self._analyze_dependencies(tree, result)
            self._detect_python_issues(tree, code, result)
            self._generate_python_suggestions(result)
            self._compute_quality_score(result)

        except SyntaxError as e:
            result.add_issue("critical", f"Syntax error: {e.msg}", line=e.lineno, column=e.offset)
            result.raw_analysis["ast_valid"] = False

        return result

    def _analyze_structure(self, tree: ast.AST, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        functions = []
        classes = []
        variables = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                params = [arg.arg for arg in node.args.args]
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 10
                func_lines = lines[node.lineno-1:end_line-1]
                func_length = len(func_lines)
                functions.append({
                    "name": node.name,
                    "params": params,
                    "line": node.lineno,
                    "length": func_length,
                    "docstring": ast.get_docstring(node) or ""
                })

            elif isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                classes.append({
                    "name": node.name,
                    "bases": bases,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or ""
                })

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "line": node.lineno
                        })

        result.structure = {
            "functions": functions,
            "classes": classes,
            "variables": variables
        }

    def _calculate_complexity(self, tree: ast.AST, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        result.complexity.loc = len(lines)
        result.complexity.functions_count = len(result.structure.get("functions", []))
        result.complexity.classes_count = len(result.structure.get("classes", []))

        result.complexity.cyclomatic_complexity = self._cyclomatic_complexity(tree)
        result.complexity.cognitive_complexity = self._cognitive_complexity(tree)
        result.complexity.nested_depth = self._nested_depth(tree)

        if result.complexity.functions_count > 0:
            total_length = sum(f["length"] for f in result.structure["functions"])
            result.complexity.average_method_length = total_length / result.complexity.functions_count
            result.complexity.max_method_length = max(f["length"] for f in result.structure["functions"])

        result.complexity.cohesion_score = self._calculate_cohesion(result)
        result.complexity.coupling_score = self._calculate_coupling(result)

    def _cyclomatic_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.IfExp):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
        return complexity

    def _cognitive_complexity(self, tree: ast.AST) -> int:
        complexity = 0
        def visit(node, depth=0):
            nonlocal complexity
            if isinstance(node, ast.If):
                complexity += 1 + depth
                for child in node.body:
                    visit(child, depth + 1)
                for child in node.orelse:
                    visit(child, depth + 1)
            elif isinstance(node, ast.For) or isinstance(node, ast.While):
                complexity += 1 + depth
                for child in node.body:
                    visit(child, depth + 1)
                for child in node.orelse:
                    visit(child, depth + 1)
            elif isinstance(node, ast.And) or isinstance(node, ast.Or):
                complexity += 1
            for child in ast.iter_child_nodes(node):
                visit(child, depth)
        visit(tree)
        return complexity

    def _nested_depth(self, tree: ast.AST) -> int:
        max_depth = 0
        def visit(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.If, ast.For, ast.While, ast.With)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)
        visit(tree)
        return max_depth

    def _calculate_cohesion(self, result: CodeAnalysisResult) -> float:
        functions = result.structure.get("functions", [])
        if len(functions) < 2:
            return 1.0
        shared_vars = 0
        total_possible = 0
        for i, f1 in enumerate(functions):
            vars1 = set(f1.get("params", []))
            for j, f2 in enumerate(functions):
                if i != j:
                    vars2 = set(f2.get("params", []))
                    shared_vars += len(vars1 & vars2)
                    total_possible += len(vars1 | vars2)
        return shared_vars / max(total_possible, 1)

    def _calculate_coupling(self, result: CodeAnalysisResult) -> float:
        imports = result.dependencies.get("imports", [])
        functions = result.structure.get("functions", [])
        if not functions:
            return 0.0
        return min(1.0, len(imports) / len(functions))

    def _analyze_dependencies(self, tree: ast.AST, result: CodeAnalysisResult):
        nodes = set()
        edges = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "alias": alias.asname})
                    nodes.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append({"module": node.module, "alias": None})
                    nodes.add(node.module)

            elif isinstance(node, ast.ClassDef):
                nodes.add(node.name)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        nodes.add(base.id)
                        edges.append({"from": node.name, "to": base.id})

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                nodes.add(node.name)

        result.dependencies = {
            "nodes": list(nodes),
            "edges": edges,
            "imports": imports,
            "has_cycles": self._has_cycles(edges)
        }

    def _has_cycles(self, edges: List[Dict[str, str]]) -> bool:
        graph = {}
        for edge in edges:
            if edge["from"] not in graph:
                graph[edge["from"]] = []
            graph[edge["from"]].append(edge["to"])

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def _detect_python_issues(self, tree: ast.AST, code: str, result: CodeAnalysisResult):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not ast.get_docstring(node):
                    result.add_issue("warning", f"Function '{node.name}' missing docstring", 
                                     line=node.lineno, category="documentation")

                if len(node.args.args) > 5:
                    result.add_issue("warning", f"Function '{node.name}' has too many parameters ({len(node.args.args)})", 
                                     line=node.lineno, category="complexity")

            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    result.add_issue("warning", f"Class '{node.name}' missing docstring", 
                                     line=node.lineno, category="documentation")

            elif isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    result.add_issue("warning", "Too broad exception catch", 
                                     line=node.lineno, category="error_handling")

            elif isinstance(node, ast.Global):
                result.add_issue("warning", "Use of global variables", 
                                 line=node.lineno, category="design")

            elif isinstance(node, ast.NamedExpr):
                result.add_issue("info", "Assignment expression used", 
                                 line=node.lineno, category="style")

        if result.complexity.cyclomatic_complexity > 20:
            result.add_issue("error", f"High cyclomatic complexity: {result.complexity.cyclomatic_complexity}", 
                             category="complexity")

        if result.complexity.loc > 1000:
            result.add_issue("warning", f"File is too long: {result.complexity.loc} lines", 
                             category="maintainability")

        if result.dependencies.get("has_cycles", False):
            result.add_issue("critical", "Circular dependencies detected", 
                             category="design")

    def _generate_python_suggestions(self, result: CodeAnalysisResult):
        cmplx = result.complexity

        if cmplx.cyclomatic_complexity > 15:
            result.add_suggestion(
                OptimizationPriority.HIGH,
                "complexity_reduction",
                f"Consider refactoring to reduce cyclomatic complexity from {cmplx.cyclomatic_complexity}",
                expected_improvement=0.2
            )

        if cmplx.nested_depth > 4:
            result.add_suggestion(
                OptimizationPriority.MEDIUM,
                "nested_depth_reduction",
                f"Reduce nesting depth from {cmplx.nested_depth} to improve readability",
                expected_improvement=0.15
            )

        if cmplx.average_method_length > 50:
            result.add_suggestion(
                OptimizationPriority.MEDIUM,
                "method_refactoring",
                f"Average method length is {cmplx.average_method_length:.1f} lines, consider splitting",
                expected_improvement=0.12
            )

        if cmplx.coupling_score > 0.7:
            result.add_suggestion(
                OptimizationPriority.HIGH,
                "decoupling",
                "High coupling detected, consider reducing external dependencies",
                expected_improvement=0.18
            )

        docstring_issues = [i for i in result.issues if i.category == "documentation"]
        if len(docstring_issues) > 0:
            result.add_suggestion(
                OptimizationPriority.LOW,
                "documentation",
                f"Add docstrings to {len(docstring_issues)} functions/classes",
                expected_improvement=0.05
            )

    def _compute_quality_score(self, result: CodeAnalysisResult):
        score = 1.0

        cmplx = result.complexity
        score -= min(0.3, (cmplx.cyclomatic_complexity - 10) / 50)
        score -= min(0.2, (cmplx.cognitive_complexity - 15) / 50)
        score -= min(0.15, (cmplx.nested_depth - 3) / 10)
        score -= min(0.1, (cmplx.average_method_length - 30) / 100)

        score -= len([i for i in result.issues if i.severity == "critical"]) * 0.2
        score -= len([i for i in result.issues if i.severity == "error"]) * 0.1
        score -= len([i for i in result.issues if i.severity == "warning"]) * 0.02

        score += cmplx.cohesion_score * 0.1
        score -= cmplx.coupling_score * 0.05

        result.quality_score = max(0.0, min(1.0, score))

        if result.quality_score >= 0.9:
            result.quality_level = CodeQualityLevel.EXCELLENT
        elif result.quality_score >= 0.75:
            result.quality_level = CodeQualityLevel.GOOD
        elif result.quality_score >= 0.5:
            result.quality_level = CodeQualityLevel.FAIR
        else:
            result.quality_level = CodeQualityLevel.POOR


class CAnalyzer:
    def analyze(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        result.language = LanguageType.C

        self._tokenize_c(code, result)
        self._analyze_c_structure(code, result)
        self._calculate_c_complexity(code, result)
        self._detect_c_issues(code, result)
        self._generate_c_suggestions(result)
        self._compute_quality_score(result)

        return result

    def _tokenize_c(self, code: str, result: CodeAnalysisResult):
        tokens = []
        token_pattern = re.compile(
            r'\b(?:int|float|double|char|void|if|else|for|while|do|switch|case|default|'
            r'return|break|continue|struct|typedef|enum|static|extern|const|volatile|'
            r'size_t|NULL|true|false)\b|'
            r'"(?:[^"\\]|\\.)*"|'
            r"'(?:[^'\\]|\\.)'|"
            r'[a-zA-Z_][a-zA-Z0-9_]*|'
            r'\d+\.?\d*|'
            r'[+\-*/%=<>!&|^~]|[(){}[\];,.]'
        )
        for match in token_pattern.finditer(code):
            tokens.append(match.group())
        result.raw_analysis["tokens"] = tokens

    def _analyze_c_structure(self, code: str, result: CodeAnalysisResult):
        functions = []
        structs = []
        macros = []

        func_pattern = re.compile(
            r'\b(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{',
            re.MULTILINE
        )
        for match in func_pattern.finditer(code):
            return_type = match.group(1)
            func_name = match.group(2)
            params = [p.strip() for p in match.group(3).split(',') if p.strip()]
            functions.append({
                "name": func_name,
                "return_type": return_type,
                "params": params,
                "line": code.count('\n', 0, match.start()) + 1
            })

        struct_pattern = re.compile(r'\bstruct\s+(\w+)\s*\{')
        for match in struct_pattern.finditer(code):
            structs.append({
                "name": match.group(1),
                "line": code.count('\n', 0, match.start()) + 1
            })

        macro_pattern = re.compile(r'#define\s+(\w+)\s+(.+)')
        for match in macro_pattern.finditer(code):
            macros.append({
                "name": match.group(1),
                "value": match.group(2).strip(),
                "line": code.count('\n', 0, match.start()) + 1
            })

        result.structure = {
            "functions": functions,
            "structs": structs,
            "macros": macros
        }

    def _calculate_c_complexity(self, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        result.complexity.loc = len(lines)
        result.complexity.functions_count = len(result.structure.get("functions", []))

        complexity = 1
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('if') or stripped.startswith('for') or stripped.startswith('while'):
                if '(' in stripped and ')' in stripped:
                    complexity += 1
            if stripped.startswith('case ') or stripped.startswith('default:'):
                complexity += 1
            if '&&' in stripped or '||' in stripped:
                complexity += stripped.count('&&') + stripped.count('||')

        result.complexity.cyclomatic_complexity = complexity
        result.complexity.cognitive_complexity = complexity

        brace_count = 0
        max_brace_depth = 0
        for char in code:
            if char == '{':
                brace_count += 1
                max_brace_depth = max(max_brace_depth, brace_count)
            elif char == '}':
                brace_count -= 1
        result.complexity.nested_depth = max_brace_depth

        result.complexity.coupling_score = min(1.0, code.count('#include') / max(result.complexity.functions_count, 1))
        result.complexity.cohesion_score = 0.8 if result.complexity.functions_count > 0 else 1.0
        result.complexity.average_method_length = 0
        result.complexity.max_method_length = 0

    def _detect_c_issues(self, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            if 'malloc(' in stripped and 'free(' not in code:
                result.add_issue("warning", "Possible memory leak - malloc without free", 
                                 line=i+1, category="memory")

            if 'strcpy(' in stripped or 'strcat(' in stripped:
                result.add_issue("error", "Use of unsafe string function (strcpy/strcat)", 
                                 line=i+1, category="security")

            if 'gets(' in stripped:
                result.add_issue("critical", "Use of extremely unsafe gets()", 
                                 line=i+1, category="security")

            if 'scanf("' in stripped and '%s' in stripped:
                result.add_issue("warning", "Potential buffer overflow in scanf", 
                                 line=i+1, category="security")

            if '=' in stripped and '==' not in stripped and '!=' not in stripped:
                if re.search(r'\bif\s*\([^=]*=[^=][^)]*\)', stripped):
                    result.add_issue("warning", "Possible assignment in condition", 
                                     line=i+1, category="logic")

            if 'void main' in stripped:
                result.add_issue("warning", "Non-standard void main()", 
                                 line=i+1, category="standard")

            if 'goto ' in stripped:
                result.add_issue("warning", "Use of goto statement", 
                                 line=i+1, category="style")

    def _generate_c_suggestions(self, result: CodeAnalysisResult):
        cmplx = result.complexity

        if cmplx.cyclomatic_complexity > 15:
            result.add_suggestion(
                OptimizationPriority.HIGH,
                "complexity_reduction",
                f"Reduce cyclomatic complexity from {cmplx.cyclomatic_complexity}",
                expected_improvement=0.22
            )

        security_issues = [i for i in result.issues if i.category == "security"]
        if security_issues:
            result.add_suggestion(
                OptimizationPriority.CRITICAL,
                "security_fixes",
                f"Fix {len(security_issues)} security issues",
                expected_improvement=0.3
            )

        memory_issues = [i for i in result.issues if i.category == "memory"]
        if memory_issues:
            result.add_suggestion(
                OptimizationPriority.HIGH,
                "memory_management",
                f"Address {len(memory_issues)} memory management issues",
                expected_improvement=0.25
            )

        if cmplx.nested_depth > 4:
            result.add_suggestion(
                OptimizationPriority.MEDIUM,
                "nested_depth_reduction",
                f"Reduce nesting depth from {cmplx.nested_depth}",
                expected_improvement=0.15
            )

    def _compute_quality_score(self, result: CodeAnalysisResult):
        score = 1.0
        cmplx = result.complexity

        score -= min(0.3, (cmplx.cyclomatic_complexity - 10) / 50)
        score -= min(0.2, (cmplx.nested_depth - 3) / 10)

        score -= len([i for i in result.issues if i.severity == "critical"]) * 0.3
        score -= len([i for i in result.issues if i.severity == "error"]) * 0.15
        score -= len([i for i in result.issues if i.severity == "warning"]) * 0.03

        result.quality_score = max(0.0, min(1.0, score))

        if result.quality_score >= 0.9:
            result.quality_level = CodeQualityLevel.EXCELLENT
        elif result.quality_score >= 0.75:
            result.quality_level = CodeQualityLevel.GOOD
        elif result.quality_score >= 0.5:
            result.quality_level = CodeQualityLevel.FAIR
        else:
            result.quality_level = CodeQualityLevel.POOR


class AssemblyAnalyzer:
    def analyze(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        result.language = LanguageType.ASSEMBLY

        self._tokenize_assembly(code, result)
        self._analyze_assembly_structure(code, result)
        self._calculate_assembly_complexity(code, result)
        self._detect_assembly_issues(code, result)
        self._generate_assembly_suggestions(code, result)
        self._compute_quality_score(result)

        return result

    def _tokenize_assembly(self, code: str, result: CodeAnalysisResult):
        instructions = []
        directives = []
        labels = []

        lines = code.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue

            if stripped.startswith('.'):
                directives.append(stripped)
            elif ':' in stripped:
                parts = stripped.split(':')
                labels.append(parts[0].strip())
            else:
                parts = stripped.split()
                if parts:
                    instructions.append(parts[0])

        result.raw_analysis["instructions"] = instructions
        result.raw_analysis["directives"] = directives
        result.raw_analysis["labels"] = labels

    def _analyze_assembly_structure(self, code: str, result: CodeAnalysisResult):
        functions = []
        labels = []

        func_pattern = re.compile(r'(\w+):\s*;?\s*function')
        for match in func_pattern.finditer(code):
            functions.append({
                "name": match.group(1),
                "line": code.count('\n', 0, match.start()) + 1
            })

        label_pattern = re.compile(r'^(\w+):', re.MULTILINE)
        for match in label_pattern.finditer(code):
            labels.append({
                "name": match.group(1),
                "line": code.count('\n', 0, match.start()) + 1
            })

        result.structure = {
            "functions": functions,
            "labels": labels
        }

    def _calculate_assembly_complexity(self, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        result.complexity.loc = len(lines)
        result.complexity.functions_count = len(result.structure.get("functions", []))

        branches = code.count('jmp ') + code.count('jz ') + code.count('jnz ') + \
                   code.count('je ') + code.count('jne ') + code.count('jl ') + \
                   code.count('jg ') + code.count('call ')

        result.complexity.cyclomatic_complexity = 1 + branches
        result.complexity.cognitive_complexity = branches
        result.complexity.nested_depth = code.count('push ') - code.count('pop ') // 4
        result.complexity.cohesion_score = 0.7
        result.complexity.coupling_score = 0.3
        result.complexity.average_method_length = 0
        result.complexity.max_method_length = 0

    def _detect_assembly_issues(self, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            if 'ret' not in code:
                result.add_issue("warning", "Missing return instruction", 
                                 line=i+1, category="control_flow")

            if 'int 0x80' in stripped or 'syscall' in stripped:
                result.add_issue("info", "System call detected", 
                                 line=i+1, category="system")

            if 'cli' in stripped or 'sti' in stripped:
                result.add_issue("warning", "Interrupt state modification", 
                                 line=i+1, category="system")

            if 'hlt' in stripped:
                result.add_issue("warning", "Halt instruction", 
                                 line=i+1, category="system")

            if 'xor' in stripped and 'eax' in stripped and 'eax' in stripped:
                result.add_issue("info", "Common zeroing pattern", 
                                 line=i+1, category="optimization")

    def _generate_assembly_suggestions(self, code: str, result: CodeAnalysisResult):
        cmplx = result.complexity

        if cmplx.cyclomatic_complexity > 20:
            result.add_suggestion(
                OptimizationPriority.HIGH,
                "branch_reduction",
                f"Reduce branch count from {cmplx.cyclomatic_complexity - 1}",
                expected_improvement=0.25
            )

        if 'xor eax, eax' not in code:
            result.add_suggestion(
                OptimizationPriority.LOW,
                "optimization",
                "Consider using 'xor eax, eax' for zeroing registers",
                expected_improvement=0.05
            )

    def _compute_quality_score(self, result: CodeAnalysisResult):
        score = 1.0
        cmplx = result.complexity

        score -= min(0.3, (cmplx.cyclomatic_complexity - 15) / 50)

        score -= len([i for i in result.issues if i.severity == "critical"]) * 0.25
        score -= len([i for i in result.issues if i.severity == "error"]) * 0.12
        score -= len([i for i in result.issues if i.severity == "warning"]) * 0.025

        result.quality_score = max(0.0, min(1.0, score))

        if result.quality_score >= 0.9:
            result.quality_level = CodeQualityLevel.EXCELLENT
        elif result.quality_score >= 0.75:
            result.quality_level = CodeQualityLevel.GOOD
        elif result.quality_score >= 0.5:
            result.quality_level = CodeQualityLevel.FAIR
        else:
            result.quality_level = CodeQualityLevel.POOR


class CrossLanguageCodeAnalyzer:
    def __init__(self):
        self.python_analyzer = PythonAnalyzer()
        self.c_analyzer = CAnalyzer()
        self.assembly_analyzer = AssemblyAnalyzer()
        self.analysis_history: List[CodeAnalysisResult] = []

    def detect_language(self, code: str, filename: str = "") -> LanguageType:
        if filename.endswith('.py'):
            return LanguageType.PYTHON
        elif filename.endswith('.c') or filename.endswith('.h'):
            return LanguageType.C
        elif filename.endswith('.asm') or filename.endswith('.s'):
            return LanguageType.ASSEMBLY

        if 'import ' in code or 'def ' in code or 'class ' in code:
            return LanguageType.PYTHON
        elif '#include' in code or 'void main' in code:
            return LanguageType.C
        elif 'jmp ' in code or 'mov ' in code or 'push ' in code:
            return LanguageType.ASSEMBLY

        return LanguageType.PYTHON

    def analyze(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        language = self.detect_language(code, filename)

        if language == LanguageType.PYTHON:
            result = self.python_analyzer.analyze(code, filename)
        elif language == LanguageType.C:
            result = self.c_analyzer.analyze(code, filename)
        elif language == LanguageType.ASSEMBLY:
            result = self.assembly_analyzer.analyze(code, filename)
        else:
            result = self.python_analyzer.analyze(code, filename)

        result.raw_analysis["analysis_time"] = time.time()
        self.analysis_history.append(result)
        return result

    def analyze_python(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        return self.python_analyzer.analyze(code, filename)

    def analyze_c(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        return self.c_analyzer.analyze(code, filename)

    def analyze_assembly(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        return self.assembly_analyzer.analyze(code, filename)

    def analyze_file(self, file_path: str) -> Optional[CodeAnalysisResult]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.analyze(code, file_path)
        except Exception:
            return None

    def get_summary_report(self) -> Dict[str, Any]:
        if not self.analysis_history:
            return {"total_analyses": 0}

        total = len(self.analysis_history)
        clean = len([r for r in self.analysis_history if r.is_clean()])

        quality_distribution = {q.value: 0 for q in CodeQualityLevel}
        for result in self.analysis_history:
            quality_distribution[result.quality_level.value] += 1

        avg_quality = sum(r.quality_score for r in self.analysis_history) / total

        language_distribution = {}
        for result in self.analysis_history:
            lang = result.language.value
            language_distribution[lang] = language_distribution.get(lang, 0) + 1

        all_issues = []
        for result in self.analysis_history:
            all_issues.extend(result.issues)

        severity_counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for issue in all_issues:
            if issue.severity in severity_counts:
                severity_counts[issue.severity] += 1

        return {
            "total_analyses": total,
            "clean_analyses": clean,
            "clean_rate": clean / total,
            "avg_quality_score": avg_quality,
            "quality_distribution": quality_distribution,
            "language_distribution": language_distribution,
            "total_issues": len(all_issues),
            "severity_counts": severity_counts,
            "total_suggestions": sum(len(r.optimization_suggestions) for r in self.analysis_history)
        }

    def get_top_optimizations(self, limit: int = 10) -> List[OptimizationSuggestion]:
        all_suggestions = []
        for result in self.analysis_history:
            all_suggestions.extend(result.optimization_suggestions)

        all_suggestions.sort(key=lambda s: (s.priority.value, s.expected_improvement), reverse=True)
        return all_suggestions[:limit]

    def reset(self):
        self.analysis_history.clear()