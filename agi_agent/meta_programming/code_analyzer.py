import ast
import inspect
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class IssueSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CodeIssue:
    def __init__(self, severity: IssueSeverity, message: str, 
                 line: int = 0, column: int = 0, code: str = ""):
        self.severity = severity
        self.message = message
        self.line = line
        self.column = column
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code": self.code
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "loc": self.loc,
            "functions_count": self.functions_count,
            "classes_count": self.classes_count,
            "nested_depth": self.nested_depth,
            "average_method_length": self.average_method_length,
            "max_method_length": self.max_method_length
        }


class DependencyGraph:
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()
        self.imports: List[Dict[str, Any]] = []

    def add_node(self, name: str):
        self.nodes.add(name)

    def add_edge(self, from_node: str, to_node: str):
        self.edges.add((from_node, to_node))

    def add_import(self, module: str, alias: str = None):
        self.imports.append({"module": module, "alias": alias})

    def has_cycles(self) -> bool:
        in_degree = {node: 0 for node in self.nodes}
        adjacency = {node: [] for node in self.nodes}
        
        for from_node, to_node in self.edges:
            adjacency[from_node].append(to_node)
            in_degree[to_node] += 1
        
        queue = [node for node in self.nodes if in_degree[node] == 0]
        visited = 0
        
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return visited != len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [{"from": f, "to": t} for f, t in self.edges],
            "imports": self.imports,
            "has_cycles": self.has_cycles()
        }


class CodeStructure:
    def __init__(self):
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.modules: List[str] = []
        self.variables: List[Dict[str, Any]] = []
        self.decorators: List[str] = []

    def add_function(self, name: str, params: List[str], 
                     docstring: str, line: int):
        self.functions.append({
            "name": name,
            "params": params,
            "docstring": docstring,
            "line": line
        })

    def add_class(self, name: str, bases: List[str], 
                  docstring: str, line: int):
        self.classes.append({
            "name": name,
            "bases": bases,
            "docstring": docstring,
            "line": line
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functions": self.functions,
            "classes": self.classes,
            "modules": self.modules,
            "variables": self.variables,
            "decorators": self.decorators
        }


class CodeAnalysisResult:
    def __init__(self):
        self.structure: CodeStructure = CodeStructure()
        self.complexity: ComplexityMetrics = ComplexityMetrics()
        self.dependencies: DependencyGraph = DependencyGraph()
        self.issues: List[CodeIssue] = []
        self.raw_ast = None

    def add_issue(self, severity: IssueSeverity, message: str,
                  line: int = 0, column: int = 0, code: str = ""):
        self.issues.append(CodeIssue(severity, message, line, column, code))

    def is_clean(self) -> bool:
        return len([i for i in self.issues if i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)]) == 0

    def get_severity_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure": self.structure.to_dict(),
            "complexity": self.complexity.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "is_clean": self.is_clean(),
            "severity_counts": self.get_severity_counts()
        }


class CodeAnalyzer:
    def __init__(self):
        self.analysis_history: List[CodeAnalysisResult] = []

    def analyze(self, code: str, filename: str = "<unknown>") -> CodeAnalysisResult:
        result = CodeAnalysisResult()
        
        try:
            tree = ast.parse(code)
            result.raw_ast = tree
            
            self._analyze_structure(tree, result)
            self._analyze_complexity(tree, code, result)
            self._analyze_dependencies(tree, result)
            self._detect_issues(tree, code, result)
            
        except SyntaxError as e:
            result.add_issue(
                IssueSeverity.CRITICAL,
                f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset
            )
        
        self.analysis_history.append(result)
        return result

    def _analyze_structure(self, tree: ast.AST, result: CodeAnalysisResult):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                params = [arg.arg for arg in node.args.args]
                docstring = ast.get_docstring(node) or ""
                result.structure.add_function(
                    node.name, params, docstring, node.lineno
                )
            
            elif isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                docstring = ast.get_docstring(node) or ""
                result.structure.add_class(
                    node.name, bases, docstring, node.lineno
                )
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result.structure.modules.append(alias.name)
                    result.dependencies.add_import(alias.name, alias.asname)
            
            elif isinstance(node, ast.ImportFrom):
                result.structure.modules.append(node.module or "")
                result.dependencies.add_import(node.module or "", None)

    def _analyze_complexity(self, tree: ast.AST, code: str, result: CodeAnalysisResult):
        lines = code.split("\n")
        result.complexity.loc = len(lines)
        
        stack = [(tree, 0)]
        max_depth = 0
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    stack.append((child, depth + 1))
        
        result.complexity.nested_depth = max_depth
        result.complexity.functions_count = len(result.structure.functions)
        result.complexity.classes_count = len(result.structure.classes)
        
        if result.complexity.functions_count > 0:
            total_length = sum(
                len(inspect.getsource(func).split("\n")) 
                for func in result.structure.functions
            ) if hasattr(inspect, 'getsource') else len(code)
            result.complexity.average_method_length = total_length / result.complexity.functions_count
        
        result.complexity.cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        result.complexity.cognitive_complexity = self._calculate_cognitive_complexity(tree)

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.IfExp):
                complexity += 1
        
        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
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

    def _analyze_dependencies(self, tree: ast.AST, result: CodeAnalysisResult):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result.dependencies.add_node(node.name)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        result.dependencies.add_edge(node.name, base.id)
            
            elif isinstance(node, ast.FunctionDef):
                result.dependencies.add_node(node.name)

    def _detect_issues(self, tree: ast.AST, code: str, result: CodeAnalysisResult):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not ast.get_docstring(node):
                    result.add_issue(
                        IssueSeverity.WARNING,
                        f"Function '{node.name}' missing docstring",
                        line=node.lineno
                    )
                
                if len(node.args.args) > 5:
                    result.add_issue(
                        IssueSeverity.WARNING,
                        f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        line=node.lineno
                    )
            
            elif isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    result.add_issue(
                        IssueSeverity.WARNING,
                        f"Class '{node.name}' missing docstring",
                        line=node.lineno
                    )
            
            elif isinstance(node, ast.ExceptHandler):
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    result.add_issue(
                        IssueSeverity.WARNING,
                        "Too broad exception catch",
                        line=node.lineno
                    )
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname and len(alias.asname) < 2:
                        result.add_issue(
                            IssueSeverity.INFO,
                            f"Alias '{alias.asname}' for '{alias.name}' is too short",
                            line=node.lineno
                        )

        if result.complexity.cyclomatic_complexity > 20:
            result.add_issue(
                IssueSeverity.ERROR,
                f"High cyclomatic complexity: {result.complexity.cyclomatic_complexity}",
                line=1
            )
        
        if result.complexity.loc > 1000:
            result.add_issue(
                IssueSeverity.WARNING,
                f"File is too long: {result.complexity.loc} lines",
                line=1
            )
        
        if result.dependencies.has_cycles():
            result.add_issue(
                IssueSeverity.CRITICAL,
                "Circular dependencies detected",
                line=1
            )

    def analyze_function(self, func: Callable) -> CodeAnalysisResult:
        try:
            source = inspect.getsource(func)
            return self.analyze(source, filename=inspect.getfile(func))
        except Exception as e:
            result = CodeAnalysisResult()
            result.add_issue(IssueSeverity.ERROR, f"Failed to analyze function: {e}")
            return result

    def analyze_module(self, module) -> CodeAnalysisResult:
        try:
            source = inspect.getsource(module)
            return self.analyze(source, filename=inspect.getfile(module))
        except Exception as e:
            result = CodeAnalysisResult()
            result.add_issue(IssueSeverity.ERROR, f"Failed to analyze module: {e}")
            return result

    def get_analysis_stats(self) -> Dict[str, Any]:
        total = len(self.analysis_history)
        clean = len([r for r in self.analysis_history if r.is_clean()])
        
        all_issues = []
        for result in self.analysis_history:
            all_issues.extend(result.issues)
        
        severity_counts = {s.value: 0 for s in IssueSeverity}
        for issue in all_issues:
            severity_counts[issue.severity.value] += 1
        
        avg_complexity = sum(
            r.complexity.cyclomatic_complexity 
            for r in self.analysis_history
        ) / total if total > 0 else 0.0
        
        return {
            "total_analyses": total,
            "clean_analyses": clean,
            "clean_rate": clean / total if total > 0 else 0.0,
            "avg_cyclomatic_complexity": avg_complexity,
            "issue_counts": severity_counts,
            "total_issues": len(all_issues)
        }