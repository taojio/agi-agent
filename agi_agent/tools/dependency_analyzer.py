import ast
import os
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


class DependencyAnalyzer:
    def __init__(self, project_root: str = "agi_agent"):
        self.project_root = os.path.abspath(project_root)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.all_modules: Set[str] = set()
        self.circular_dependencies: List[List[str]] = []

    def _extract_imports(self, file_path: str) -> List[str]:
        imports = []
        module_name = self._file_to_module(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name.startswith("agi_agent"):
                            imports.append(name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("agi_agent"):
                        base_module = node.module.split(".")[0]
                        imports.append(base_module)
                        
                        full_path = node.module
                        for level in range(len(node.names)):
                            imports.append(full_path)
                            if "." in full_path:
                                full_path = full_path.rsplit(".", 1)[0]
                            else:
                                break
        
        except (SyntaxError, UnicodeDecodeError):
            pass
        
        return [m for m in imports if m != module_name]

    def _file_to_module(self, file_path: str) -> str:
        rel_path = os.path.relpath(file_path, self.project_root)
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]
        if rel_path == "__init__":
            return "agi_agent"
        if rel_path.endswith("__init__"):
            rel_path = rel_path[:-9]
        parts = rel_path.replace(os.sep, ".").split(".")
        return ".".join([p for p in parts if p])

    def _get_module_name(self, file_path: str) -> str:
        rel_path = os.path.relpath(file_path, self.project_root)
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]
        if rel_path.endswith("__init__"):
            rel_path = rel_path[:-9]
        parts = rel_path.replace(os.sep, ".").split(".")
        return "agi_agent." + ".".join([p for p in parts if p])

    def scan(self) -> None:
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "node_modules"]]
            
            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    file_path = os.path.join(root, file)
                    module_name = self._get_module_name(file_path)
                    self.all_modules.add(module_name)
                    
                    imports = self._extract_imports(file_path)
                    for imp in imports:
                        self.dependency_graph[module_name].add(imp)
                        self.reverse_dependency_graph[imp].add(module_name)

    def find_cycles(self) -> List[List[str]]:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            if node not in visited:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in self.dependency_graph.get(node, []):
                    if neighbor not in visited:
                        dfs(neighbor, path.copy())
                    elif neighbor in rec_stack:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:]
                        if cycle not in cycles:
                            cycles.append(cycle)

                rec_stack.discard(node)

        for node in sorted(self.all_modules):
            dfs(node, [])

        self.circular_dependencies = cycles
        return cycles

    def get_dependency_stats(self) -> Dict[str, Any]:
        stats = {
            "total_modules": len(self.all_modules),
            "total_dependencies": sum(len(v) for v in self.dependency_graph.values()),
            "circular_dependencies": len(self.circular_dependencies),
            "modules_with_most_dependencies": sorted(
                self.dependency_graph.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10],
            "modules_with_most_reverse_dependencies": sorted(
                self.reverse_dependency_graph.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10],
            "independent_modules": [
                m for m in self.all_modules if len(self.dependency_graph[m]) == 0
            ],
            "dependency_depth": self._calculate_dependency_depth()
        }
        return stats

    def _calculate_dependency_depth(self) -> Dict[str, int]:
        depth: Dict[str, int] = {}
        
        def calculate(node: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            if node in depth:
                return depth[node]
            if node in visited:
                return 0
            
            visited.add(node)
            max_depth = 0
            for dep in self.dependency_graph.get(node, []):
                max_depth = max(max_depth, calculate(dep, visited.copy()))
            
            depth[node] = max_depth + 1
            return depth[node]

        for node in self.all_modules:
            calculate(node)
        
        return depth

    def generate_report(self) -> str:
        stats = self.get_dependency_stats()
        cycles = self.circular_dependencies

        report = ["# 循环依赖分析报告\n"]
        report.append(f"## 统计概览\n")
        report.append(f"- 总模块数: {stats['total_modules']}\n")
        report.append(f"- 总依赖数: {stats['total_dependencies']}\n")
        report.append(f"- 循环依赖数: {stats['circular_dependencies']}\n")
        report.append(f"\n")

        if cycles:
            report.append(f"## 循环依赖路径\n")
            for i, cycle in enumerate(cycles, 1):
                report.append(f"### 循环 #{i}\n")
                report.append(f" → ".join(cycle) + f" → {cycle[0]}\n")
                report.append(f"\n")

        report.append(f"## 高依赖模块 TOP 10\n")
        for module, deps in stats["modules_with_most_dependencies"]:
            report.append(f"- {module}: {deps} 个依赖\n")
        report.append(f"\n")

        report.append(f"## 被依赖最多模块 TOP 10\n")
        for module, deps in stats["modules_with_most_reverse_dependencies"]:
            report.append(f"- {module}: {deps} 个反向依赖\n")
        report.append(f"\n")

        report.append(f"## 独立模块 (无外部依赖)\n")
        for module in stats["independent_modules"]:
            report.append(f"- {module}\n")

        return "".join(report)


def analyze_dependencies(project_root: str = "agi_agent") -> DependencyAnalyzer:
    analyzer = DependencyAnalyzer(project_root)
    analyzer.scan()
    analyzer.find_cycles()
    return analyzer