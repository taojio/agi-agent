import ast
import inspect
import random
import string
import types
import unittest
import coverage
import io
import json
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


class TestType(Enum):
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    REGRESSION_TEST = "regression_test"


class TestStrategy(Enum):
    BOUNDARY_VALUE = "boundary_value"
    RANDOM_TEST = "random_test"
    EQUIVALENCE_PARTITIONING = "equivalence_partitioning"
    PATH_COVERING = "path_covering"
    MUTATION_TESTING = "mutation_testing"
    EXHAUSTIVE = "exhaustive"


class TestResultStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    test_id: str
    test_name: str
    test_type: TestType
    strategy: TestStrategy
    target_function: str
    input_data: Dict[str, Any]
    expected_output: Optional[Any] = None
    expected_exception: Optional[type] = None
    description: str = ""
    priority: int = 1


@dataclass
class TestResult:
    test_id: str
    test_name: str
    status: TestResultStatus
    actual_output: Optional[Any] = None
    expected_output: Optional[Any] = None
    execution_time_ms: float = 0.0
    error_message: str = ""
    stack_trace: str = ""


@dataclass
class CoverageReport:
    function_coverage: float = 0.0
    branch_coverage: float = 0.0
    line_coverage: float = 0.0
    covered_lines: Set[int] = field(default_factory=set)
    uncovered_lines: Set[int] = field(default_factory=set)
    covered_branches: Set[Tuple[int, int]] = field(default_factory=set)
    uncovered_branches: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass
class TestSuite:
    suite_id: str
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    coverage_report: Optional[CoverageReport] = None
    created_at: float = 0.0
    executed_at: float = 0.0


class InputGenerator:
    def __init__(self):
        self.type_generators = {
            int: self._generate_int,
            float: self._generate_float,
            str: self._generate_string,
            bool: self._generate_bool,
            list: self._generate_list,
            dict: self._generate_dict,
            tuple: self._generate_tuple,
            set: self._generate_set,
        }

    def generate_inputs(self, func: Callable, strategy: TestStrategy,
                        count: int = 10) -> List[Dict[str, Any]]:
        try:
            sig = inspect.signature(func)
            inputs = []
            for _ in range(count):
                args = {}
                for param_name, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty:
                        args[param_name] = self._generate_value(param.annotation, strategy)
                    else:
                        if strategy == TestStrategy.BOUNDARY_VALUE:
                            args[param_name] = self._generate_boundary_value(param.annotation, param.default)
                        else:
                            args[param_name] = self._generate_value(param.annotation, strategy)
                inputs.append(args)
            return inputs
        except (ValueError, TypeError):
            return [{}]

    def _generate_value(self, annotation: Any, strategy: TestStrategy) -> Any:
        if annotation == inspect.Parameter.empty:
            return self._generate_random()
        
        if annotation in self.type_generators:
            return self.type_generators[annotation](strategy)
        
        if hasattr(annotation, '__origin__'):
            return self._generate_generic(annotation, strategy)
        
        return self._generate_random()

    def _generate_generic(self, annotation: Any, strategy: TestStrategy) -> Any:
        origin = annotation.__origin__
        args = annotation.__args__ if hasattr(annotation, '__args__') else []
        
        if origin == list and args:
            return [self._generate_value(args[0], strategy) for _ in range(random.randint(0, 5))]
        if origin == dict and len(args) >= 2:
            return {self._generate_value(args[0], strategy): self._generate_value(args[1], strategy) 
                    for _ in range(random.randint(0, 3))}
        if origin == tuple and args:
            return tuple(self._generate_value(arg, strategy) for arg in args)
        if origin == set and args:
            return {self._generate_value(args[0], strategy) for _ in range(random.randint(0, 3))}
        
        return self._generate_random()

    def _generate_int(self, strategy: TestStrategy) -> int:
        if strategy == TestStrategy.BOUNDARY_VALUE:
            return random.choice([-2147483648, -1, 0, 1, 10, 100, 2147483647])
        if strategy == TestStrategy.EQUIVALENCE_PARTITIONING:
            return random.choice([random.randint(-1000, -1), 0, random.randint(1, 1000)])
        return random.randint(-10000, 10000)

    def _generate_float(self, strategy: TestStrategy) -> float:
        if strategy == TestStrategy.BOUNDARY_VALUE:
            return random.choice([-float('inf'), float('-inf'), -1.0, 0.0, 1.0, float('inf'), 
                                 1e-10, 1e10, float('nan')])
        return random.uniform(-10000.0, 10000.0)

    def _generate_string(self, strategy: TestStrategy) -> str:
        if strategy == TestStrategy.BOUNDARY_VALUE:
            choices = ["", "a", "ab", string.ascii_lowercase, string.ascii_uppercase,
                       string.digits, string.punctuation, " " * 100]
            return random.choice(choices)
        length = random.randint(0, 50)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_bool(self, strategy: TestStrategy) -> bool:
        return random.choice([True, False])

    def _generate_list(self, strategy: TestStrategy) -> List:
        length = random.randint(0, 10) if strategy != TestStrategy.BOUNDARY_VALUE else random.choice([0, 1, 5, 10])
        return [self._generate_random() for _ in range(length)]

    def _generate_dict(self, strategy: TestStrategy) -> Dict:
        length = random.randint(0, 5) if strategy != TestStrategy.BOUNDARY_VALUE else random.choice([0, 1, 3])
        return {str(i): self._generate_random() for i in range(length)}

    def _generate_tuple(self, strategy: TestStrategy) -> Tuple:
        length = random.randint(0, 5) if strategy != TestStrategy.BOUNDARY_VALUE else random.choice([0, 1, 3])
        return tuple(self._generate_random() for _ in range(length))

    def _generate_set(self, strategy: TestStrategy) -> Set:
        length = random.randint(0, 5) if strategy != TestStrategy.BOUNDARY_VALUE else random.choice([0, 1, 3])
        return {self._generate_random() for _ in range(length)}

    def _generate_random(self) -> Any:
        return random.choice([
            random.randint(-1000, 1000),
            random.uniform(-1000.0, 1000.0),
            ''.join(random.choices(string.ascii_letters, k=random.randint(0, 10))),
            random.choice([True, False]),
            None,
        ])

    def _generate_boundary_value(self, annotation: Any, default: Any) -> Any:
        if annotation == int:
            return random.choice([default - 1, default, default + 1, 0, -1, 1])
        if annotation == float:
            return random.choice([default - 0.001, default, default + 0.001, 0.0])
        if annotation == str:
            return random.choice(["", default, default[:-1] if default else "", default + "x"])
        if annotation == bool:
            return random.choice([True, False])
        return default


class CodeAnalyzer:
    def __init__(self):
        self.function_info_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        func_id = f"{func.__module__}.{func.__name__}"
        if func_id in self.function_info_cache:
            return self.function_info_cache[func_id]

        info = {
            "name": func.__name__,
            "module": func.__module__,
            "parameters": [],
            "return_type": None,
            "docstring": inspect.getdoc(func),
            "source_code": inspect.getsource(func) if hasattr(func, '__code__') else "",
            "complexity": 0,
            "branch_count": 0,
            "line_count": 0,
            "has_recursion": False,
            "calls": [],
        }

        try:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                info["parameters"].append({
                    "name": param_name,
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "has_default": param.default != inspect.Parameter.empty,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                })
                info["return_type"] = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
        except (ValueError, TypeError):
            pass

        try:
            source = info["source_code"]
            tree = ast.parse(source)
            info["line_count"] = len(source.splitlines())
            info["complexity"] = self._calculate_complexity(tree)
            info["branch_count"] = self._count_branches(tree)
            info["has_recursion"] = self._detect_recursion(tree, func.__name__)
            info["calls"] = self._extract_calls(tree)
        except SyntaxError:
            pass

        self.function_info_cache[func_id] = info
        return info

    def _calculate_complexity(self, tree: ast.AST) -> int:
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.IfExp):
                complexity += 1
        return complexity

    def _count_branches(self, tree: ast.AST) -> int:
        branches = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                branches += 2
            elif isinstance(node, (ast.And, ast.Or)):
                branches += 1
        return branches

    def _detect_recursion(self, tree: ast.AST, func_name: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == func_name:
                return True
        return False

    def _extract_calls(self, tree: ast.AST) -> List[str]:
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        return list(set(calls))

    def analyze_module(self, module: types.ModuleType) -> List[Dict[str, Any]]:
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                if name.startswith('_'):
                    continue
                functions.append(self.analyze_function(obj))
        return functions


class TestCaseGenerator:
    def __init__(self):
        self.input_generator = InputGenerator()
        self.code_analyzer = CodeAnalyzer()

    def generate_test_cases(self, func: Callable, strategies: List[TestStrategy],
                            count_per_strategy: int = 5) -> List[TestCase]:
        test_cases = []
        func_info = self.code_analyzer.analyze_function(func)
        func_name = func_info["name"]

        for strategy in strategies:
            inputs = self.input_generator.generate_inputs(func, strategy, count_per_strategy)
            for i, input_data in enumerate(inputs):
                test_id = f"test_{func_name}_{strategy.value}_{i}"
                test_name = f"test_{func_name}_{strategy.value}_{i}"
                
                expected_output = None
                expected_exception = None
                description = f"{strategy.value} test for {func_name}"

                test_case = TestCase(
                    test_id=test_id,
                    test_name=test_name,
                    test_type=TestType.UNIT_TEST,
                    strategy=strategy,
                    target_function=func_name,
                    input_data=input_data,
                    expected_output=expected_output,
                    expected_exception=expected_exception,
                    description=description,
                    priority=self._determine_priority(func_info, strategy)
                )
                test_cases.append(test_case)

        return test_cases

    def _determine_priority(self, func_info: Dict[str, Any], strategy: TestStrategy) -> int:
        complexity = func_info.get("complexity", 0)
        if strategy == TestStrategy.BOUNDARY_VALUE:
            return 1
        if complexity > 10:
            return 1
        if func_info.get("has_recursion", False):
            return 1
        return 2

    def generate_regression_tests(self, func: Callable, previous_results: List[TestResult]) -> List[TestCase]:
        regression_tests = []
        func_name = func.__name__
        
        for result in previous_results:
            if result.status == TestResultStatus.FAILED:
                test_id = f"regression_{func_name}_{result.test_id}"
                test_case = TestCase(
                    test_id=test_id,
                    test_name=f"test_{func_name}_regression",
                    test_type=TestType.REGRESSION_TEST,
                    strategy=TestStrategy.BOUNDARY_VALUE,
                    target_function=func_name,
                    input_data={},
                    description=f"Regression test for previously failed case {result.test_id}",
                    priority=1
                )
                regression_tests.append(test_case)
        
        return regression_tests


class TestExecutor:
    def __init__(self):
        self.coverage = None

    def execute_test(self, func: Callable, test_case: TestCase) -> TestResult:
        start_time = time.time()
        
        try:
            if test_case.expected_exception:
                try:
                    result = func(**test_case.input_data)
                    status = TestResultStatus.FAILED
                    actual_output = result
                    error_message = f"Expected {test_case.expected_exception.__name__} but got {type(result).__name__}"
                except test_case.expected_exception:
                    status = TestResultStatus.PASSED
                    actual_output = None
                    error_message = ""
                except Exception as e:
                    status = TestResultStatus.FAILED
                    actual_output = None
                    error_message = f"Expected {test_case.expected_exception.__name__} but got {type(e).__name__}: {e}"
            else:
                result = func(**test_case.input_data)
                status = TestResultStatus.PASSED
                actual_output = result
                error_message = ""
                
        except Exception as e:
            status = TestResultStatus.ERROR
            actual_output = None
            error_message = str(e)
            
        execution_time_ms = (time.time() - start_time) * 1000
        
        return TestResult(
            test_id=test_case.test_id,
            test_name=test_case.test_name,
            status=status,
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            execution_time_ms=execution_time_ms,
            error_message=error_message
        )

    def execute_test_suite(self, func: Callable, test_suite: TestSuite,
                           enable_coverage: bool = False) -> TestSuite:
        if enable_coverage:
            self.coverage = coverage.Coverage()
            self.coverage.start()

        test_suite.executed_at = time.time()
        test_suite.test_results = []

        for test_case in test_suite.test_cases:
            result = self.execute_test(func, test_case)
            test_suite.test_results.append(result)

        if enable_coverage:
            self.coverage.stop()
            test_suite.coverage_report = self._generate_coverage_report(func)

        return test_suite

    def _generate_coverage_report(self, func: Callable) -> CoverageReport:
        try:
            source_file = inspect.getsourcefile(func)
            if not source_file:
                return CoverageReport()

            coverage_data = self.coverage.get_data()
            lines = coverage_data.lines(source_file) or []
            covered_lines = set(lines)
            
            source = inspect.getsource(func)
            total_lines = len(source.splitlines())
            
            return CoverageReport(
                function_coverage=100.0 if covered_lines else 0.0,
                line_coverage=len(covered_lines) / total_lines * 100 if total_lines > 0 else 0.0,
                covered_lines=covered_lines,
                uncovered_lines=set(range(1, total_lines + 1)) - covered_lines,
            )
        except Exception:
            return CoverageReport()


class TestReportGenerator:
    def __init__(self):
        pass

    def generate_text_report(self, test_suite: TestSuite) -> str:
        buffer = io.StringIO()
        
        buffer.write(f"{'='*60}\n")
        buffer.write(f"Test Suite Report: {test_suite.name}\n")
        buffer.write(f"{'='*60}\n\n")
        
        buffer.write(f"Suite ID: {test_suite.suite_id}\n")
        buffer.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_suite.created_at))}\n")
        buffer.write(f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(test_suite.executed_at))}\n")
        buffer.write(f"Total Tests: {len(test_suite.test_cases)}\n")
        
        passed = sum(1 for r in test_suite.test_results if r.status == TestResultStatus.PASSED)
        failed = sum(1 for r in test_suite.test_results if r.status == TestResultStatus.FAILED)
        errors = sum(1 for r in test_suite.test_results if r.status == TestResultStatus.ERROR)
        skipped = sum(1 for r in test_suite.test_results if r.status == TestResultStatus.SKIPPED)
        
        buffer.write(f"Passed: {passed}\n")
        buffer.write(f"Failed: {failed}\n")
        buffer.write(f"Errors: {errors}\n")
        buffer.write(f"Skipped: {skipped}\n")
        
        if test_suite.test_results:
            avg_time = sum(r.execution_time_ms for r in test_suite.test_results) / len(test_suite.test_results)
            buffer.write(f"Average Execution Time: {avg_time:.2f}ms\n")
        
        buffer.write("\n")
        
        if test_suite.coverage_report:
            buffer.write("Coverage Report:\n")
            buffer.write("-" * 40 + "\n")
            buffer.write(f"Line Coverage: {test_suite.coverage_report.line_coverage:.1f}%\n")
            buffer.write(f"Function Coverage: {test_suite.coverage_report.function_coverage:.1f}%\n")
            buffer.write(f"Branch Coverage: {test_suite.coverage_report.branch_coverage:.1f}%\n")
            
            if test_suite.coverage_report.uncovered_lines:
                buffer.write(f"Uncovered Lines: {sorted(test_suite.coverage_report.uncovered_lines)}\n")
        
        buffer.write("\nTest Results:\n")
        buffer.write("-" * 40 + "\n")
        
        for result in test_suite.test_results:
            status_icon = "✓" if result.status == TestResultStatus.PASSED else "✗"
            buffer.write(f"{status_icon} {result.test_name}: {result.status.value}\n")
            if result.status in (TestResultStatus.FAILED, TestResultStatus.ERROR):
                buffer.write(f"    Error: {result.error_message}\n")
        
        buffer.write(f"\n{'='*60}\n")
        return buffer.getvalue()

    def generate_json_report(self, test_suite: TestSuite) -> str:
        report = {
            "suite_id": test_suite.suite_id,
            "name": test_suite.name,
            "created_at": test_suite.created_at,
            "executed_at": test_suite.executed_at,
            "total_tests": len(test_suite.test_cases),
            "results_summary": {
                "passed": sum(1 for r in test_suite.test_results if r.status == TestResultStatus.PASSED),
                "failed": sum(1 for r in test_suite.test_results if r.status == TestResultStatus.FAILED),
                "errors": sum(1 for r in test_suite.test_results if r.status == TestResultStatus.ERROR),
                "skipped": sum(1 for r in test_suite.test_results if r.status == TestResultStatus.SKIPPED),
            },
            "coverage": {
                "line_coverage": test_suite.coverage_report.line_coverage if test_suite.coverage_report else 0.0,
                "function_coverage": test_suite.coverage_report.function_coverage if test_suite.coverage_report else 0.0,
                "branch_coverage": test_suite.coverage_report.branch_coverage if test_suite.coverage_report else 0.0,
            },
            "test_cases": [
                {
                    "test_id": tc.test_id,
                    "test_name": tc.test_name,
                    "strategy": tc.strategy.value,
                    "input_data": str(tc.input_data),
                }
                for tc in test_suite.test_cases
            ],
            "test_results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "execution_time_ms": r.execution_time_ms,
                    "error_message": r.error_message,
                }
                for r in test_suite.test_results
            ],
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)

    def generate_html_report(self, test_suite: TestSuite) -> str:
        passed = sum(1 for r in test_suite.test_results if r.status == TestResultStatus.PASSED)
        total = len(test_suite.test_results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {test_suite.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ flex: 1; padding: 15px; border-radius: 6px; text-align: center; }}
        .card.passed {{ background: #e8f5e9; color: #2e7d32; }}
        .card.failed {{ background: #ffebee; color: #c62828; }}
        .card.error {{ background: #fff3e0; color: #ef6c00; }}
        .card.total {{ background: #e3f2fd; color: #1565c0; }}
        .coverage {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
        .coverage-bar {{ height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }}
        .coverage-fill {{ height: 100%; background: {'#4CAF50' if percentage >= 80 else '#ff9800'}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f1f1f1; }}
        .status-passed {{ color: #4CAF50; font-weight: bold; }}
        .status-failed {{ color: #f44336; font-weight: bold; }}
        .status-error {{ color: #ff9800; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Suite Report: {test_suite.name}</h1>
        <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="card total"><strong>{total}</strong><br>Total Tests</div>
            <div class="card passed"><strong>{passed}</strong><br>Passed</div>
            <div class="card failed"><strong>{sum(1 for r in test_suite.test_results if r.status == TestResultStatus.FAILED)}</strong><br>Failed</div>
            <div class="card error"><strong>{sum(1 for r in test_suite.test_results if r.status == TestResultStatus.ERROR)}</strong><br>Errors</div>
        </div>
        
        <div class="coverage">
            <h3>Coverage Report</h3>
            <div class="coverage-bar">
                <div class="coverage-fill" style="width: {percentage}%"></div>
            </div>
            <p>Pass Rate: {percentage:.1f}%</p>
        </div>
        
        <h3>Test Results</h3>
        <table>
            <tr><th>Test Name</th><th>Status</th><th>Time (ms)</th><th>Error</th></tr>
"""
        
        for result in test_suite.test_results:
            status_class = f"status-{result.status.value.lower()}"
            error = result.error_message if result.status in (TestResultStatus.FAILED, TestResultStatus.ERROR) else "-"
            html += f"""<tr>
                <td>{result.test_name}</td>
                <td class="{status_class}">{result.status.value}</td>
                <td>{result.execution_time_ms:.2f}</td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">{error}</td>
            </tr>"""
        
        html += """</table></div></body></html>"""
        return html


class AutomatedTestSuite:
    def __init__(self):
        self.generator = TestCaseGenerator()
        self.executor = TestExecutor()
        self.report_generator = TestReportGenerator()

    def create_and_execute(self, func: Callable, strategies,
                           count_per_strategy: int = 5, enable_coverage: bool = False) -> TestSuite:
        strategy_enums = []
        for s in strategies:
            if isinstance(s, TestStrategy):
                strategy_enums.append(s)
            elif isinstance(s, str):
                try:
                    strategy_enums.append(TestStrategy[s.upper()])
                except KeyError:
                    pass
        
        if not strategy_enums:
            strategy_enums = [TestStrategy.BOUNDARY_VALUE]
            
        test_cases = self.generator.generate_test_cases(func, strategy_enums, count_per_strategy)
        
        test_suite = TestSuite(
            suite_id=f"suite_{int(time.time() * 1000)}",
            name=f"Test Suite for {func.__name__}",
            test_cases=test_cases,
            created_at=time.time()
        )
        
        return self.executor.execute_test_suite(func, test_suite, enable_coverage)

    def generate_report(self, test_suite: TestSuite, format_type: str = "text") -> str:
        if format_type == "json":
            return self.report_generator.generate_json_report(test_suite)
        elif format_type == "html":
            return self.report_generator.generate_html_report(test_suite)
        else:
            return self.report_generator.generate_text_report(test_suite)

    def get_coverage(self, test_suite: TestSuite) -> Optional[CoverageReport]:
        return test_suite.coverage_report

    def get_failed_tests(self, test_suite: TestSuite) -> List[TestResult]:
        return [r for r in test_suite.test_results if r.status == TestResultStatus.FAILED]

    def get_test_summary(self, test_suite: TestSuite) -> Dict[str, Any]:
        results = test_suite.test_results
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == TestResultStatus.PASSED),
            "failed": sum(1 for r in results if r.status == TestResultStatus.FAILED),
            "errors": sum(1 for r in results if r.status == TestResultStatus.ERROR),
            "skipped": sum(1 for r in results if r.status == TestResultStatus.SKIPPED),
            "pass_rate": sum(1 for r in results if r.status == TestResultStatus.PASSED) / len(results) * 100 if results else 0,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results) if results else 0,
        }