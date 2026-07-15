import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SymbolTable, TypeDefinition, ValidationError, ValidationResult
)
from .python_rules import PythonLanguageRules
from .c_rules import CLanguageRules
from .assembly_rules import AssemblyLanguageRules


class SemanticAnalyzer:
    def __init__(self, language: LanguageType):
        self.language = language
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table
        self.analysis_errors: List[ValidationError] = []

    def analyze(self, code: str) -> ValidationResult:
        self.symbol_table = SymbolTable()
        self.current_scope = self.symbol_table
        self.analysis_errors = []

        result = ValidationResult(language=self.language)

        if self.language == LanguageType.PYTHON:
            self._analyze_python(code, result)
        elif self.language == LanguageType.C:
            self._analyze_c(code, result)
        elif self.language == LanguageType.ASSEMBLY:
            self._analyze_assembly(code, result)

        for error in self.analysis_errors:
            if error.severity in (RuleSeverity.ERROR, RuleSeverity.CRITICAL):
                result.add_error(error)
            elif error.severity == RuleSeverity.WARNING:
                result.add_warning(error)
            else:
                result.add_info(error)

        return result

    def _analyze_python(self, code: str, result: ValidationResult):
        try:
            tree = ast.parse(code)
            self._visit_python_ast(tree, result)
        except SyntaxError as e:
            result.add_error(ValidationError(
                rule_name="syntax_error",
                language=self.language,
                category=GrammarCategory.STATEMENTS,
                message=f"Syntax error: {e.msg}",
                severity=RuleSeverity.CRITICAL,
                line=e.lineno,
                column=e.offset
            ))

    def _visit_python_ast(self, node: ast.AST, result: ValidationResult, depth: int = 0):
        if isinstance(node, ast.Module):
            for child in node.body:
                self._visit_python_ast(child, result, depth)

        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            func_name = node.name
            parent_scope = self.current_scope
            parent_scope.add_symbol(func_name, "function", line=node.lineno)
            
            self.current_scope = self.current_scope.create_child()
            
            for arg in node.args.args:
                arg_name = arg.arg
                self.current_scope.add_symbol(arg_name, "parameter", line=node.lineno)
            
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            
            self.current_scope = self.current_scope.parent

        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            self.current_scope.add_symbol(class_name, "class", line=node.lineno)
            self.current_scope = self.current_scope.create_child()
            
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            
            self.current_scope = self.current_scope.parent

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    self.current_scope.add_symbol(var_name, "variable", line=node.lineno)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.current_scope.add_symbol(elt.id, "variable", line=node.lineno)
            
            for value_node in ast.iter_child_nodes(node):
                if not isinstance(value_node, ast.Name):
                    self._visit_python_ast(value_node, result, depth)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                type_name = node.annotation.id if isinstance(node.annotation, ast.Name) else str(node.annotation)
                self.current_scope.add_symbol(var_name, "variable", data_type=type_name, line=node.lineno)
            
            if node.value:
                self._visit_python_ast(node.value, result, depth)

        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                var_name = node.id
                if not self.current_scope.has_symbol(var_name):
                    builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 
                               'bool', 'type', 'isinstance', 'abs', 'sum', 'min', 'max', 'open', 'input',
                               'str', 'repr', 'ascii', 'bin', 'hex', 'oct', 'chr', 'ord', 'bool',
                               'hash', 'id', 'type', 'enumerate', 'iter', 'next', 'slice',
                               'callable', 'compile', 'eval', 'exec', 'globals', 'locals', 'vars',
                               'dir', 'help', '__import__', 'reload', 'quit', 'exit',
                               'format', 'bytes', 'bytearray', 'memoryview', 'array',
                               'frozenset', 'complex', 'round', 'pow', 'divmod',
                               'all', 'any', 'sorted', 'reversed', 'filter', 'map', 'zip',
                               'object', 'super', 'classmethod', 'staticmethod', 'property',
                               'None', 'True', 'False', 'NotImplemented', 'Ellipsis', '__debug__',
                               '__doc__', '__name__', '__package__', '__loader__', '__spec__',
                               '__file__', '__cached__', '__builtins__'}
                    if var_name not in PythonLanguageRules.get_keywords() and var_name not in builtins:
                        self.analysis_errors.append(ValidationError(
                            rule_name="undeclared_variable",
                            language=self.language,
                            category=GrammarCategory.STATEMENTS,
                            message=f"Variable '{var_name}' is not defined",
                            severity=RuleSeverity.ERROR,
                            line=node.lineno,
                            code_snippet=var_name
                        ))

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if not self.current_scope.has_symbol(func_name):
                    builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 
                               'bool', 'type', 'isinstance', 'abs', 'sum', 'min', 'max', 'open', 'input'}
                    if func_name not in builtins and func_name not in PythonLanguageRules.get_keywords():
                        self.analysis_errors.append(ValidationError(
                            rule_name="invalid_function_call",
                            language=self.language,
                            category=GrammarCategory.FUNCTIONS,
                            message=f"Function '{func_name}' is not defined",
                            severity=RuleSeverity.ERROR,
                            line=node.lineno,
                            code_snippet=func_name
                        ))
            
            for arg in node.args:
                self._visit_python_ast(arg, result, depth)

        elif isinstance(node, ast.Return):
            func_scope = self._find_function_scope()
            if func_scope is None:
                self.analysis_errors.append(ValidationError(
                    rule_name="return_outside_function",
                    language=self.language,
                    category=GrammarCategory.STATEMENTS,
                    message="return statement outside function",
                    severity=RuleSeverity.ERROR,
                    line=node.lineno
                ))
            
            if node.value:
                self._visit_python_ast(node.value, result, depth)

        elif isinstance(node, ast.Yield):
            func_scope = self._find_function_scope()
            if func_scope is None:
                self.analysis_errors.append(ValidationError(
                    rule_name="yield_outside_function",
                    language=self.language,
                    category=GrammarCategory.STATEMENTS,
                    message="yield statement outside function",
                    severity=RuleSeverity.ERROR,
                    line=node.lineno
                ))
            
            if node.value:
                self._visit_python_ast(node.value, result, depth)

        elif isinstance(node, ast.Await):
            async_scope = self._find_async_scope()
            if async_scope is None:
                self.analysis_errors.append(ValidationError(
                    rule_name="await_outside_async",
                    language=self.language,
                    category=GrammarCategory.STATEMENTS,
                    message="await statement outside async function",
                    severity=RuleSeverity.ERROR,
                    line=node.lineno
                ))
            
            self._visit_python_ast(node.value, result, depth)

        elif isinstance(node, ast.BinOp):
            self._visit_python_ast(node.left, result, depth)
            self._visit_python_ast(node.right, result, depth)

        elif isinstance(node, ast.UnaryOp):
            self._visit_python_ast(node.operand, result, depth)

        elif isinstance(node, ast.If):
            self._visit_python_ast(node.test, result, depth)
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            for child in node.orelse:
                self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.For):
            self._visit_python_ast(node.target, result, depth)
            self._visit_python_ast(node.iter, result, depth)
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            for child in node.orelse:
                self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.While):
            self._visit_python_ast(node.test, result, depth)
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            for child in node.orelse:
                self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.Expr):
            self._visit_python_ast(node.value, result, depth)

        elif isinstance(node, ast.Pass):
            pass

        elif isinstance(node, ast.Break):
            pass

        elif isinstance(node, ast.Continue):
            pass

        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._visit_python_ast(target, result, depth)

        elif isinstance(node, ast.Try):
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)
            for handler in node.handlers:
                if handler.type:
                    self._visit_python_ast(handler.type, result, depth)
                for child in handler.body:
                    self._visit_python_ast(child, result, depth + 1)
            for child in node.orelse:
                self._visit_python_ast(child, result, depth + 1)
            for child in node.finalbody:
                self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.With):
            for item in node.items:
                self._visit_python_ast(item.context_expr, result, depth)
                if item.optional_vars:
                    self._visit_python_ast(item.optional_vars, result, depth)
            for child in node.body:
                self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.Match):
            self._visit_python_ast(node.subject, result, depth)
            for case in node.cases:
                self._visit_python_ast(case.pattern, result, depth)
                for child in case.body:
                    self._visit_python_ast(child, result, depth + 1)

        elif isinstance(node, ast.Constant):
            pass

        else:
            for child in ast.iter_child_nodes(node):
                self._visit_python_ast(child, result, depth)

    def _find_function_scope(self) -> Optional[SymbolTable]:
        scope = self.current_scope
        while scope:
            if any(s["type"] == "function" for s in scope.symbols.values()):
                return scope
            scope = scope.parent
        return None

    def _find_async_scope(self) -> Optional[SymbolTable]:
        scope = self.current_scope
        while scope:
            if any(s["type"] == "function" for s in scope.symbols.values()):
                return scope
            scope = scope.parent
        return None

    def _analyze_c(self, code: str, result: ValidationResult):
        lines = code.split('\n')
        in_function = False
        function_stack = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('#'):
                continue
            
            func_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', stripped)
            if func_match:
                ret_type = func_match.group(1).strip()
                func_name = func_match.group(3)
                self.current_scope.add_symbol(func_name, "function", data_type=ret_type, line=line_num)
                self.current_scope = self.current_scope.create_child()
                function_stack.append(func_name)
                in_function = True
                
                param_match = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
                for param_type, param_name in param_match:
                    self.current_scope.add_symbol(param_name, "parameter", data_type=param_type, line=line_num)

            if in_function and '{' in line:
                continue
            
            if in_function and '}' in line:
                if function_stack:
                    function_stack.pop()
                    self.current_scope = self.current_scope.parent
                    in_function = len(function_stack) > 0

            var_decl_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*)\s+([a-zA-Z_][a-zA-Z0-9_]*)(\s*=\s*[^;]+)?\s*;', stripped)
            if var_decl_match:
                var_type = var_decl_match.group(1).strip()
                var_name = var_decl_match.group(3)
                self.current_scope.add_symbol(var_name, "variable", data_type=var_type, line=line_num)

            stripped_no_strings = re.sub(r'"[^"]*"', '""', stripped)
            usage_match = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', stripped_no_strings)
            keywords = CLanguageRules.get_keywords()
            
            stdlib_functions = {'printf', 'scanf', 'sprintf', 'fprintf', 'malloc', 'calloc', 'realloc', 'free',
                               'strcpy', 'strncpy', 'strcmp', 'strlen', 'strcat', 'strncat',
                               'memcpy', 'memset', 'memcmp', 'memmove',
                               'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs',
                               'getchar', 'putchar', 'gets', 'puts',
                               'exit', 'abort', 'assert',
                               'abs', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp',
                               'rand', 'srand', 'time',
                               'qsort', 'bsearch', 'qsort_r'}
            
            for identifier in usage_match:
                if identifier not in keywords and not self.current_scope.has_symbol(identifier):
                    if not re.match(r'^(if|else|for|while|do|switch|case|default|return|break|continue|goto)$', identifier):
                        is_declaration = re.search(r'\b' + re.escape(identifier) + r'\s+\w+\s*[=;]', stripped)
                        if not is_declaration:
                            if identifier in stdlib_functions:
                                continue
                            self.analysis_errors.append(ValidationError(
                                rule_name="undeclared_variable",
                                language=self.language,
                                category=GrammarCategory.STATEMENTS,
                                message=f"Variable '{identifier}' is not declared",
                                severity=RuleSeverity.ERROR,
                                line=line_num,
                                code_snippet=identifier
                            ))

            if 'return' in stripped and ';' in stripped:
                ret_match = re.match(r'return\s+(.+?)\s*;', stripped)
                if ret_match and function_stack:
                    func_sym = self.symbol_table.lookup(function_stack[-1])
                    if func_sym and func_sym['data_type'] == 'void':
                        self.analysis_errors.append(ValidationError(
                            rule_name="void_function_return",
                            language=self.language,
                            category=GrammarCategory.FUNCTIONS,
                            message="void function cannot return a value",
                            severity=RuleSeverity.ERROR,
                            line=line_num
                        ))

    def _analyze_assembly(self, code: str, result: ValidationResult):
        lines = code.split('\n')
        labels = set()
        referenced_labels = set()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if not stripped or stripped.startswith(('#', ';')):
                continue

            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:', stripped)
            if label_match:
                label_name = label_match.group(1)
                if label_name in labels:
                    self.analysis_errors.append(ValidationError(
                        rule_name="duplicate_label",
                        language=self.language,
                        category=GrammarCategory.KEYWORDS,
                        message=f"Label '{label_name}' defined multiple times",
                        severity=RuleSeverity.ERROR,
                        line=line_num,
                        code_snippet=label_name
                    ))
                else:
                    labels.add(label_name)
                continue

            tokens = stripped.split()
            if not tokens:
                continue

            opcode = tokens[0].lower()
            
            is_valid_instruction = opcode in AssemblyLanguageRules.KEYWORDS['pseudo_instructions']
            is_valid_directive = opcode in AssemblyLanguageRules.KEYWORDS['directives']
            is_valid_register = opcode in AssemblyLanguageRules.KEYWORDS['registers']
            is_dot_directive = opcode.startswith('.')
            
            if not (is_valid_instruction or is_valid_directive or is_valid_register or is_dot_directive):
                self.analysis_errors.append(ValidationError(
                    rule_name="invalid_instruction",
                    language=self.language,
                    category=GrammarCategory.EXPRESSIONS,
                    message=f"Invalid instruction '{opcode}'",
                    severity=RuleSeverity.ERROR,
                    line=line_num,
                    code_snippet=opcode
                ))

            for operand in tokens[1:]:
                operand = operand.replace(',', '').replace('[', '').replace(']', '')
                
                if operand in AssemblyLanguageRules.KEYWORDS['registers']:
                    continue
                
                if operand in AssemblyLanguageRules.KEYWORDS['sse_registers']:
                    continue
                
                if operand in AssemblyLanguageRules.KEYWORDS['avx_registers']:
                    continue
                
                if operand in AssemblyLanguageRules.KEYWORDS['fpu_registers']:
                    continue
                
                if operand in AssemblyLanguageRules.KEYWORDS['mmx_registers']:
                    continue
                
                if operand.startswith(('$', '0x')) or operand.isdigit():
                    continue
                
                if operand in AssemblyLanguageRules.KEYWORDS['segment_registers']:
                    continue
                
                jump_ops = {'jmp', 'je', 'jne', 'jz', 'jnz', 'jb', 'jnb', 'ja', 'jna',
                            'jl', 'jge', 'jg', 'jle', 'jo', 'jno', 'js', 'jns',
                            'jcxz', 'jecxz', 'loop', 'loope', 'loopne', 'call'}
                
                if opcode in jump_ops:
                    referenced_labels.add(operand)

        for label in referenced_labels:
            if label not in labels:
                self.analysis_errors.append(ValidationError(
                    rule_name="undefined_label",
                    language=self.language,
                    category=GrammarCategory.KEYWORDS,
                    message=f"Label '{label}' referenced but not defined",
                    severity=RuleSeverity.ERROR,
                    code_snippet=label
                ))

    def evaluate_expression(self, expr: str, context: Dict[str, Any] = None) -> Any:
        if self.language == LanguageType.PYTHON:
            return self._evaluate_python_expression(expr, context)
        return None

    def _evaluate_python_expression(self, expr: str, context: Dict[str, Any] = None) -> Any:
        try:
            tree = ast.parse(expr, mode='eval')
            local_vars = context or {}
            return eval(compile(tree, '<expr>', 'eval'), {}, local_vars)
        except Exception:
            return None

    def get_symbol_table(self) -> SymbolTable:
        return self.symbol_table