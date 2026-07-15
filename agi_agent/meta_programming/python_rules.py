from typing import List, Union, Set
from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SyntaxRule, SemanticRule, TypeDefinition
)


class PythonLanguageRules:
    KEYWORDS = {
        "control_flow": ["if", "elif", "else", "for", "while", "break", "continue", "return", "pass", "raise", "try", "except", "finally", "with", "match", "case"],
        "data_types": ["True", "False", "None", "int", "float", "str", "bool", "list", "tuple", "dict", "set", "type"],
        "modular": ["import", "from", "as", "def", "class", "lambda", "yield", "async", "await", "del", "global", "nonlocal"],
        "operators": ["and", "or", "not", "in", "is"]
    }

    DATA_TYPES = [
        TypeDefinition("int", size=28, base_type="numeric"),
        TypeDefinition("float", size=24, base_type="numeric"),
        TypeDefinition("str", size=49, base_type="sequence"),
        TypeDefinition("bool", size=28, base_type="logical"),
        TypeDefinition("list", size=48, base_type="sequence"),
        TypeDefinition("tuple", size=40, base_type="sequence"),
        TypeDefinition("dict", size=48, base_type="mapping"),
        TypeDefinition("set", size=48, base_type="collection"),
        TypeDefinition("None", size=16, base_type="special"),
        TypeDefinition("complex", size=32, base_type="numeric"),
        TypeDefinition("bytes", size=33, base_type="sequence"),
        TypeDefinition("bytearray", size=56, base_type="sequence")
    ]

    OPERATORS = {
        "arithmetic": ["+", "-", "*", "/", "//", "%", "**"],
        "comparison": ["==", "!=", "<", ">", "<=", ">="],
        "logical": ["and", "or", "not"],
        "bitwise": ["&", "|", "^", "~", "<<", ">>"],
        "assignment": ["=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>="],
        "membership": ["in", "not in"],
        "identity": ["is", "is not"]
    }

    SYNTAX_RULES = [
        SyntaxRule(
            name="function_definition",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(:|\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:)',
            description="函数定义必须以def关键字开头，后跟函数名、参数列表和冒号",
            example="def add(a: int, b: int) -> int:\n    return a + b",
            error_message="函数定义格式错误：缺少def关键字、函数名或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="class_definition",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^class\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\([^)]*\))?\s*:',
            description="类定义必须以class关键字开头，后跟类名和冒号",
            example="class MyClass:\n    def __init__(self):\n        pass",
            error_message="类定义格式错误：缺少class关键字、类名或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="if_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^if\s+[^:]+:',
            description="if语句必须以if关键字开头，后跟条件表达式和冒号",
            example="if x > 0:\n    print('positive')",
            error_message="if语句格式错误：缺少条件表达式或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="for_loop",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^for\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+[^:]+:',
            description="for循环必须包含for、变量名、in关键字、可迭代对象和冒号",
            example="for item in my_list:\n    print(item)",
            error_message="for循环格式错误：缺少in关键字或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="while_loop",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^while\s+[^:]+:',
            description="while循环必须以while关键字开头，后跟条件表达式和冒号",
            example="while x < 10:\n    x += 1",
            error_message="while循环格式错误：缺少条件表达式或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="return_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^return(\s+[^;]+)?$',
            description="return语句可以返回一个表达式或无返回值",
            example="return result\nreturn",
            error_message="return语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="import_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MODULES,
            pattern=r'^(import\s+[a-zA-Z_][a-zA-Z0-9_.]*(\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?)|(from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*(,\s*[a-zA-Z_][a-zA-Z0-9_]*)*(\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?)$',
            description="import语句用于导入模块",
            example="import math\nfrom utils import helper as h",
            error_message="import语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="variable_assignment",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*:\s*[a-zA-Z_][a-zA-Z0-9_.]*)?\s*=\s*.*$',
            description="变量赋值必须以有效标识符开头，可以包含类型注解",
            example="x = 10\nname: str = 'test'",
            error_message="变量赋值格式错误：无效的变量名或缺少等号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="lambda_expression",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^lambda\s+[^:]+:\s*[^:]+$',
            description="lambda表达式必须以lambda关键字开头，后跟参数、冒号和表达式",
            example="lambda x, y: x + y",
            error_message="lambda表达式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="try_except",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^try\s*:\s*\n.*\n^except(\s+[a-zA-Z_][a-zA-Z0-9_.]*)?\s*:',
            description="try-except块必须包含try和except部分",
            example="try:\n    risky_op()\nexcept ValueError:\n    handle_error()",
            error_message="try-except块格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="with_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^with\s+[^:]+(:\s*[a-zA-Z_][a-zA-Z0-9_]*)?\s*:',
            description="with语句用于资源管理",
            example="with open('file.txt') as f:\n    content = f.read()",
            error_message="with语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="async_def",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^async\s+def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(:|\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:)',
            description="异步函数定义必须以async def开头",
            example="async def fetch_data():\n    return await api_call()",
            error_message="异步函数定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="match_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^match\s+[^:]+:',
            description="match语句必须以match关键字开头，后跟表达式和冒号",
            example="match value:\n    case 1:\n        pass",
            error_message="match语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="case_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^case\s+[^:]+:',
            description="case语句必须以case关键字开头，后跟模式和冒号",
            example="case 1 | 2:\n    pass",
            error_message="case语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="identifier",
            language=LanguageType.PYTHON,
            category=GrammarCategory.KEYWORDS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$',
            description="标识符必须以字母或下划线开头，只能包含字母、数字和下划线",
            example="my_variable, _private, MAX_SIZE",
            error_message="无效的标识符：必须以字母或下划线开头",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="list_comprehension",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^\[.*\s+for\s+.*\s+in\s+.*\]$',
            description="列表推导式用于简洁地创建列表",
            example="[x**2 for x in range(10)]\n[x for x in nums if x > 0]",
            error_message="列表推导式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="dict_comprehension",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^\{.*\s*:\s*.*\s+for\s+.*\s+in\s+.*\}$',
            description="字典推导式用于简洁地创建字典",
            example="{k: v for k, v in items()}\n{x: x**2 for x in range(5)}",
            error_message="字典推导式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="set_comprehension",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^\{.*\s+for\s+.*\s+in\s+.*\}$',
            description="集合推导式用于简洁地创建集合",
            example="{x**2 for x in range(10)}\n{x for x in nums if x % 2 == 0}",
            error_message="集合推导式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="generator_expression",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^\(.*\s+for\s+.*\s+in\s+.*\)$',
            description="生成器表达式用于惰性计算，节省内存",
            example="sum(x**2 for x in range(10))\nmax(x for x in nums if x > 0)",
            error_message="生成器表达式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="decorator",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^@[a-zA-Z_][a-zA-Z0-9_.]*(\(.*\))?$',
            description="装饰器用于修改函数或类的行为，以@符号开头",
            example="@staticmethod\n@decorator(arg)\n@module.decorator",
            error_message="装饰器格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="yield_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^yield(\s+.*)?$',
            description="yield语句用于生成器函数，产生一个值并暂停执行",
            example="yield value\nyield from iterable\nyield",
            error_message="yield语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="f_string",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^f["\'].*\{.*\}.*["\']$',
            description="f-string用于格式化字符串，支持在字符串中嵌入表达式",
            example="f'Hello, {name}!'\nf'Result: {x + y:.2f}'",
            error_message="f-string格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="walrus_operator",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'.*:=\s*.*',
            description="海象运算符(:=)用于在表达式中赋值（Python 3.8+）",
            example="if (n := len(a)) > 10:\nwhile (line := f.readline()):",
            error_message="海象运算符使用格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="unpacking_operator",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^\*.*$',
            description="解包运算符(*)用于解包可迭代对象，(**)用于解包字典",
            example="first, *rest = items\nfunc(*args, **kwargs)",
            error_message="解包运算符使用格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="ternary_expression",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'.*\s+if\s+.*\s+else\s+.*',
            description="三元表达式用于简洁的条件赋值",
            example="x = a if condition else b\nresult = 'yes' if flag else 'no'",
            error_message="三元表达式格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="raise_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^raise(\s+[a-zA-Z_][a-zA-Z0-9_.]*(\(.*\))?)?(\s+from\s+.*)?$',
            description="raise语句用于抛出异常",
            example="raise ValueError('invalid')\nraise\nraise NewError from original_error",
            error_message="raise语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="assert_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^assert\s+.*(,\s*.*)?$',
            description="assert语句用于调试时的断言检查",
            example="assert x > 0\nassert condition, 'error message'",
            error_message="assert语句格式错误",
            severity=RuleSeverity.WARNING
        ),
        SyntaxRule(
            name="global_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            pattern=r'^global\s+[a-zA-Z_][a-zA-Z0-9_]*(,\s*[a-zA-Z_][a-zA-Z0-9_]*)*$',
            description="global语句声明变量为全局变量",
            example="global x\nglobal count, total",
            error_message="global语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="nonlocal_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            pattern=r'^nonlocal\s+[a-zA-Z_][a-zA-Z0-9_]*(,\s*[a-zA-Z_][a-zA-Z0-9_]*)*$',
            description="nonlocal语句声明变量为外层嵌套函数的变量",
            example="nonlocal x\nonlocal count, total",
            error_message="nonlocal语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="del_statement",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^del\s+.*$',
            description="del语句用于删除变量或项",
            example="del x\ndel dict[key]\ndel list[0]",
            error_message="del语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="class_method_decorator",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^@(classmethod|staticmethod|property)$',
            description="类方法装饰器包括classmethod、staticmethod和property",
            example="@classmethod\n@staticmethod\n@property",
            error_message="类方法装饰器格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="context_manager",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^with\s+.*\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:',
            description="上下文管理器用于资源的自动获取和释放",
            example="with open('file.txt') as f:\n    content = f.read()",
            error_message="上下文管理器格式错误",
            severity=RuleSeverity.ERROR
        )
    ]

    SEMANTIC_RULES = [
        SemanticRule(
            name="undeclared_variable",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            description="使用未定义的变量会导致NameError",
            validation_logic="检查符号表中是否存在该变量",
            error_message="变量未定义",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="invalid_function_call",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="调用不存在的函数或方法会导致NameError或AttributeError",
            validation_logic="检查函数名是否在当前作用域中定义",
            error_message="函数未定义",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="wrong_argument_count",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="函数调用时参数数量必须匹配",
            validation_logic="比较实际参数数量与函数定义的参数数量",
            error_message="参数数量不匹配",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="invalid_attribute_access",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="访问对象不存在的属性会导致AttributeError",
            validation_logic="检查对象是否拥有该属性",
            error_message="属性不存在",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="type_mismatch",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="操作数类型不兼容会导致TypeError",
            validation_logic="检查操作数类型是否兼容",
            error_message="类型不兼容",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="division_by_zero",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="除以零会导致ZeroDivisionError",
            validation_logic="检查除数是否为零",
            error_message="除以零",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="return_outside_function",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            description="return语句只能在函数内部使用",
            validation_logic="检查return语句是否在函数作用域内",
            error_message="return语句不能在函数外部使用",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="yield_outside_function",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            description="yield语句只能在函数内部使用",
            validation_logic="检查yield语句是否在函数作用域内",
            error_message="yield语句不能在函数外部使用",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="await_outside_async",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            description="await只能在异步函数内部使用",
            validation_logic="检查await语句是否在async函数作用域内",
            error_message="await只能在异步函数内部使用",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="global_declaration",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            description="global声明必须在函数内部使用，且在变量使用之前",
            validation_logic="检查global声明的位置和时机",
            error_message="global声明使用不正确",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="nonlocal_declaration",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            description="nonlocal声明必须在嵌套函数内部使用",
            validation_logic="检查nonlocal声明是否在嵌套函数内",
            error_message="nonlocal声明只能在嵌套函数中使用",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="mutating_immutable",
            language=LanguageType.PYTHON,
            category=GrammarCategory.DATA_TYPES,
            description="尝试修改不可变对象会导致TypeError",
            validation_logic="检查是否尝试修改tuple、str、int等不可变类型",
            error_message="不可变对象无法修改",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="index_out_of_range",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="访问超出序列长度的索引会导致IndexError",
            validation_logic="检查索引是否在序列长度范围内",
            error_message="索引越界",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="key_not_found",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="访问字典中不存在的键会导致KeyError",
            validation_logic="检查键是否存在于字典中",
            error_message="键不存在",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="attribute_error",
            language=LanguageType.PYTHON,
            category=GrammarCategory.EXPRESSIONS,
            description="访问对象不存在的属性或方法会导致AttributeError",
            validation_logic="检查对象是否具有指定的属性",
            error_message="属性不存在",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="type_inference",
            language=LanguageType.PYTHON,
            category=GrammarCategory.DATA_TYPES,
            description="Python支持动态类型推断，变量类型由赋值决定",
            validation_logic="根据赋值表达式推断变量类型",
            error_message="无法推断类型",
            severity=RuleSeverity.INFO
        ),
        SemanticRule(
            name="function_signature_match",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="函数调用的参数类型应与函数签名兼容",
            validation_logic="检查参数类型是否与函数定义的类型注解兼容",
            error_message="函数参数类型不匹配",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="return_type_match",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="函数返回值类型应与返回类型注解兼容",
            validation_logic="检查return语句的表达式类型是否与返回类型注解兼容",
            error_message="返回值类型不匹配",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="variable_shadowing",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            description="内层作用域变量会遮蔽外层作用域的同名变量",
            validation_logic="检查是否存在变量遮蔽现象",
            error_message="变量遮蔽外层作用域",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="unused_variable",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MEMORY,
            description="定义但未使用的变量可能是代码异味",
            validation_logic="检查变量是否在定义后被使用",
            error_message="变量定义后未使用",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="import_not_used",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MODULES,
            description="导入但未使用的模块应被移除",
            validation_logic="检查导入的模块是否在代码中被使用",
            error_message="导入未使用",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="circular_import",
            language=LanguageType.PYTHON,
            category=GrammarCategory.MODULES,
            description="循环导入会导致运行时错误",
            validation_logic="检测模块间的循环导入关系",
            error_message="循环导入",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="generator_close",
            language=LanguageType.PYTHON,
            category=GrammarCategory.STATEMENTS,
            description="生成器应正确关闭以避免资源泄漏",
            validation_logic="检查生成器是否在使用后被正确关闭",
            error_message="生成器未正确关闭",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="context_manager_usage",
            language=LanguageType.PYTHON,
            category=GrammarCategory.CONTROL_FLOW,
            description="资源对象应使用上下文管理器确保正确释放",
            validation_logic="检查文件、网络连接等资源是否使用with语句",
            error_message="资源未使用上下文管理器",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="decorator_application",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="装饰器应用应保持被装饰函数的元数据",
            validation_logic="检查装饰器是否使用functools.wraps",
            error_message="装饰器未保留函数元数据",
            severity=RuleSeverity.INFO
        ),
        SemanticRule(
            name="mutable_default_argument",
            language=LanguageType.PYTHON,
            category=GrammarCategory.FUNCTIONS,
            description="使用可变对象作为默认参数会导致意外行为",
            validation_logic="检查函数默认参数是否为可变类型",
            error_message="可变默认参数",
            severity=RuleSeverity.WARNING
        )
    ]

    @classmethod
    def get_all_rules(cls) -> List[Union[SyntaxRule, SemanticRule]]:
        return cls.SYNTAX_RULES + cls.SEMANTIC_RULES

    @classmethod
    def get_syntax_rules(cls) -> List[SyntaxRule]:
        return cls.SYNTAX_RULES

    @classmethod
    def get_semantic_rules(cls) -> List[SemanticRule]:
        return cls.SEMANTIC_RULES

    @classmethod
    def get_rules_by_category(cls, category: GrammarCategory) -> List[Union[SyntaxRule, SemanticRule]]:
        return [rule for rule in cls.get_all_rules() if rule.category == category]

    @classmethod
    def get_keywords(cls) -> Set[str]:
        keywords = set()
        for category in cls.KEYWORDS.values():
            keywords.update(category)
        return keywords

    @classmethod
    def get_data_types(cls) -> List[TypeDefinition]:
        return cls.DATA_TYPES