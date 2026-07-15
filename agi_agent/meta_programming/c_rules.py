from typing import List
from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SyntaxRule, SemanticRule, TypeDefinition
)


class CLanguageRules:
    KEYWORDS = {
        "data_types": ["void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "bool"],
        "control_flow": ["if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue", "return", "goto"],
        "storage": ["auto", "register", "static", "extern", "const", "volatile", "restrict"],
        "structures": ["struct", "union", "enum", "typedef"],
        "other": ["sizeof", "_Alignof"]
    }

    DATA_TYPES = [
        TypeDefinition("void", size=0, base_type="special"),
        TypeDefinition("char", size=1, base_type="integer"),
        TypeDefinition("unsigned char", size=1, base_type="integer"),
        TypeDefinition("signed char", size=1, base_type="integer"),
        TypeDefinition("short", size=2, base_type="integer"),
        TypeDefinition("unsigned short", size=2, base_type="integer"),
        TypeDefinition("int", size=4, base_type="integer"),
        TypeDefinition("unsigned int", size=4, base_type="integer"),
        TypeDefinition("long", size=4, base_type="integer"),
        TypeDefinition("unsigned long", size=4, base_type="integer"),
        TypeDefinition("long long", size=8, base_type="integer"),
        TypeDefinition("unsigned long long", size=8, base_type="integer"),
        TypeDefinition("float", size=4, base_type="floating"),
        TypeDefinition("double", size=8, base_type="floating"),
        TypeDefinition("long double", size=16, base_type="floating"),
        TypeDefinition("bool", size=1, base_type="logical")
    ]

    OPERATORS = {
        "arithmetic": ["+", "-", "*", "/", "%", "++", "--"],
        "comparison": ["==", "!=", "<", ">", "<=", ">="],
        "logical": ["&&", "||", "!"],
        "bitwise": ["&", "|", "^", "~", "<<", ">>"],
        "assignment": ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="],
        "pointer": ["*", "&", "->", "."],
        "memory": ["sizeof", "_Alignof"]
    }

    SYNTAX_RULES = [
        SyntaxRule(
            name="function_definition",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(\{|\s*;)',
            description="函数定义必须包含返回类型、函数名、参数列表和函数体",
            example="int add(int a, int b) {\n    return a + b;\n}",
            error_message="函数定义格式错误：缺少返回类型、函数名或参数列表",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="function_declaration",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*;',
            description="函数声明以分号结尾",
            example="int add(int a, int b);",
            error_message="函数声明格式错误：缺少分号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="variable_declaration",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*(\s+\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*=\s*[^;]+)?\s*;',
            description="变量声明必须包含类型和变量名，可以包含初始化",
            example="int x;\nint y = 10;\nchar *str = \"hello\";",
            error_message="变量声明格式错误：缺少类型或分号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="struct_definition",
            language=LanguageType.C,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^struct(\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s*\{[^}]+\}\s*([a-zA-Z_][a-zA-Z0-9_]*(\s*=\s*\{[^}]+\})?)?\s*;',
            description="结构体定义使用struct关键字",
            example="struct Point {\n    int x;\n    int y;\n};",
            error_message="结构体定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="union_definition",
            language=LanguageType.C,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^union(\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s*\{[^}]+\}\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*;',
            description="联合体定义使用union关键字",
            example="union Data {\n    int i;\n    float f;\n};",
            error_message="联合体定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="enum_definition",
            language=LanguageType.C,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^enum(\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s*\{[^}]+\}\s*([a-zA-Z_][a-zA-Z0-9_]*(\s*=\s*\d+)?)?\s*;',
            description="枚举定义使用enum关键字",
            example="enum Color { RED, GREEN, BLUE };",
            error_message="枚举定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="typedef",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^typedef\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;',
            description="typedef用于定义类型别名",
            example="typedef int MyInt;",
            error_message="typedef格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="if_statement",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^if\s*\([^)]+\)\s*(\{|\S)',
            description="if语句条件必须用括号包围",
            example="if (x > 0) {\n    printf(\"positive\");\n}",
            error_message="if语句格式错误：缺少括号或条件",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="for_loop",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^for\s*\([^;]+;[^;]+;[^)]+\)\s*(\{|\S)',
            description="for循环包含初始化、条件和增量表达式",
            example="for (int i = 0; i < 10; i++) { ... }",
            error_message="for循环格式错误：缺少分号或括号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="while_loop",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^while\s*\([^)]+\)\s*(\{|\S)',
            description="while循环条件必须用括号包围",
            example="while (x < 10) { x++; }",
            error_message="while循环格式错误：缺少括号或条件",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="do_while_loop",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^do\s*\{[^}]+\}\s*while\s*\([^)]+\)\s*;',
            description="do-while循环先执行后判断",
            example="do { x++; } while (x < 10);",
            error_message="do-while循环格式错误：缺少while或分号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="switch_statement",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^switch\s*\([^)]+\)\s*\{[^}]+\}',
            description="switch语句必须包含case和break",
            example="switch (x) { case 1: break; }",
            error_message="switch语句格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="case_statement",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^case\s+[^:]+:',
            description="case语句必须以常量表达式和冒号结尾",
            example="case 1:\n    break;",
            error_message="case语句格式错误：缺少常量表达式或冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="return_statement",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^return(\s+[^;]+)?\s*;',
            description="return语句必须以分号结尾",
            example="return 0;\nreturn;",
            error_message="return语句格式错误：缺少分号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="pointer_declaration",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*[a-zA-Z_][a-zA-Z0-9_]*(\s*=\s*[^;]+)?\s*;',
            description="指针声明使用星号",
            example="int *ptr;\nchar *str = NULL;",
            error_message="指针声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="array_declaration",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\[\d*\](\s*=\s*\{[^}]+\})?\s*;',
            description="数组声明使用方括号",
            example="int arr[10];\nchar str[] = \"hello\";",
            error_message="数组声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="preprocessor_directive",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            pattern=r'^#\s*(define|include|if|ifdef|ifndef|else|elif|endif|pragma|error|warning)\s+.*$',
            description="预处理指令以#开头",
            example="#include <stdio.h>\n#define MAX 100",
            error_message="预处理指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="identifier",
            language=LanguageType.C,
            category=GrammarCategory.KEYWORDS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$',
            description="标识符必须以字母或下划线开头",
            example="my_variable, _private, MAX_SIZE",
            error_message="无效的标识符",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="function_pointer",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*\*[a-zA-Z_][a-zA-Z0-9_]*\s*\)\s*\([^)]*\)\s*;',
            description="函数指针声明用于指向函数的指针",
            example="int (*func_ptr)(int, int);\nvoid (*callback)(void*);",
            error_message="函数指针声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="bit_field",
            language=LanguageType.C,
            category=GrammarCategory.STRUCTURES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*\d+\s*;',
            description="位域用于结构体中指定成员占用的位数",
            example="struct Flags {\n    unsigned int flag1 : 1;\n    unsigned int flag2 : 2;\n};",
            error_message="位域声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="macro_function",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            pattern=r'^#\s*define\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s+.*$',
            description="函数式宏定义接受参数",
            example="#define MAX(a, b) ((a) > (b) ? (a) : (b))",
            error_message="函数式宏定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="inline_function",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^inline\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*\{',
            description="内联函数建议编译器内联展开",
            example="inline int add(int a, int b) { return a + b; }",
            error_message="内联函数定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="static_function",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^static\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(\{|;)',
            description="静态函数只能在当前文件中访问",
            example="static void helper(void) { ... }",
            error_message="静态函数定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="extern_declaration",
            language=LanguageType.C,
            category=GrammarCategory.MEMORY,
            pattern=r'^extern\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;',
            description="extern声明变量或函数在其他文件中定义",
            example="extern int global_var;\nextern void func(void);",
            error_message="extern声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="const_variable",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^const\s+[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^;]+\s*;',
            description="const变量声明后不可修改",
            example="const int MAX = 100;\nconst char *NAME = \"test\";",
            error_message="const变量声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="conditional_compilation",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            pattern=r'^#\s*(ifdef|ifndef|if)\s+[a-zA-Z_][a-zA-Z0-9_]*$',
            description="条件编译用于根据条件包含或排除代码",
            example="#ifdef DEBUG\n#ifndef NDEBUG\n#if defined(WIN32)",
            error_message="条件编译指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="include_directive",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            pattern=r'^#\s*include\s+[<"][a-zA-Z0-9_./\\]+[>"]$',
            description="#include指令用于包含头文件",
            example="#include <stdio.h>\n#include \"myheader.h\"",
            error_message="#include指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="struct_typedef",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^typedef\s+struct\s*\{[^}]+\}\s*[a-zA-Z_][a-zA-Z0-9_]*\s*;',
            description="typedef与struct结合定义匿名结构体类型",
            example="typedef struct {\n    int x;\n    int y;\n} Point;",
            error_message="typedef struct格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="enum_typedef",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^typedef\s+enum\s*\{[^}]+\}\s*[a-zA-Z_][a-zA-Z0-9_]*\s*;',
            description="typedef与enum结合定义匿名枚举类型",
            example="typedef enum { RED, GREEN, BLUE } Color;",
            error_message="typedef enum格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="pointer_to_pointer",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\*\*\s*[a-zA-Z_][a-zA-Z0-9_]*(\s*=\s*[^;]+)?\s*;',
            description="指向指针的指针（二级指针）",
            example="int **ptr_ptr;\nchar **argv;",
            error_message="二级指针声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="array_of_pointers",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\[\d*\]\s*;',
            description="指针数组，数组元素为指针类型",
            example="char *names[] = {\"a\", \"b\"};\nint *ptrs[10];",
            error_message="指针数组声明格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="ternary_operator",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'.*\?.*:.*',
            description="三元运算符用于条件表达式",
            example="int max = a > b ? a : b;",
            error_message="三元运算符使用格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="compound_literal",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'\([a-zA-Z_][a-zA-Z0-9_.\s]*\)\s*\{[^}]+\}',
            description="复合字面量用于创建匿名结构体/数组（C99）",
            example="(Point){1, 2}\n(int[]){1, 2, 3}",
            error_message="复合字面量格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="designated_initializer",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'\.\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=',
            description="指定初始化器用于按名称初始化结构体成员（C99）",
            example="Point p = {.x = 1, .y = 2};",
            error_message="指定初始化器格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="for_each_macro",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            pattern=r'^#\s*define\s+[a-zA-Z_][a-zA-Z0-9_]*\s+.*\\$',
            description="多行宏定义使用反斜杠续行",
            example="#define MULTI_LINE(x) \\\n    do { \\\n        process(x); \\\n    } while(0)",
            error_message="多行宏定义格式错误",
            severity=RuleSeverity.ERROR
        )
    ]

    SEMANTIC_RULES = [
        SemanticRule(
            name="undeclared_variable",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            description="使用未声明的变量会导致编译错误",
            validation_logic="检查符号表中是否存在该变量",
            error_message="变量未声明",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="type_mismatch_assignment",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            description="赋值操作两边类型必须兼容",
            validation_logic="检查赋值操作数类型是否兼容",
            error_message="类型不兼容的赋值",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="return_type_mismatch",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="返回值类型必须与函数声明的返回类型匹配",
            validation_logic="比较返回表达式类型与函数返回类型",
            error_message="返回类型不匹配",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="void_function_return",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="void函数不能返回值",
            validation_logic="检查void函数中是否有return表达式",
            error_message="void函数不能返回值",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="function_call_mismatch",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="函数调用参数数量和类型必须与声明匹配",
            validation_logic="比较实际参数与函数参数类型和数量",
            error_message="函数调用参数不匹配",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="implicit_int_deprecated",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            description="隐式int类型声明在C99中已弃用",
            validation_logic="检查函数声明是否缺少返回类型",
            error_message="隐式int声明已弃用，需要显式指定返回类型",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="division_by_zero",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="除以零会导致未定义行为",
            validation_logic="检查除数是否为零常量",
            error_message="除以零",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="pointer_arithmetic",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="指针只能与整数进行加减运算",
            validation_logic="检查指针操作数类型",
            error_message="无效的指针运算",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="null_pointer_dereference",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="解引用空指针会导致未定义行为",
            validation_logic="检查指针是否可能为空",
            error_message="可能的空指针解引用",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="array_index_out_of_bounds",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="数组索引超出范围会导致未定义行为",
            validation_logic="检查数组索引是否在有效范围内",
            error_message="数组索引可能越界",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="type_conversion_warning",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="隐式类型转换可能导致精度丢失",
            validation_logic="检查是否存在可能导致精度丢失的隐式转换",
            error_message="隐式类型转换可能导致精度丢失",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="unused_variable",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            description="声明但未使用的变量会产生警告",
            validation_logic="检查变量是否被引用",
            error_message="变量声明但未使用",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="missing_prototype",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="函数在使用前未声明会产生警告",
            validation_logic="检查函数调用前是否有声明",
            error_message="函数缺少原型声明",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="const_correctness",
            language=LanguageType.C,
            category=GrammarCategory.DATA_TYPES,
            description="尝试修改const变量会导致错误",
            validation_logic="检查是否尝试修改const限定的变量",
            error_message="不能修改const变量",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="memory_leak",
            language=LanguageType.C,
            category=GrammarCategory.MEMORY,
            description="动态分配的内存应在使用后释放",
            validation_logic="检查malloc/calloc/realloc分配的内存是否有对应的free",
            error_message="可能的内存泄漏",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="double_free",
            language=LanguageType.C,
            category=GrammarCategory.MEMORY,
            description="重复释放同一块内存会导致未定义行为",
            validation_logic="检查是否对同一指针多次调用free",
            error_message="重复释放内存",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="use_after_free",
            language=LanguageType.C,
            category=GrammarCategory.MEMORY,
            description="释放后使用内存会导致未定义行为",
            validation_logic="检查是否在free后继续使用指针",
            error_message="释放后使用内存",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="buffer_overflow",
            language=LanguageType.C,
            category=GrammarCategory.MEMORY,
            description="数组或缓冲区写入可能超出边界",
            validation_logic="检查写入操作是否可能超出数组大小",
            error_message="可能的缓冲区溢出",
            severity=RuleSeverity.CRITICAL
        ),
        SemanticRule(
            name="format_string_vulnerability",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="printf等函数的格式字符串不应来自用户输入",
            validation_logic="检查格式字符串是否为常量",
            error_message="格式字符串漏洞风险",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="uninitialized_variable",
            language=LanguageType.C,
            category=GrammarCategory.STATEMENTS,
            description="使用未初始化的变量会导致未定义行为",
            validation_logic="检查变量在使用前是否已赋值",
            error_message="变量可能未初始化",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="incompatible_pointer_types",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="不同类型的指针之间赋值需要显式转换",
            validation_logic="检查指针赋值的类型兼容性",
            error_message="指针类型不兼容",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="sizeof_pointer_mistake",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="对指针使用sizeof可能得到指针大小而非数组大小",
            validation_logic="检查sizeof是否应用于已退化为指针的数组",
            error_message="sizeof可能返回指针大小而非数组大小",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="switch_fallthrough",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            description="switch语句中case穿透可能是无意的",
            validation_logic="检查case语句是否缺少break",
            error_message="case穿透可能是无意的",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="assignment_in_condition",
            language=LanguageType.C,
            category=GrammarCategory.CONTROL_FLOW,
            description="条件语句中的赋值可能是错误的",
            validation_logic="检查if/while条件中是否使用了赋值运算符",
            error_message="条件中使用赋值运算符，可能是比较的笔误",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="macro_side_effects",
            language=LanguageType.C,
            category=GrammarCategory.MODULES,
            description="带副作用的表达式作为宏参数可能导致多次计算",
            validation_logic="检查宏参数是否包含自增、自减、函数调用等副作用",
            error_message="宏参数可能有副作用，导致多次计算",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="recursion_depth",
            language=LanguageType.C,
            category=GrammarCategory.FUNCTIONS,
            description="过深的递归可能导致栈溢出",
            validation_logic="分析递归函数的最大深度",
            error_message="递归可能过深，存在栈溢出风险",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="strict_aliasing_violation",
            language=LanguageType.C,
            category=GrammarCategory.EXPRESSIONS,
            description="违反严格别名规则会导致未定义行为",
            validation_logic="检查是否通过不兼容类型的指针访问内存",
            error_message="可能违反严格别名规则",
            severity=RuleSeverity.WARNING
        )
    ]

    @classmethod
    def get_all_rules(cls) -> list:
        return cls.SYNTAX_RULES + cls.SEMANTIC_RULES

    @classmethod
    def get_syntax_rules(cls) -> list:
        return cls.SYNTAX_RULES

    @classmethod
    def get_semantic_rules(cls) -> list:
        return cls.SEMANTIC_RULES

    @classmethod
    def get_rules_by_category(cls, category: GrammarCategory) -> list:
        return [rule for rule in cls.get_all_rules() if rule.category == category]

    @classmethod
    def get_keywords(cls) -> set:
        keywords = set()
        for category in cls.KEYWORDS.values():
            keywords.update(category)
        return keywords

    @classmethod
    def get_data_types(cls) -> list:
        return cls.DATA_TYPES