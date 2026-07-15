from typing import Any, Dict, List

from .language_rules import LanguageType, GrammarCategory, RuleSeverity
from .python_rules import PythonLanguageRules
from .c_rules import CLanguageRules
from .assembly_rules import AssemblyLanguageRules
from .rule_validator import RuleQueryEngine


class RuleDocumentationGenerator:
    def __init__(self):
        self.query_engine = RuleQueryEngine()

    def generate_full_documentation(self) -> str:
        docs = []
        docs.append("# 元编程语言规则文档")
        docs.append("")
        docs.append("本文档包含 Python、C 和汇编语言的完整语法规则与语义定义。")
        docs.append("")
        
        docs.append("## 目录")
        docs.append("- [1. 语言概览](#1-语言概览)")
        docs.append("- [2. Python 语言规则](#2-python-语言规则)")
        docs.append("- [3. C 语言规则](#3-c-语言规则)")
        docs.append("- [4. 汇编语言规则](#4-汇编语言规则)")
        docs.append("- [5. 语言规则差异对比](#5-语言规则差异对比)")
        docs.append("- [6. 规则查询指南](#6-规则查询指南)")
        docs.append("")
        
        docs.extend(self._generate_language_overview())
        docs.extend(self._generate_language_section(LanguageType.PYTHON))
        docs.extend(self._generate_language_section(LanguageType.C))
        docs.extend(self._generate_language_section(LanguageType.ASSEMBLY))
        docs.extend(self._generate_comparison())
        docs.extend(self._generate_query_guide())
        
        return "\n".join(docs)

    def _generate_language_overview(self) -> List[str]:
        docs = []
        docs.append("## 1. 语言概览")
        docs.append("")
        
        for lang in [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]:
            info = self.query_engine.get_language_info(lang)
            docs.append(f"### {lang.value.capitalize()}")
            docs.append(f"- **关键词数量**: {len(info['keywords'])}")
            docs.append(f"- **数据类型数量**: {len(info['data_types'])}")
            docs.append(f"- **语法规则数量**: {info['syntax_rule_count']}")
            docs.append(f"- **语义规则数量**: {info['semantic_rule_count']}")
            docs.append(f"- **总规则数量**: {info['total_rule_count']}")
            docs.append("")
        
        return docs

    def _generate_language_section(self, language: LanguageType) -> List[str]:
        docs = []
        lang_name = language.value.capitalize()
        
        docs.append(f"## {self._get_language_number(language)}. {lang_name} 语言规则")
        docs.append("")
        
        docs.append(f"### {self._get_language_number(language)}.1 关键词定义")
        docs.append("")
        
        if language == LanguageType.PYTHON:
            keywords = PythonLanguageRules.KEYWORDS
        elif language == LanguageType.C:
            keywords = CLanguageRules.KEYWORDS
        elif language == LanguageType.ASSEMBLY:
            keywords = AssemblyLanguageRules.KEYWORDS
        
        for category, kw_list in keywords.items():
            docs.append(f"#### {category.replace('_', ' ').capitalize()}")
            docs.append(", ".join(kw_list))
            docs.append("")
        
        docs.append(f"### {self._get_language_number(language)}.2 数据类型")
        docs.append("")
        
        data_types = self.query_engine.get_language_info(language)['data_types']
        docs.append("| 类型名称 | 大小 (字节) | 基础类型 |")
        docs.append("|---------|------------|---------|")
        for dt in data_types:
            docs.append(f"| {dt['name']} | {dt['size']} | {dt['base_type']} |")
        docs.append("")
        
        docs.append(f"### {self._get_language_number(language)}.3 运算符")
        docs.append("")
        
        if language == LanguageType.PYTHON:
            operators = PythonLanguageRules.OPERATORS
        elif language == LanguageType.C:
            operators = CLanguageRules.OPERATORS
        elif language == LanguageType.ASSEMBLY:
            operators = {}
        
        for op_type, op_list in operators.items():
            docs.append(f"#### {op_type.replace('_', ' ').capitalize()}")
            docs.append(", ".join(op_list))
            docs.append("")
        
        docs.append(f"### {self._get_language_number(language)}.4 语法规则")
        docs.append("")
        
        syntax_rules = self.query_engine.query_syntax_rules(language)
        for rule in syntax_rules:
            docs.append(f"#### {rule.name.replace('_', ' ').capitalize()}")
            docs.append(f"- **描述**: {rule.description}")
            docs.append(f"- **模式**: `{rule.pattern}`")
            if rule.example:
                docs.append(f"- **示例**:")
                docs.append("```")
                docs.append(rule.example)
                docs.append("```")
            docs.append(f"- **错误信息**: {rule.error_message}")
            docs.append(f"- **严重程度**: {rule.severity.value}")
            docs.append("")
        
        docs.append(f"### {self._get_language_number(language)}.5 语义规则")
        docs.append("")
        
        semantic_rules = self.query_engine.query_semantic_rules(language)
        for rule in semantic_rules:
            docs.append(f"#### {rule.name.replace('_', ' ').capitalize()}")
            docs.append(f"- **描述**: {rule.description}")
            docs.append(f"- **验证逻辑**: {rule.validation_logic}")
            docs.append(f"- **错误信息**: {rule.error_message}")
            docs.append(f"- **严重程度**: {rule.severity.value}")
            docs.append("")
        
        return docs

    def _get_language_number(self, language: LanguageType) -> int:
        mapping = {
            LanguageType.PYTHON: 2,
            LanguageType.C: 3,
            LanguageType.ASSEMBLY: 4
        }
        return mapping.get(language, 0)

    def _generate_comparison(self) -> List[str]:
        docs = []
        docs.append("## 5. 语言规则差异对比")
        docs.append("")
        
        docs.append("### 5.1 语法结构对比")
        docs.append("")
        docs.append("| 语法要素 | Python | C | 汇编 |")
        docs.append("|---------|--------|----|------|")
        docs.append("| 注释符号 | `#` | `/* */` 或 `//` | `#` 或 `;` |")
        docs.append("| 语句结束 | 换行/缩进 | `;` | 换行 |")
        docs.append("| 代码块 | 缩进 | `{}` | 标签 |")
        docs.append("| 函数定义 | `def` | 返回类型 + 函数名 | 标签 + 序言 |")
        docs.append("| 变量声明 | 动态类型 | 必须声明类型 | 数据段定义 |")
        docs.append("| 条件语句 | `if:` | `if()` | `cmp` + `jcc` |")
        docs.append("| 循环语句 | `for/in` | `for(;;)` | `loop` + `jmp` |")
        docs.append("")
        
        docs.append("### 5.2 数据类型对比")
        docs.append("")
        docs.append("| Python | C | 汇编 | 说明 |")
        docs.append("|--------|----|------|------|")
        docs.append("| `int` | `int`, `long`, `long long` | `.byte`, `.word`, `.long`, `.quad` | 整数类型 |")
        docs.append("| `float` | `float`, `double` | `.float`, `.double` | 浮点类型 |")
        docs.append("| `str` | `char*`, `char[]` | `.asciz`, `.string` | 字符串 |")
        docs.append("| `bool` | `bool` | `.byte` | 布尔类型 |")
        docs.append("| `list` | `array` | 连续内存 | 列表/数组 |")
        docs.append("| `dict` | `struct` + pointers | 自定义结构 | 键值映射 |")
        docs.append("")
        
        docs.append("### 5.3 控制流对比")
        docs.append("")
        docs.append("| 控制流 | Python | C | 汇编 |")
        docs.append("|--------|--------|----|------|")
        docs.append("| 条件分支 | `if/elif/else` | `if/else`, `switch` | `cmp`, `je`, `jne`, `jl`, `jg` |")
        docs.append("| 循环 | `for`, `while` | `for`, `while`, `do-while` | `jmp`, `loop`, `loope`, `loopne` |")
        docs.append("| 跳转 | `break`, `continue` | `break`, `continue`, `goto` | `jmp`, `ret` |")
        docs.append("| 返回 | `return` | `return` | `ret` |")
        docs.append("| 异常 | `try/except/finally` | `setjmp/longjmp` | 无原生支持 |")
        docs.append("")
        
        docs.append("### 5.4 函数调用约定")
        docs.append("")
        docs.append("| 特性 | Python | C (x86-64) | 汇编 |")
        docs.append("|------|--------|------------|------|")
        docs.append("| 参数传递 | 栈 | RDI, RSI, RDX, RCX, R8, R9 | 手动管理 |")
        docs.append("| 返回值 | 任意类型 | RAX | RAX |")
        docs.append("| 栈帧 | 自动 | `push rbp; mov rbp, rsp` | 手动管理 |")
        docs.append("| 调用约定 | 自动 | cdecl/stdcall/fastcall | 手动 |")
        docs.append("")
        
        docs.append("### 5.5 内存管理")
        docs.append("")
        docs.append("| 特性 | Python | C | 汇编 |")
        docs.append("|------|--------|----|------|")
        docs.append("| 分配方式 | 自动 | `malloc`, `calloc` | `brk`, `mmap`, `syscall` |")
        docs.append("| 释放方式 | GC | `free` | `munmap`, `syscall` |")
        docs.append("| 指针 | 引用 | 显式指针 | 寄存器 + 地址 |")
        docs.append("| 数组访问 | 边界检查 | 无检查 | 手动计算地址 |")
        docs.append("")
        
        return docs

    def _generate_query_guide(self) -> List[str]:
        docs = []
        docs.append("## 6. 规则查询指南")
        docs.append("")
        
        docs.append("### 6.1 查询方法")
        docs.append("")
        docs.append("使用 `RuleQueryEngine` 类进行规则查询：")
        docs.append("")
        docs.append("```python")
        docs.append("from agi_agent.meta_programming.rule_validator import RuleQueryEngine")
        docs.append("from agi_agent.meta_programming.language_rules import LanguageType, GrammarCategory")
        docs.append("")
        docs.append("engine = RuleQueryEngine()")
        docs.append("")
        docs.append("# 查询所有Python规则")
        docs.append("python_rules = engine.query_by_language(LanguageType.PYTHON)")
        docs.append("")
        docs.append("# 查询所有控制流规则")
        docs.append("cf_rules = engine.query_by_category(GrammarCategory.CONTROL_FLOW)")
        docs.append("")
        docs.append("# 搜索包含'function'的规则")
        docs.append("func_rules = engine.search_rules('function')")
        docs.append("")
        docs.append("# 获取规则详情")
        docs.append("details = engine.get_rule_details('function_definition', LanguageType.PYTHON)")
        docs.append("```")
        docs.append("")
        
        docs.append("### 6.2 查询接口列表")
        docs.append("")
        docs.append("| 方法 | 描述 | 参数 |")
        docs.append("|------|------|------|")
        docs.append("| `query_by_language` | 按语言查询规则 | `language: LanguageType` |")
        docs.append("| `query_by_category` | 按类别查询规则 | `category: GrammarCategory, language: Optional[LanguageType]` |")
        docs.append("| `query_by_keyword` | 按关键词搜索规则 | `keyword: str, language: Optional[LanguageType]` |")
        docs.append("| `query_by_name` | 按名称查询规则 | `name: str, language: Optional[LanguageType]` |")
        docs.append("| `query_syntax_rules` | 查询语法规则 | `language: Optional[LanguageType]` |")
        docs.append("| `query_semantic_rules` | 查询语义规则 | `language: Optional[LanguageType]` |")
        docs.append("| `get_language_info` | 获取语言信息 | `language: LanguageType` |")
        docs.append("| `get_all_languages` | 获取所有支持的语言 | 无 |")
        docs.append("| `get_all_categories` | 获取所有语法类别 | 无 |")
        docs.append("| `search_rules` | 综合搜索规则 | `query: str, language: Optional[LanguageType], category: Optional[GrammarCategory]` |")
        docs.append("| `get_rule_details` | 获取规则详细信息 | `rule_name: str, language: Optional[LanguageType]` |")
        docs.append("")
        
        docs.append("### 6.3 规则验证")
        docs.append("")
        docs.append("使用 `RuleValidator` 类验证代码：")
        docs.append("")
        docs.append("```python")
        docs.append("from agi_agent.meta_programming.rule_validator import RuleValidator")
        docs.append("from agi_agent.meta_programming.language_rules import LanguageType")
        docs.append("")
        docs.append("validator = RuleValidator(LanguageType.PYTHON)")
        docs.append("")
        docs.append("# 完整验证")
        docs.append("result = validator.validate(code)")
        docs.append("")
        docs.append("# 仅语法验证")
        docs.append("syntax_result = validator.validate_syntax_only(code)")
        docs.append("")
        docs.append("# 仅语义验证")
        docs.append("semantic_result = validator.validate_semantic_only(code)")
        docs.append("")
        docs.append("# 检查结果")
        docs.append("if result.is_valid:")
        docs.append("    print('代码验证通过')")
        docs.append("else:")
        docs.append("    for error in result.errors:")
        docs.append("        print(f\"错误: {error.message}\")")
        docs.append("```")
        docs.append("")
        
        return docs

    def generate_language_documentation(self, language: LanguageType) -> str:
        docs = []
        lang_name = language.value.capitalize()
        
        docs.append(f"# {lang_name} 语言规则文档")
        docs.append("")
        
        info = self.query_engine.get_language_info(language)
        docs.append(f"## 概览")
        docs.append(f"- **关键词数量**: {len(info['keywords'])}")
        docs.append(f"- **数据类型数量**: {len(info['data_types'])}")
        docs.append(f"- **语法规则数量**: {info['syntax_rule_count']}")
        docs.append(f"- **语义规则数量**: {info['semantic_rule_count']}")
        docs.append(f"- **总规则数量**: {info['total_rule_count']}")
        docs.append("")
        
        docs.extend(self._generate_language_section(language))
        
        return "\n".join(docs)

    def generate_rule_reference(self) -> str:
        docs = []
        docs.append("# 规则速查手册")
        docs.append("")
        
        for language in [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]:
            docs.append(f"## {language.value.capitalize()}")
            docs.append("")
            
            syntax_rules = self.query_engine.query_syntax_rules(language)
            semantic_rules = self.query_engine.query_semantic_rules(language)
            
            docs.append("### 语法规则")
            docs.append("| 规则名称 | 类别 | 严重程度 |")
            docs.append("|---------|------|---------|")
            for rule in syntax_rules:
                docs.append(f"| {rule.name} | {rule.category.value} | {rule.severity.value} |")
            docs.append("")
            
            docs.append("### 语义规则")
            docs.append("| 规则名称 | 类别 | 严重程度 |")
            docs.append("|---------|------|---------|")
            for rule in semantic_rules:
                docs.append(f"| {rule.name} | {rule.category.value} | {rule.severity.value} |")
            docs.append("")
        
        return "\n".join(docs)

    def generate_category_documentation(self, category: GrammarCategory) -> str:
        docs = []
        cat_name = category.value.replace('_', ' ').capitalize()
        
        docs.append(f"# {cat_name} 规则文档")
        docs.append("")
        
        for language in [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]:
            rules = self.query_engine.query_by_category(category, language)
            if rules:
                docs.append(f"## {language.value.capitalize()}")
                docs.append("")
                
                for rule in rules:
                    docs.append(f"### {rule.name.replace('_', ' ').capitalize()}")
                    docs.append(f"- **描述**: {rule.description}")
                    if hasattr(rule, 'pattern'):
                        docs.append(f"- **模式**: `{rule.pattern}`")
                    if hasattr(rule, 'validation_logic'):
                        docs.append(f"- **验证逻辑**: {rule.validation_logic}")
                    if rule.example:
                        docs.append(f"- **示例**:")
                        docs.append("```")
                        docs.append(rule.example)
                        docs.append("```")
                    docs.append(f"- **错误信息**: {rule.error_message}")
                    docs.append(f"- **严重程度**: {rule.severity.value}")
                    docs.append("")
        
        return "\n".join(docs)