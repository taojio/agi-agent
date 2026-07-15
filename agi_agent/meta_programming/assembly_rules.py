from typing import List, Dict, Optional, Set
from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SyntaxRule, SemanticRule, TypeDefinition
)


class AssemblyLanguageRules:
    KEYWORDS = {
        "registers": ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
                      "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp",
                      "ax", "bx", "cx", "dx", "si", "di", "bp", "sp",
                      "al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"],
        "segment_registers": ["cs", "ds", "es", "fs", "gs", "ss"],
        "flags": ["cf", "pf", "af", "zf", "sf", "tf", "if", "df", "of"],
        "fpu_registers": ["st0", "st1", "st2", "st3", "st4", "st5", "st6", "st7"],
        "mmx_registers": ["mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"],
        "sse_registers": ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
                          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"],
        "avx_registers": ["ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
                          "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"],
        "directives": [".section", ".text", ".data", ".bss", ".rodata",
                       ".global", ".extern", ".equ", ".asciz", ".string",
                       ".byte", ".word", ".long", ".quad", ".double", ".float",
                       ".align", ".globl", ".type", ".size", ".end",
                       ".macro", ".endm", ".endmacro", ".rept", ".endr",
                       ".if", ".else", ".endif", ".ifdef", ".ifndef", ".elif",
                       ".include", ".incbin", ".fill", ".space", ".skip",
                       ".code32", ".code64", ".intel_syntax", ".att_syntax"],
        "pseudo_instructions": ["mov", "lea", "add", "sub", "mul", "div", "imul", "idiv",
                                "and", "or", "xor", "not",
                                "shl", "shr", "sal", "sar",
                                "rol", "ror", "rcl", "rcr",
                                "cmp", "test",
                                "jmp", "je", "jne", "jz", "jnz", "jb", "jnb", "ja", "jna",
                                "jl", "jge", "jg", "jle", "jo", "jno", "js", "jns",
                                "jcxz", "jecxz", "loop", "loope", "loopne",
                                "call", "ret", "int", "syscall",
                                "push", "pop", "pushq", "popq",
                                "inc", "dec", "neg", "adc", "sbb",
                                "movsx", "movzx", "cwde", "cdqe",
                                "xchg", "bswap", "cmpxchg",
                                "in", "out", "cli", "sti", "hlt",
                                "nop", "ud2",
                                "fld", "fst", "fstp", "fadd", "fsub", "fmul", "fdiv",
                                "fild", "fist", "fistp",
                                "fcom", "fcomp", "fcompp", "fxch",
                                "sqrtss", "sqrtsd",
                                "addss", "addsd", "subss", "subsd",
                                "mulss", "mulsd", "divss", "divsd",
                                "movss", "movsd", "movaps", "movupd",
                                "addps", "addpd", "subps", "subpd",
                                "mulps", "mulpd", "divps", "divpd",
                                "cmpcc", "setcc", "cmovcc",
                                "vmovss", "vmovsd", "vaddss", "vaddsd",
                                "vsubss", "vsubsd", "vmulss", "vmulsd",
                                "vdivss", "vdivsd",
                                "movdqa", "movdqu", "paddb", "paddw", "paddd",
                                "psubb", "psubw", "psubd",
                                "pmuludq", "pxor", "pand", "por"],
        "macros": ["macro", "endmacro", "ifdef", "ifndef", "else", "endif", "include", "equ"]
    }

    DATA_TYPES = [
        TypeDefinition("byte", size=1, base_type="integer"),
        TypeDefinition("word", size=2, base_type="integer"),
        TypeDefinition("dword", size=4, base_type="integer"),
        TypeDefinition("qword", size=8, base_type="integer"),
        TypeDefinition("float", size=4, base_type="floating"),
        TypeDefinition("double", size=8, base_type="floating"),
        TypeDefinition("tbyte", size=10, base_type="floating"),
        TypeDefinition("string", size=0, base_type="sequence"),
        TypeDefinition("oword", size=16, base_type="integer"),
        TypeDefinition("yword", size=32, base_type="integer"),
        TypeDefinition("zword", size=64, base_type="integer")
    ]

    ADDRESSING_MODES = {
        "register": r'^[a-zA-Z][a-zA-Z0-9]*$',
        "immediate": r'^(\$|0x)?[0-9a-fA-F]+$',
        "memory_direct": r'^[a-zA-Z_][a-zA-Z0-9_]*$',
        "memory_displacement": r'^[a-zA-Z_][a-zA-Z0-9_]*\([a-zA-Z][a-zA-Z0-9]*\)$',
        "memory_scaled": r'^[a-zA-Z_][a-zA-Z0-9_]*?\([a-zA-Z][a-zA-Z0-9]*(\s*,\s*[a-zA-Z][a-zA-Z0-9]*(\s*,\s*[0-9]+)?)?\)$',
        "memory_indexed": r'^\([a-zA-Z][a-zA-Z0-9]*(\s*,\s*[a-zA-Z][a-zA-Z0-9]*)?\)$',
        "memory_rip_relative": r'^\[?rip(\s*\+\s*[0-9a-fA-Fx]+)?\]?$',
        "memory_sib": r'^\[?[a-zA-Z][a-zA-Z0-9]*\s*\+\s*[a-zA-Z][a-zA-Z0-9]*\s*\*\s*[1248]\]?$',
        "memory_displacement_only": r'^\[?[0-9a-fA-Fx]+\]?$'
    }

    CALLING_CONVENTIONS = {
        "system_v_amd64": {
            "integer_args": ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
            "floating_args": ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"],
            "return_value": ["rax"],
            "callee_saved": ["rbx", "rbp", "r12", "r13", "r14", "r15"],
            "caller_saved": ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11",
                             "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"],
            "stack_alignment": 16
        },
        "microsoft_x64": {
            "integer_args": ["rcx", "rdx", "r8", "r9"],
            "floating_args": ["xmm0", "xmm1", "xmm2", "xmm3"],
            "return_value": ["rax"],
            "callee_saved": ["rbx", "rbp", "rdi", "rsi", "r12", "r13", "r14", "r15"],
            "caller_saved": ["rax", "rcx", "rdx", "r8", "r9", "r10", "r11",
                             "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"],
            "stack_alignment": 16,
            "shadow_space": 32
        }
    }

    SYNTAX_RULES = [
        SyntaxRule(
            name="instruction_format",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*(\s+[a-zA-Z0-9_\$\[\]\(\),]+)*$',
            description="指令由操作码和可选的操作数组成",
            example="mov rax, rbx\nadd rcx, 10\njmp label",
            error_message="指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="label_definition",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.KEYWORDS,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:',
            description="标签以冒号结尾",
            example="start:\nloop_label:",
            error_message="标签定义格式错误：缺少冒号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="directive_format",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.[a-zA-Z_][a-zA-Z0-9_]*(\s+.*)?$',
            description="伪指令以点号开头",
            example=".text\n.data\n.global main",
            error_message="伪指令格式错误：缺少点号",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="section_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.(section|text|data|bss|rodata)\s*(\"[^\"]*\")?$',
            description="段定义伪指令",
            example=".section .text\n.text\n.data",
            error_message="段定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="global_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.(global|globl)\s+[a-zA-Z_][a-zA-Z0-9_]+$',
            description=".global声明全局符号",
            example=".global main\n.global _start",
            error_message=".global格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="extern_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.extern\s+[a-zA-Z_][a-zA-Z0-9_]+$',
            description=".extern声明外部符号",
            example=".extern printf\n.extern malloc",
            error_message=".extern格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="data_definition_byte",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(\s*=\s*)?\s*\.byte\s+[0-9]+(,\s*[0-9]+)*$',
            description=".byte定义字节数据",
            example="value: .byte 42\narray: .byte 1, 2, 3",
            error_message=".byte定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="data_definition_word",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(\s*=\s*)?\s*\.word\s+[0-9]+(,\s*[0-9]+)*$',
            description=".word定义字数据",
            example="value: .word 1000\narray: .word 1, 2, 3",
            error_message=".word定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="data_definition_long",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(\s*=\s*)?\s*\.long\s+[0-9]+(,\s*[0-9]+)*$',
            description=".long定义双字数据",
            example="value: .long 100000\narray: .long 1, 2, 3",
            error_message=".long定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="data_definition_quad",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(\s*=\s*)?\s*\.quad\s+[0-9]+(,\s*[0-9]+)*$',
            description=".quad定义四字数据",
            example="value: .quad 1000000000\narray: .quad 1, 2, 3",
            error_message=".quad定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="string_definition",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(\s*=\s*)?\s*\.(asciz|string)\s+"[^"]*"$',
            description=".asciz或.string定义字符串",
            example="msg: .asciz \"Hello, World!\"\nstr: .string \"test\"",
            error_message="字符串定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="equ_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[0-9]+$',
            description="equ定义常量",
            example="MAX = 100\nSIZE = 42",
            error_message="equ定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="memory_addressing",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            pattern=r'^(\[)?[a-zA-Z_][a-zA-Z0-9_]*(\])?(\s*,\s*[a-zA-Z][a-zA-Z0-9]*(\s*,\s*[0-9]+)?)*$',
            description="内存寻址使用方括号",
            example="mov rax, [rbx]\nmov rax, [rbx+rcx*4]\nmov rax, [array+rdi*8]",
            error_message="内存寻址格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="immediate_value",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^(\$|0x)?[0-9a-fA-F]+$',
            description="立即数可以是十进制或十六进制",
            example="mov rax, 10\nmov rax, $10\nmov rax, 0x10",
            error_message="立即数格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="comment_line",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.STATEMENTS,
            pattern=r'^(\s*#|\s*;).*$',
            description="注释以#或;开头",
            example="# This is a comment\n; Another comment",
            error_message="注释格式错误",
            severity=RuleSeverity.INFO
        ),
        SyntaxRule(
            name="function_prologue",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^pushq?\s+rbp\s*\n^\s*movq?\s+rbp,\s*rsp',
            description="函数序言：保存栈帧",
            example="push rbp\nmov rbp, rsp",
            error_message="函数序言格式错误",
            severity=RuleSeverity.WARNING
        ),
        SyntaxRule(
            name="function_epilogue",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^movq?\s+rsp,\s*rbp\s*\n^\s*popq?\s+rbp\s*\n^\s*ret',
            description="函数尾声：恢复栈帧",
            example="mov rsp, rbp\npop rbp\nret",
            error_message="函数尾声格式错误",
            severity=RuleSeverity.WARNING
        ),
        SyntaxRule(
            name="system_call",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            pattern=r'^movq?\s+rax,\s*\d+\s*\n^\s*syscall',
            description="系统调用需要设置rax为系统调用号",
            example="mov rax, 1\nmov rdi, 1\nsyscall",
            error_message="系统调用格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="sse_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^(movss|movsd|movaps|movupd|addss|addsd|subss|subsd|mulss|mulsd|divss|divsd|sqrtss|sqrtsd|addps|addpd|subps|subpd|mulps|mulpd|divps|divpd)\s+(xmm\d+|.*\[.*\]),\s*(xmm\d+|.*\[.*\])$',
            description="SSE指令用于单指令多数据流操作",
            example="addss xmm0, xmm1\nmovss xmm0, [value]",
            error_message="SSE指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="avx_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^v[a-zA-Z]+\s+(xmm|ymm)\d+,\s+(xmm|ymm)\d+,\s+(xmm|ymm)\d+$',
            description="AVX指令支持三操作数格式",
            example="vaddss xmm0, xmm1, xmm2\nvmulsd ymm0, ymm1, ymm2",
            error_message="AVX指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="fpu_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            pattern=r'^(fld|fst|fstp|fadd|fsub|fmul|fdiv|fild|fist|fistp|fcom|fcomp|fcompp|fxch)(\s+(st\d+|\[.*\]|.*))?$',
            description="FPU指令操作浮点寄存器栈",
            example="fld qword [value]\nfadd st0, st1\nfstp st0",
            error_message="FPU指令格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="macro_definition",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.macro\s+[a-zA-Z_][a-zA-Z0-9_]*(\s+[a-zA-Z_][a-zA-Z0-9_]*)*$',
            description="宏定义使用.macro/.endm",
            example=".macro add_value reg, val\n    add \\reg, \\val\n.endm",
            error_message="宏定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="macro_end",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.(endm|endmacro)\s*$',
            description="宏定义结束使用.endm或.endmacro",
            example=".endm\n.endmacro",
            error_message="宏结束格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="conditional_assembly",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.(if|ifdef|ifndef|elif|else|endif)\s+.*$',
            description="条件汇编根据条件包含或排除代码",
            example=".ifdef DEBUG\n    mov rax, 1\n.endif",
            error_message="条件汇编格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="repeat_block",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.CONTROL_FLOW,
            pattern=r'^\.rept\s+\d+$',
            description=".rept用于重复汇编代码块",
            example=".rept 10\n    nop\n.endr",
            error_message="重复块格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="rip_relative_addressing",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            pattern=r'^\[?rip(\s*[+\-]\s*[0-9a-fA-Fx]+)?\]?$',
            description="RIP相对寻址用于位置无关代码",
            example="lea rax, [rip+label]\nmov rax, [rip+0x100]",
            error_message="RIP相对寻址格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="sib_addressing",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            pattern=r'^\[?[a-zA-Z_][a-zA-Z0-9_]*\s*\+\s*[a-zA-Z][a-zA-Z0-9]*\s*\*\s*[1248]\]?$',
            description="SIB寻址支持基址+索引*比例因子",
            example="mov rax, [rbx+rcx*4]\nmov rax, [array+rdi*8]",
            error_message="SIB寻址格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="data_definition_float",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*\.(float|double)\s+[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?$',
            description=".float和.double定义浮点数据",
            example="pi: .float 3.14159\ne: .double 2.71828",
            error_message="浮点数据定义格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="fill_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^\.fill\s+\d+,\s*\d+,\s*[0-9a-fA-Fx]+$',
            description=".fill用于填充指定数量的字节",
            example=".fill 100, 1, 0\n.fill 16, 4, 0xFFFFFFFF",
            error_message=".fill格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="space_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            pattern=r'^\.(space|skip)\s+\d+(\s*,\s*[0-9a-fA-Fx]+)?$',
            description=".space/.skip用于分配未初始化空间",
            example=".space 256\n.skip 100, 0",
            error_message=".space/.skip格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="align_directive",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.align\s+\d+$',
            description=".align用于对齐数据或代码",
            example=".align 16\n.align 4",
            error_message=".align格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="intel_syntax",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.intel_syntax(\s+noprefix|\s+prefix)?$',
            description=".intel_syntax切换为Intel语法",
            example=".intel_syntax\n.intel_syntax noprefix",
            error_message=".intel_syntax格式错误",
            severity=RuleSeverity.ERROR
        ),
        SyntaxRule(
            name="att_syntax",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            pattern=r'^\.att_syntax\s*$',
            description=".att_syntax切换为AT&T语法",
            example=".att_syntax",
            error_message=".att_syntax格式错误",
            severity=RuleSeverity.ERROR
        )
    ]

    SEMANTIC_RULES = [
        SemanticRule(
            name="invalid_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="使用不存在的指令会导致汇编错误",
            validation_logic="检查指令是否在指令集中",
            error_message="无效的指令",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="invalid_register",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="使用不存在的寄存器会导致汇编错误",
            validation_logic="检查寄存器名是否有效",
            error_message="无效的寄存器名",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="operand_count_mismatch",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="指令操作数数量必须正确",
            validation_logic="检查指令操作数数量",
            error_message="操作数数量错误",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="operand_type_mismatch",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="指令操作数类型必须兼容",
            validation_logic="检查操作数类型是否与指令兼容",
            error_message="操作数类型不兼容",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="invalid_addressing_mode",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="指令不支持某些寻址模式",
            validation_logic="检查寻址模式是否与指令兼容",
            error_message="无效的寻址模式",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="undefined_label",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.KEYWORDS,
            description="引用未定义的标签会导致链接错误",
            validation_logic="检查跳转目标标签是否已定义",
            error_message="标签未定义",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="duplicate_label",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.KEYWORDS,
            description="重复定义标签会导致汇编错误",
            validation_logic="检查标签是否已存在",
            error_message="标签重复定义",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="stack_alignment",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="栈必须保持16字节对齐",
            validation_logic="检查栈指针是否对齐",
            error_message="栈未对齐",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="missing_function_prologue",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            description="函数缺少标准序言",
            validation_logic="检查函数开头是否保存栈帧",
            error_message="函数缺少序言",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="missing_function_epilogue",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            description="函数缺少标准尾声",
            validation_logic="检查函数结尾是否恢复栈帧",
            error_message="函数缺少尾声",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="unreachable_code",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.STATEMENTS,
            description="跳转指令后的代码不可达",
            validation_logic="检查跳转后是否有代码",
            error_message="不可达代码",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="missing_ret",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            description="函数缺少ret指令",
            validation_logic="检查函数是否以ret结尾",
            error_message="函数缺少返回指令",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="invalid_syscall_number",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            description="系统调用号必须有效",
            validation_logic="检查rax中的系统调用号是否在有效范围内",
            error_message="无效的系统调用号",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="data_section_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.STATEMENTS,
            description="指令不能出现在数据段中",
            validation_logic="检查指令位置",
            error_message="指令不能出现在数据段中",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="calling_convention_check",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.FUNCTIONS,
            description="函数调用应遵循调用约定",
            validation_logic="检查参数传递寄存器和栈使用是否符合约定",
            error_message="可能违反调用约定",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="register_size_mismatch",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="指令操作数的寄存器大小应匹配",
            validation_logic="检查操作数的寄存器大小是否一致",
            error_message="寄存器大小不匹配",
            severity=RuleSeverity.ERROR
        ),
        SemanticRule(
            name="flag_register_modification",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="某些指令会修改标志寄存器",
            validation_logic="检查指令是否影响标志位",
            error_message="指令会修改标志寄存器",
            severity=RuleSeverity.INFO
        ),
        SemanticRule(
            name="stack_imbalance",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="push和pop应配对使用以避免栈不平衡",
            validation_logic="检查函数中push和pop的数量是否匹配",
            error_message="可能的栈不平衡",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="uninitialized_data_use",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.DATA_TYPES,
            description="未初始化的数据段变量可能包含随机值",
            validation_logic="检查bss段数据是否在使用前被初始化",
            error_message="未初始化数据可能包含随机值",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="simd_alignment",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="某些SIMD指令需要内存操作数对齐",
            validation_logic="检查movdqa等指令的内存操作数是否对齐",
            error_message="SIMD内存操作数可能未对齐",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="self_modifying_code",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.STATEMENTS,
            description="修改代码段可能导致安全问题",
            validation_logic="检查是否有写入指令到代码段的操作",
            error_message="可能的自修改代码",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="privileged_instruction",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="特权指令只能在内核模式下执行",
            validation_logic="检查是否使用了特权指令",
            error_message="特权指令可能在用户模式下导致异常",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="fpu_stack_overflow",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.EXPRESSIONS,
            description="FPU寄存器栈深度有限（8级）",
            validation_logic="检查FPU寄存器栈是否可能溢出",
            error_message="FPU寄存器栈可能溢出",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="branch_prediction_hint",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.CONTROL_FLOW,
            description="分支指令的预测方向可能影响性能",
            validation_logic="分析分支模式并提供优化建议",
            error_message="分支预测提示",
            severity=RuleSeverity.INFO
        ),
        SemanticRule(
            name="memory_ordering",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MEMORY,
            description="内存操作顺序可能需要内存屏障",
            validation_logic="检查是否需要内存屏障指令",
            error_message="可能需要内存屏障",
            severity=RuleSeverity.INFO
        ),
        SemanticRule(
            name="macro_parameter_mismatch",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            description="宏调用参数数量应与定义匹配",
            validation_logic="比较宏调用参数与宏定义参数数量",
            error_message="宏参数数量不匹配",
            severity=RuleSeverity.WARNING
        ),
        SemanticRule(
            name="section_mismatch",
            language=LanguageType.ASSEMBLY,
            category=GrammarCategory.MODULES,
            description="代码应在.text段，数据应在.data/.bss段",
            validation_logic="检查内容是否在正确的段中",
            error_message="内容可能在错误的段中",
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

    @classmethod
    def get_register_size(cls, reg: str) -> int:
        size_map = {
            'rax': 8, 'rbx': 8, 'rcx': 8, 'rdx': 8,
            'rsi': 8, 'rdi': 8, 'rbp': 8, 'rsp': 8,
            'r8': 8, 'r9': 8, 'r10': 8, 'r11': 8,
            'r12': 8, 'r13': 8, 'r14': 8, 'r15': 8,
            'eax': 4, 'ebx': 4, 'ecx': 4, 'edx': 4,
            'esi': 4, 'edi': 4, 'ebp': 4, 'esp': 4,
            'ax': 2, 'bx': 2, 'cx': 2, 'dx': 2,
            'si': 2, 'di': 2, 'bp': 2, 'sp': 2,
            'al': 1, 'bl': 1, 'cl': 1, 'dl': 1,
            'ah': 1, 'bh': 1, 'ch': 1, 'dh': 1
        }
        return size_map.get(reg.lower(), 0)