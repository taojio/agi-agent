from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class LanguageType(Enum):
    PYTHON = "python"
    C = "c"
    ASSEMBLY = "assembly"


class GrammarCategory(Enum):
    KEYWORDS = "keywords"
    DATA_TYPES = "data_types"
    CONTROL_FLOW = "control_flow"
    FUNCTIONS = "functions"
    EXPRESSIONS = "expressions"
    STATEMENTS = "statements"
    STRUCTURES = "structures"
    MODULES = "modules"
    MEMORY = "memory"
    OPERATORS = "operators"


class RuleSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SyntaxRule:
    def __init__(self, 
                 name: str,
                 language: LanguageType,
                 category: GrammarCategory,
                 pattern: str,
                 description: str,
                 example: str = "",
                 error_message: str = "",
                 severity: RuleSeverity = RuleSeverity.ERROR):
        self.name = name
        self.language = language
        self.category = category
        self.pattern = pattern
        self.description = description
        self.example = example
        self.error_message = error_message
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language.value,
            "category": self.category.value,
            "pattern": self.pattern,
            "description": self.description,
            "example": self.example,
            "error_message": self.error_message,
            "severity": self.severity.value
        }


class SemanticRule:
    def __init__(self,
                 name: str,
                 language: LanguageType,
                 category: GrammarCategory,
                 description: str,
                 validation_logic: str,
                 error_message: str = "",
                 severity: RuleSeverity = RuleSeverity.ERROR):
        self.name = name
        self.language = language
        self.category = category
        self.description = description
        self.validation_logic = validation_logic
        self.error_message = error_message
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language.value,
            "category": self.category.value,
            "description": self.description,
            "validation_logic": self.validation_logic,
            "error_message": self.error_message,
            "severity": self.severity.value
        }


class ValidationError:
    def __init__(self,
                 rule_name: str,
                 language: LanguageType,
                 category: GrammarCategory,
                 message: str,
                 severity: RuleSeverity,
                 line: int = 0,
                 column: int = 0,
                 code_snippet: str = ""):
        self.rule_name = rule_name
        self.language = language
        self.category = category
        self.message = message
        self.severity = severity
        self.line = line
        self.column = column
        self.code_snippet = code_snippet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "language": self.language.value,
            "category": self.category.value,
            "message": self.message,
            "severity": self.severity.value,
            "line": self.line,
            "column": self.column,
            "code_snippet": self.code_snippet
        }


class ValidationResult:
    def __init__(self, language: LanguageType):
        self.language = language
        self.is_valid = True
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info_messages: List[ValidationError] = []
        self.syntax_errors: List[ValidationError] = []
        self.semantic_errors: List[ValidationError] = []

    def add_error(self, error: ValidationError):
        self.errors.append(error)
        self.is_valid = False
        if error.severity == RuleSeverity.ERROR or error.severity == RuleSeverity.CRITICAL:
            self.syntax_errors.append(error)
        else:
            self.semantic_errors.append(error)

    def add_warning(self, warning: ValidationError):
        self.warnings.append(warning)

    def add_info(self, info: ValidationError):
        self.info_messages.append(info)

    def get_error_counts(self) -> Dict[str, int]:
        return {
            "total": len(self.errors),
            "critical": len([e for e in self.errors if e.severity == RuleSeverity.CRITICAL]),
            "error": len([e for e in self.errors if e.severity == RuleSeverity.ERROR]),
            "warning": len(self.warnings),
            "info": len(self.info_messages)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language.value,
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info_messages": [i.to_dict() for i in self.info_messages],
            "error_counts": self.get_error_counts()
        }


class SymbolTable:
    def __init__(self, parent: Optional["SymbolTable"] = None):
        self.parent = parent
        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.children: List["SymbolTable"] = []

    def add_symbol(self, name: str, symbol_type: str, data_type: str = "", 
                   value: Any = None, line: int = 0):
        self.symbols[name] = {
            "name": name,
            "type": symbol_type,
            "data_type": data_type,
            "value": value,
            "line": line,
            "scope": self._get_scope_name()
        }

    def lookup(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def has_symbol(self, name: str) -> bool:
        return self.lookup(name) is not None

    def create_child(self) -> "SymbolTable":
        child = SymbolTable(parent=self)
        self.children.append(child)
        return child

    def _get_scope_name(self) -> str:
        if self.parent:
            parent_scope = self.parent._get_scope_name()
            return f"{parent_scope}.child_{len(self.parent.children)}"
        return "global"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self._get_scope_name(),
            "symbols": self.symbols,
            "children": [child.to_dict() for child in self.children]
        }


class TypeDefinition:
    def __init__(self, name: str, size: int = 0, 
                 base_type: Optional[str] = None, members: List[Dict[str, Any]] = None):
        self.name = name
        self.size = size
        self.base_type = base_type
        self.members = members or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "base_type": self.base_type,
            "members": self.members
        }