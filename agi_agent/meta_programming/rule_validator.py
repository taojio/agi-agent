import re
from typing import Any, Dict, List, Optional, Set, Union

from .language_rules import (
    LanguageType, GrammarCategory, RuleSeverity,
    SyntaxRule, SemanticRule, ValidationError, ValidationResult
)
from .python_rules import PythonLanguageRules
from .c_rules import CLanguageRules
from .assembly_rules import AssemblyLanguageRules
from .semantic_analyzer import SemanticAnalyzer


class RuleValidator:
    def __init__(self, language: LanguageType):
        self.language = language
        self.syntax_rules = self._get_syntax_rules()
        self.semantic_rules = self._get_semantic_rules()
        self.semantic_analyzer = SemanticAnalyzer(language)

    def _get_syntax_rules(self) -> List[SyntaxRule]:
        if self.language == LanguageType.PYTHON:
            return PythonLanguageRules.SYNTAX_RULES
        elif self.language == LanguageType.C:
            return CLanguageRules.SYNTAX_RULES
        elif self.language == LanguageType.ASSEMBLY:
            return AssemblyLanguageRules.SYNTAX_RULES
        return []

    def _get_semantic_rules(self) -> List[SemanticRule]:
        if self.language == LanguageType.PYTHON:
            return PythonLanguageRules.SEMANTIC_RULES
        elif self.language == LanguageType.C:
            return CLanguageRules.SEMANTIC_RULES
        elif self.language == LanguageType.ASSEMBLY:
            return AssemblyLanguageRules.SEMANTIC_RULES
        return []

    def validate(self, code: str) -> ValidationResult:
        result = ValidationResult(language=self.language)
        
        self._validate_syntax(code, result)
        
        semantic_result = self.semantic_analyzer.analyze(code)
        for error in semantic_result.errors:
            result.add_error(error)
        for warning in semantic_result.warnings:
            result.add_warning(warning)
        for info in semantic_result.info_messages:
            result.add_info(info)

        return result

    def _validate_syntax(self, code: str, result: ValidationResult):
        lines = code.split('\n')
        
        for rule in self.syntax_rules:
            pattern = re.compile(rule.pattern, re.MULTILINE)
            
            if rule.category == GrammarCategory.KEYWORDS and rule.name == "identifier":
                identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
                keywords = self._get_keywords()
                for identifier in identifiers:
                    if identifier not in keywords:
                        if not re.match(rule.pattern, identifier):
                            result.add_error(ValidationError(
                                rule_name=rule.name,
                                language=self.language,
                                category=rule.category,
                                message=rule.error_message,
                                severity=rule.severity,
                                code_snippet=identifier
                            ))
                continue

            matches = pattern.finditer(code)
            for match in matches:
                pass

            if rule.name in ["function_definition", "class_definition", "if_statement",
                             "for_loop", "while_loop", "return_statement", "import_statement",
                             "variable_assignment", "lambda_expression", "try_except",
                             "with_statement", "async_def", "match_statement", "case_statement",
                             "struct_definition", "union_definition", "enum_definition",
                             "typedef", "do_while_loop", "switch_statement",
                             "pointer_declaration", "array_declaration", "preprocessor_directive",
                             "label_definition", "directive_format", "section_directive",
                             "global_directive", "extern_directive"]:
                for line_num, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if stripped and not stripped.startswith(('#', ';', '//')):
                        if re.match(rule.pattern, stripped):
                            pass

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if self.language == LanguageType.C and stripped:
                if '=' in stripped and ';' in stripped and '==' not in stripped:
                    if 'if' not in stripped and 'while' not in stripped and 'for' not in stripped:
                        parts = stripped.split('=')
                        if len(parts) >= 2:
                            left = parts[0].strip()
                            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*\*)*\s+[a-zA-Z_][a-zA-Z0-9_]*$', left):
                                pass

            if self.language == LanguageType.ASSEMBLY and stripped and not stripped.startswith(('#', ';')):
                if ':' not in stripped:
                    tokens = stripped.split()
                    if tokens:
                        opcode = tokens[0].lower()
                        is_valid = False
                        if opcode in AssemblyLanguageRules.KEYWORDS['pseudo_instructions']:
                            is_valid = True
                        elif opcode in AssemblyLanguageRules.KEYWORDS['directives']:
                            is_valid = True
                        elif opcode in AssemblyLanguageRules.KEYWORDS['registers']:
                            is_valid = True
                        elif opcode.startswith('.'):
                            is_valid = True
                        if not is_valid:
                            result.add_error(ValidationError(
                                rule_name="invalid_instruction",
                                language=self.language,
                                category=GrammarCategory.EXPRESSIONS,
                                message=f"Invalid instruction '{opcode}'",
                                severity=RuleSeverity.ERROR,
                                line=line_num,
                                code_snippet=opcode
                            ))

    def validate_syntax_only(self, code: str) -> ValidationResult:
        result = ValidationResult(language=self.language)
        self._validate_syntax(code, result)
        return result

    def validate_semantic_only(self, code: str) -> ValidationResult:
        return self.semantic_analyzer.analyze(code)

    def _get_keywords(self) -> Set[str]:
        if self.language == LanguageType.PYTHON:
            return PythonLanguageRules.get_keywords()
        elif self.language == LanguageType.C:
            return CLanguageRules.get_keywords()
        elif self.language == LanguageType.ASSEMBLY:
            return AssemblyLanguageRules.get_keywords()
        return set()


class RuleQueryEngine:
    def __init__(self):
        self.rules: Dict[LanguageType, Dict[GrammarCategory, List[Union[SyntaxRule, SemanticRule]]]] = {}
        self._load_all_rules()

    def _load_all_rules(self):
        for lang in [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]:
            self.rules[lang] = {}
            if lang == LanguageType.PYTHON:
                all_rules = PythonLanguageRules.get_all_rules()
            elif lang == LanguageType.C:
                all_rules = CLanguageRules.get_all_rules()
            elif lang == LanguageType.ASSEMBLY:
                all_rules = AssemblyLanguageRules.get_all_rules()
            else:
                all_rules = []
            
            for rule in all_rules:
                if rule.category not in self.rules[lang]:
                    self.rules[lang][rule.category] = []
                self.rules[lang][rule.category].append(rule)

    def query_by_language(self, language: LanguageType) -> List[Union[SyntaxRule, SemanticRule]]:
        return [rule for cat_rules in self.rules.get(language, {}).values() for rule in cat_rules]

    def query_by_category(self, category: GrammarCategory, 
                          language: Optional[LanguageType] = None) -> List[Union[SyntaxRule, SemanticRule]]:
        results = []
        if language:
            results = self.rules.get(language, {}).get(category, [])
        else:
            for lang in self.rules:
                results.extend(self.rules[lang].get(category, []))
        return results

    def query_by_keyword(self, keyword: str, 
                         language: Optional[LanguageType] = None) -> List[Union[SyntaxRule, SemanticRule]]:
        results = []
        languages = [language] if language else [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]
        
        for lang in languages:
            for cat_rules in self.rules.get(lang, {}).values():
                for rule in cat_rules:
                    if keyword.lower() in rule.name.lower() or keyword.lower() in rule.description.lower():
                        results.append(rule)
        return results

    def query_by_name(self, name: str, 
                      language: Optional[LanguageType] = None) -> Optional[Union[SyntaxRule, SemanticRule]]:
        languages = [language] if language else [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]
        
        for lang in languages:
            for cat_rules in self.rules.get(lang, {}).values():
                for rule in cat_rules:
                    if rule.name == name:
                        return rule
        return None

    def query_syntax_rules(self, language: Optional[LanguageType] = None) -> List[SyntaxRule]:
        results = []
        languages = [language] if language else [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]
        
        for lang in languages:
            for cat_rules in self.rules.get(lang, {}).values():
                for rule in cat_rules:
                    if isinstance(rule, SyntaxRule):
                        results.append(rule)
        return results

    def query_semantic_rules(self, language: Optional[LanguageType] = None) -> List[SemanticRule]:
        results = []
        languages = [language] if language else [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]
        
        for lang in languages:
            for cat_rules in self.rules.get(lang, {}).values():
                for rule in cat_rules:
                    if isinstance(rule, SemanticRule):
                        results.append(rule)
        return results

    def get_language_info(self, language: LanguageType) -> Dict[str, Any]:
        if language == LanguageType.PYTHON:
            rules_class = PythonLanguageRules
        elif language == LanguageType.C:
            rules_class = CLanguageRules
        elif language == LanguageType.ASSEMBLY:
            rules_class = AssemblyLanguageRules
        else:
            return {}

        return {
            "language": language.value,
            "keywords": list(rules_class.get_keywords()),
            "data_types": [dt.to_dict() for dt in rules_class.get_data_types()],
            "syntax_rule_count": len(rules_class.get_syntax_rules()),
            "semantic_rule_count": len(rules_class.get_semantic_rules()),
            "total_rule_count": len(rules_class.get_all_rules())
        }

    def get_all_languages(self) -> List[LanguageType]:
        return list(self.rules.keys())

    def get_all_categories(self) -> List[GrammarCategory]:
        categories = set()
        for lang in self.rules:
            categories.update(self.rules[lang].keys())
        return sorted(list(categories), key=lambda x: x.value)

    def search_rules(self, query: str, 
                     language: Optional[LanguageType] = None,
                     category: Optional[GrammarCategory] = None) -> List[Union[SyntaxRule, SemanticRule]]:
        results = []
        languages = [language] if language else [LanguageType.PYTHON, LanguageType.C, LanguageType.ASSEMBLY]
        query_lower = query.lower()
        
        for lang in languages:
            cats = [category] if category else self.rules[lang].keys()
            for cat in cats:
                for rule in self.rules[lang].get(cat, []):
                    if (query_lower in rule.name.lower() or
                            query_lower in rule.description.lower() or
                            (hasattr(rule, 'example') and query_lower in rule.example.lower()) or
                            (hasattr(rule, 'pattern') and query_lower in rule.pattern.lower())):
                        results.append(rule)
        return results

    def get_rule_details(self, rule_name: str, 
                         language: Optional[LanguageType] = None) -> Optional[Dict[str, Any]]:
        rule = self.query_by_name(rule_name, language)
        if rule:
            return rule.to_dict()
        return None