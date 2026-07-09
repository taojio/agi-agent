"""
data_standards/validator.py - 数据验证

提供统一的数据验证框架，支持 Schema 定义和字段级验证
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """验证严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """验证错误"""
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "value": str(self.value) if self.value is not None else None
        }


@dataclass
class ValidationResult:
    """验证结果"""
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """是否通过验证（无 ERROR 及以上级别错误）"""
        return not any(
            e.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for e in self.errors
        )

    @property
    def has_warnings(self) -> bool:
        """是否有警告"""
        return any(e.severity == ValidationSeverity.WARNING for e in self.errors)

    @property
    def error_count(self) -> int:
        """错误总数"""
        return len(self.errors)

    def add_error(self, field: str, message: str,
                  severity: ValidationSeverity = ValidationSeverity.ERROR,
                  value: Any = None):
        """添加错误"""
        self.errors.append(ValidationError(
            field=field,
            message=message,
            severity=severity,
            value=value
        ))

    def merge(self, other: "ValidationResult"):
        """合并另一个验证结果"""
        self.errors.extend(other.errors)

    def get_errors_by_severity(self, severity: ValidationSeverity) -> List[ValidationError]:
        """按严重级别获取错误"""
        return [e for e in self.errors if e.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors]
        }

    def __str__(self) -> str:
        if self.is_valid and not self.has_warnings:
            return "Validation passed"
        lines = [f"Validation {'passed' if self.is_valid else 'failed'} "
                 f"({len(self.errors)} issues)"]
        for e in self.errors:
            lines.append(f"  [{e.severity.value.upper()}] {e.field}: {e.message}")
        return "\n".join(lines)


ValidatorFunc = Callable[[Any, str, ValidationResult], None]


@dataclass
class FieldSchema:
    """字段 Schema 定义"""
    name: str
    type_: Type = str
    required: bool = True
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    custom_validators: List[ValidatorFunc] = field(default_factory=list)
    nested_schema: Optional["Schema"] = None
    description: str = ""


class Schema:
    """数据 Schema

    定义数据的结构和验证规则
    """

    def __init__(self, fields: List[FieldSchema] = None,
                 name: str = "",
                 allow_extra: bool = False):
        self.fields: Dict[str, FieldSchema] = {}
        self.name = name
        self.allow_extra = allow_extra
        if fields:
            for f in fields:
                self.fields[f.name] = f

    def add_field(self, field: FieldSchema) -> "Schema":
        """添加字段定义"""
        self.fields[field.name] = field
        return self

    def validate(self, data: Dict[str, Any],
                 result: ValidationResult = None) -> ValidationResult:
        """验证数据

        Args:
            data: 待验证的数据字典
            result: 可选的验证结果对象，用于累积错误

        Returns:
            验证结果
        """
        if result is None:
            result = ValidationResult()

        for field_name, field_schema in self.fields.items():
            value = data.get(field_name, field_schema.default)

            if value is None and field_schema.required:
                result.add_error(
                    field_name,
                    f"Field '{field_name}' is required",
                    ValidationSeverity.ERROR
                )
                continue

            if value is None and not field_schema.required:
                continue

            self._validate_type(field_name, value, field_schema, result)
            self._validate_range(field_name, value, field_schema, result)
            self._validate_length(field_name, value, field_schema, result)
            self._validate_allowed(field_name, value, field_schema, result)
            self._validate_pattern(field_name, value, field_schema, result)
            self._validate_nested(field_name, value, field_schema, result)

            for validator in field_schema.custom_validators:
                try:
                    validator(value, field_name, result)
                except Exception as e:
                    result.add_error(
                        field_name,
                        f"Custom validator failed: {e}",
                        ValidationSeverity.ERROR,
                        value
                    )

        if not self.allow_extra:
            for key in data:
                if key not in self.fields:
                    result.add_error(
                        key,
                        f"Unexpected field '{key}'",
                        ValidationSeverity.WARNING
                    )

        return result

    def _validate_type(self, name: str, value: Any,
                       schema: FieldSchema, result: ValidationResult):
        if schema.type_ == Any:
            return
        try:
            if not isinstance(value, schema.type_):
                result.add_error(
                    name,
                    f"Expected type {schema.type_.__name__}, "
                    f"got {type(value).__name__}",
                    ValidationSeverity.ERROR,
                    value
                )
        except Exception:
            pass

    def _validate_range(self, name: str, value: Any,
                        schema: FieldSchema, result: ValidationResult):
        if schema.min_value is not None and hasattr(value, '__lt__'):
            try:
                if value < schema.min_value:
                    result.add_error(
                        name,
                        f"Value {value} is less than minimum {schema.min_value}",
                        ValidationSeverity.ERROR,
                        value
                    )
            except Exception:
                pass

        if schema.max_value is not None and hasattr(value, '__gt__'):
            try:
                if value > schema.max_value:
                    result.add_error(
                        name,
                        f"Value {value} is greater than maximum {schema.max_value}",
                        ValidationSeverity.ERROR,
                        value
                    )
            except Exception:
                pass

    def _validate_length(self, name: str, value: Any,
                         schema: FieldSchema, result: ValidationResult):
        if not hasattr(value, '__len__'):
            return

        length = len(value)

        if schema.min_length is not None and length < schema.min_length:
            result.add_error(
                name,
                f"Length {length} is less than minimum {schema.min_length}",
                ValidationSeverity.ERROR,
                value
            )

        if schema.max_length is not None and length > schema.max_length:
            result.add_error(
                name,
                f"Length {length} is greater than maximum {schema.max_length}",
                ValidationSeverity.ERROR,
                value
            )

    def _validate_allowed(self, name: str, value: Any,
                          schema: FieldSchema, result: ValidationResult):
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                result.add_error(
                    name,
                    f"Value '{value}' not in allowed values: {schema.allowed_values}",
                    ValidationSeverity.ERROR,
                    value
                )

    def _validate_pattern(self, name: str, value: Any,
                          schema: FieldSchema, result: ValidationResult):
        if schema.pattern and isinstance(value, str):
            if not re.match(schema.pattern, value):
                result.add_error(
                    name,
                    f"Value '{value}' does not match pattern '{schema.pattern}'",
                    ValidationSeverity.ERROR,
                    value
                )

    def _validate_nested(self, name: str, value: Any,
                         schema: FieldSchema, result: ValidationResult):
        if schema.nested_schema and isinstance(value, dict):
            nested_result = schema.nested_schema.validate(value)
            for err in nested_result.errors:
                err.field = f"{name}.{err.field}"
                result.errors.append(err)


class DataValidator:
    """数据验证器

    管理多个 Schema，提供统一的验证入口
    """

    def __init__(self):
        self._schemas: Dict[str, Schema] = {}

    def register_schema(self, name: str, schema: Schema) -> None:
        """注册 Schema"""
        self._schemas[name] = schema

    def unregister_schema(self, name: str) -> bool:
        """注销 Schema"""
        if name in self._schemas:
            del self._schemas[name]
            return True
        return False

    def get_schema(self, name: str) -> Optional[Schema]:
        """获取 Schema"""
        return self._schemas.get(name)

    def validate(self, data: Dict[str, Any],
                 schema_name: str = None) -> ValidationResult:
        """验证数据

        Args:
            data: 待验证的数据
            schema_name: Schema 名称，None 则通用验证

        Returns:
            验证结果
        """
        result = ValidationResult()

        if schema_name:
            schema = self._schemas.get(schema_name)
            if not schema:
                result.add_error(
                    "_schema",
                    f"Schema '{schema_name}' not found",
                    ValidationSeverity.CRITICAL
                )
                return result
            schema.validate(data, result)
        else:
            self._validate_basic(data, result)

        return result

    def _validate_basic(self, data: Dict[str, Any], result: ValidationResult):
        """基础验证"""
        if not isinstance(data, dict):
            result.add_error(
                "_root",
                f"Expected dict, got {type(data).__name__}",
                ValidationSeverity.CRITICAL
            )


_global_validator: Optional[DataValidator] = None


def get_validator() -> DataValidator:
    """获取全局数据验证器单例"""
    global _global_validator
    if _global_validator is None:
        _global_validator = DataValidator()
    return _global_validator


def validate_data(data: Dict[str, Any], schema_name: str = None) -> ValidationResult:
    """便捷函数：验证数据"""
    return get_validator().validate(data, schema_name)
