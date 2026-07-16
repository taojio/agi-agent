import json
import inspect
import typing
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints
from collections import defaultdict


class InterfaceStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class DataFormat(Enum):
    JSON = "json"
    BSON = "bson"
    PROTOBUF = "protobuf"
    MESSAGEPACK = "messagepack"


class CompatibilityLevel(Enum):
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    NONE = "none"


@dataclass
class FieldDefinition:
    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Optional[Any] = None
    enum: Optional[List[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    pattern: Optional[str] = None
    examples: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("default") if result["default"] is None else None
        result.pop("enum") if result["enum"] is None else None
        result.pop("min") if result["min"] is None else None
        result.pop("max") if result["max"] is None else None
        result.pop("pattern") if result["pattern"] is None else None
        return result


@dataclass
class SchemaDefinition:
    name: str
    description: str = ""
    fields: List[FieldDefinition] = field(default_factory=list)
    version: str = "1.0.0"
    format: DataFormat = DataFormat.JSON

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "version": self.version,
            "format": self.format.value,
        }

    def validate(self, data: Dict[str, Any]) -> List[str]:
        errors = []

        for field_def in self.fields:
            if field_def.required and field_def.name not in data:
                errors.append(f"Missing required field: {field_def.name}")
                continue

            if field_def.name in data:
                value = data[field_def.name]
                errors.extend(self._validate_field(field_def, value))

        for key in data:
            if key not in [f.name for f in self.fields]:
                errors.append(f"Unknown field: {key}")

        return errors

    def _validate_field(self, field_def: FieldDefinition, value: Any) -> List[str]:
        errors = []

        if field_def.type == "int":
            if not isinstance(value, int):
                errors.append(f"Field {field_def.name} must be int")
        elif field_def.type == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"Field {field_def.name} must be float")
        elif field_def.type == "str":
            if not isinstance(value, str):
                errors.append(f"Field {field_def.name} must be string")
        elif field_def.type == "bool":
            if not isinstance(value, bool):
                errors.append(f"Field {field_def.name} must be boolean")
        elif field_def.type == "list":
            if not isinstance(value, list):
                errors.append(f"Field {field_def.name} must be list")
        elif field_def.type == "dict":
            if not isinstance(value, dict):
                errors.append(f"Field {field_def.name} must be dict")

        if field_def.min is not None and value < field_def.min:
            errors.append(f"Field {field_def.name} below minimum {field_def.min}")
        if field_def.max is not None and value > field_def.max:
            errors.append(f"Field {field_def.name} above maximum {field_def.max}")
        if field_def.enum and value not in field_def.enum:
            errors.append(f"Field {field_def.name} must be one of {field_def.enum}")

        return errors


@dataclass
class InterfaceDefinition:
    name: str
    description: str = ""
    version: str = "1.0.0"
    status: InterfaceStatus = InterfaceStatus.ACTIVE
    input_schema: Optional[SchemaDefinition] = None
    output_schema: