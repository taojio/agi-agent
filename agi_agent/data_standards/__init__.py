"""
data_standards/__init__.py - 数据标准层

提供统一的数据模型、序列化、验证和版本控制能力
"""
from .models import (
    BaseDataModel,
    DataModelMeta,
    TimestampedMixin,
    VersionedMixin,
    IdentifiableMixin,
)
from .serialization import (
    Serializer,
    JSONSerializer,
    MessagePackSerializer,
    serialize,
    deserialize,
)
from .validator import (
    DataValidator,
    ValidationResult,
    Schema,
    FieldSchema,
    validate_data,
)

__all__ = [
    "BaseDataModel",
    "DataModelMeta",
    "TimestampedMixin",
    "VersionedMixin",
    "IdentifiableMixin",
    "Serializer",
    "JSONSerializer",
    "MessagePackSerializer",
    "serialize",
    "deserialize",
    "DataValidator",
    "ValidationResult",
    "Schema",
    "FieldSchema",
    "validate_data",
]
