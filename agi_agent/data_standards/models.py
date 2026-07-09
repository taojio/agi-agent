"""
data_standards/models.py - 统一数据模型

提供所有模块共享的数据模型基类和 Mixin，确保数据结构一致性
"""
import abc
import time
import uuid
from typing import Any, Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass, field, asdict, is_dataclass

T = TypeVar('T', bound='BaseDataModel')


class DataModelMeta(abc.ABCMeta):
    """数据模型元类

    自动注册数据模型类型，支持类型查找和版本管理
    """

    _registry: Dict[str, Type['BaseDataModel']] = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'BaseDataModel' and not namespace.get('__abstract__', False):
            model_name = namespace.get('model_name', name)
            mcs._registry[model_name] = cls
        return cls

    @classmethod
    def get_model(mcs, name: str) -> Optional[Type['BaseDataModel']]:
        """根据名称获取数据模型类"""
        return mcs._registry.get(name)

    @classmethod
    def list_models(mcs) -> List[str]:
        """列出所有已注册的数据模型"""
        return list(mcs._registry.keys())


@dataclass
class BaseDataModel(abc.ABC, metaclass=DataModelMeta):
    """数据模型基类

    所有模块间传递的数据模型都应继承自此基类，提供统一的：
    - 序列化/反序列化
    - 类型验证
    - 版本管理
    - 字典转换

    Attributes:
        model_version: 数据模型版本号，用于兼容性检查
    """

    model_version: str = "1.0"
    __abstract__: bool = True

    @classmethod
    def model_name(cls) -> str:
        """数据模型名称"""
        return cls.__name__

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            包含所有字段的字典
        """
        result = {}
        for k, v in asdict(self).items():
            if k.startswith('_'):
                continue
            result[k] = v
        result['_model_type'] = self.model_name()
        result['_model_version'] = self.model_version
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建实例

        Args:
            data: 字典数据

        Returns:
            数据模型实例

        Raises:
            ValueError: 数据版本不兼容时
        """
        model_type = data.get('_model_type', cls.__name__)
        model_version = data.get('_model_version', '1.0')

        if model_type != cls.__name__:
            actual_cls = DataModelMeta.get_model(model_type)
            if actual_cls and issubclass(actual_cls, cls):
                cls = actual_cls
            elif actual_cls:
                raise ValueError(
                    f"Model type mismatch: expected {cls.__name__}, got {model_type}"
                )

        clean_data = {k: v for k, v in data.items()
                      if not k.startswith('_')}

        if not cls._is_version_compatible(model_version):
            clean_data = cls._migrate_from_version(model_version, clean_data)

        # 只保留类中定义的字段，忽略额外字段
        valid_fields = {f for f in cls.__dataclass_fields__.keys()}
        filtered_data = {k: v for k, v in clean_data.items()
                         if k in valid_fields}

        return cls(**filtered_data)

    @classmethod
    def _is_version_compatible(cls, version: str) -> bool:
        """检查版本兼容性

        默认实现：主版本号相同则兼容
        """
        try:
            current_major = cls.model_version.__get__(cls).split('.')[0]
            data_major = version.split('.')[0]
            return current_major == data_major
        except Exception:
            return True

    @classmethod
    def _migrate_from_version(cls, version: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """版本迁移

        子类可重写此方法实现不同版本间的数据迁移
        """
        return data

    def validate(self) -> List[str]:
        """验证数据有效性

        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []
        for field_name, field_type in self.__annotations__.items():
            if field_name.startswith('_'):
                continue
            value = getattr(self, field_name)
            if value is None:
                continue
        return errors

    def clone(self: T) -> T:
        """创建数据模型的深拷贝"""
        return self.__class__(**{k: v for k, v in asdict(self).items()
                                 if not k.startswith('_')})


class IdentifiableMixin:
    """可识别对象 Mixin

    为数据模型添加唯一标识符
    """

    def __init__(self):
        self.id: str = ""

    def _generate_id(self, prefix: str = "obj") -> str:
        """生成唯一ID"""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TimestampedMixin:
    """时间戳 Mixin

    为数据模型添加创建和更新时间
    """

    def __init__(self):
        self.created_at: float = 0.0
        self.updated_at: float = 0.0

    def _init_timestamps(self):
        """初始化时间戳"""
        now = time.time()
        self.created_at = now
        self.updated_at = now

    def _touch(self):
        """更新时间戳"""
        self.updated_at = time.time()


class VersionedMixin:
    """版本控制 Mixin

    为数据模型添加版本号和版本历史
    """

    def __init__(self):
        self.version: int = 1
        self.version_history: List[Dict[str, Any]] = []

    def _bump_version(self, changes: Dict[str, Any] = None):
        """增加版本号并记录变更"""
        self.version += 1
        if changes:
            self.version_history.append({
                'version': self.version,
                'timestamp': time.time(),
                'changes': changes
            })
