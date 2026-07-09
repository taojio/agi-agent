"""
core/exceptions.py - 统一异常体系

所有模块的异常基类，建立标准化的错误处理机制
"""
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """错误码枚举"""
    UNKNOWN = "UNKNOWN"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    MODULE_ALREADY_EXISTS = "MODULE_ALREADY_EXISTS"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class CoreException(Exception):
    """核心异常基类

    所有项目异常都应继承自此基类，提供统一的错误信息格式
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


class ModuleException(CoreException):
    """模块异常基类"""

    def __init__(
        self,
        message: str,
        module_name: str = "",
        error_code: ErrorCode = ErrorCode.RUNTIME_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.module_name = module_name
        super().__init__(message, error_code, details)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["module_name"] = self.module_name
        return d


class ConfigurationException(CoreException):
    """配置异常"""

    def __init__(self, message: str, config_key: str = "", details: Optional[Dict[str, Any]] = None):
        self.config_key = config_key
        super().__init__(message, ErrorCode.CONFIGURATION_ERROR, details)


class InitializationException(ModuleException):
    """初始化异常"""

    def __init__(self, message: str, module_name: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, module_name, ErrorCode.INITIALIZATION_FAILED, details)


class ValidationException(CoreException):
    """验证异常"""

    def __init__(self, message: str, field: str = "", details: Optional[Dict[str, Any]] = None):
        self.field = field
        super().__init__(message, ErrorCode.VALIDATION_FAILED, details)
