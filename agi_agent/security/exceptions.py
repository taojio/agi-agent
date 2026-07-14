"""
security/exceptions.py - 安全异常定义

统一的安全异常类，用于认证、授权、验证等安全相关错误。
"""
from enum import Enum
from typing import Any, Dict, Optional


class SecurityErrorCode(Enum):
    AUTH_REQUIRED = "auth_required"
    INVALID_TOKEN = "invalid_token"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_REVOKED = "token_revoked"
    INVALID_CREDENTIALS = "invalid_credentials"
    INSUFFICIENT_PERMISSION = "insufficient_permission"
    ACCOUNT_DISABLED = "account_disabled"
    ACCOUNT_NOT_VERIFIED = "account_not_verified"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    VALIDATION_ERROR = "validation_error"
    SECURITY_BOUNDARY_VIOLATION = "security_boundary_violation"
    INPUT_SANITIZATION_FAILED = "input_sanitization_failed"
    CSRF_TOKEN_INVALID = "csrf_token_invalid"


class SecurityException(Exception):
    """安全异常基类"""

    def __init__(
        self,
        error_code: SecurityErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400,
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }


class AuthenticationException(SecurityException):
    """认证异常"""

    def __init__(
        self,
        error_code: SecurityErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(error_code, message, details, status_code=401)


class AuthorizationException(SecurityException):
    """授权异常"""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            SecurityErrorCode.INSUFFICIENT_PERMISSION,
            message,
            details,
            status_code=403,
        )


class ValidationException(SecurityException):
    """输入验证异常"""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            SecurityErrorCode.VALIDATION_ERROR,
            message,
            details,
            status_code=422,
        )


class RateLimitException(SecurityException):
    """速率限制异常"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            SecurityErrorCode.RATE_LIMIT_EXCEEDED,
            message,
            details,
            status_code=429,
        )
        self.retry_after = retry_after
