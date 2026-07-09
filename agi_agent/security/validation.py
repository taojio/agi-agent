"""
security/validation.py - 输入验证与安全加固

全局输入验证中间件，包括：
- 类型与格式验证
- XSS 检测与过滤
- SQL 注入检测
- 路径遍历防护
- 文件上传安全校验
"""
import re
import html
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from .exceptions import ValidationException


class InputValidator:
    """输入验证器"""

    EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    URL_REGEX = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
    IPV4_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
    UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    PHONE_REGEX = re.compile(r"^\+?[\d\s\-()]{7,20}$")
    USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{3,32}$")

    SQL_INJECTION_PATTERNS = [
        re.compile(r"(;|--|/\*|\*/)", re.IGNORECASE),
        re.compile(r"\b(union\s+select|drop\s+table|insert\s+into|delete\s+from|update\s+.*\s+set)\b", re.IGNORECASE),
        re.compile(r"\b(exec\(|execute\(|sp_|xp_)", re.IGNORECASE),
        re.compile(r"'\s*(or|and)\s*'.*'?=", re.IGNORECASE),
        re.compile(r"\b(select|insert|update|delete|drop|alter|create)\s+(from|table|database|index)\b", re.IGNORECASE),
    ]

    XSS_PATTERNS = [
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<img[^>]+onerror", re.IGNORECASE),
        re.compile(r"eval\(", re.IGNORECASE),
    ]

    SENSITIVE_KEYWORDS = {
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "private_key", "私钥", "密码", "密钥",
    }

    def validate_email(self, email: str) -> Tuple[bool, str]:
        if not email or not isinstance(email, str):
            return False, "Email is required"
        if len(email) > 254:
            return False, "Email too long"
        if not self.EMAIL_REGEX.match(email):
            return False, "Invalid email format"
        return True, ""

    def validate_username(self, username: str) -> Tuple[bool, str]:
        if not username or not isinstance(username, str):
            return False, "Username is required"
        if len(username) < 3:
            return False, "Username too short (min 3 chars)"
        if len(username) > 32:
            return False, "Username too long (max 32 chars)"
        if not self.USERNAME_REGEX.match(username):
            return False, "Username contains invalid characters (only letters, numbers, underscore, hyphen)"
        return True, ""

    def validate_password(self, password: str) -> Tuple[bool, str]:
        if not password or not isinstance(password, str):
            return False, "Password is required"
        if len(password) < 8:
            return False, "Password too short (min 8 chars)"
        if len(password) > 128:
            return False, "Password too long (max 128 chars)"
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        score = sum([has_upper, has_lower, has_digit, has_special])
        if score < 3:
            return False, "Password too weak (need at least 3 of: uppercase, lowercase, digit, special char)"
        return True, ""

    def validate_url(self, url: str, allow_relative: bool = False) -> Tuple[bool, str]:
        if not url or not isinstance(url, str):
            return False, "URL is required"
        if allow_relative and url.startswith("/"):
            return True, ""
        if len(url) > 2048:
            return False, "URL too long"
        if not self.URL_REGEX.match(url):
            return False, "Invalid URL format"
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Only http/https URLs allowed"
        return True, ""

    def validate_ipv4(self, ip: str) -> Tuple[bool, str]:
        if not ip or not isinstance(ip, str):
            return False, "IP is required"
        if not self.IPV4_REGEX.match(ip):
            return False, "Invalid IPv4 format"
        parts = ip.split(".")
        for part in parts:
            num = int(part)
            if num < 0 or num > 255:
                return False, "IP address out of range"
        return True, ""

    def validate_uuid(self, uuid_str: str) -> Tuple[bool, str]:
        if not uuid_str or not isinstance(uuid_str, str):
            return False, "UUID is required"
        if not self.UUID_REGEX.match(uuid_str):
            return False, "Invalid UUID format"
        return True, ""

    def validate_string_length(
        self,
        value: str,
        min_len: int = 0,
        max_len: int = 1000,
        field_name: str = "String",
    ) -> Tuple[bool, str]:
        if value is None:
            return False, f"{field_name} is required"
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        if len(value) < min_len:
            return False, f"{field_name} too short (min {min_len} chars)"
        if len(value) > max_len:
            return False, f"{field_name} too long (max {max_len} chars)"
        return True, ""

    def validate_int_range(
        self,
        value: Any,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        field_name: str = "Value",
    ) -> Tuple[bool, str]:
        if value is None:
            return False, f"{field_name} is required"
        if not isinstance(value, (int, float)):
            return False, f"{field_name} must be a number"
        if isinstance(value, float) and not value.is_integer():
            return False, f"{field_name} must be an integer"
        val = int(value)
        if min_val is not None and val < min_val:
            return False, f"{field_name} too small (min {min_val})"
        if max_val is not None and val > max_val:
            return False, f"{field_name} too large (max {max_val})"
        return True, ""

    def detect_sql_injection(self, value: str) -> Tuple[bool, str]:
        if not isinstance(value, str):
            return False, ""
        for pattern in self.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                return True, f"Potential SQL injection detected: {pattern.pattern}"
        return False, ""

    def detect_xss(self, value: str) -> Tuple[bool, str]:
        if not isinstance(value, str):
            return False, ""
        for pattern in self.XSS_PATTERNS:
            if pattern.search(value):
                return True, f"Potential XSS detected: {pattern.pattern}"
        return False, ""

    def sanitize_html(self, value: str) -> str:
        """HTML 转义，防止 XSS"""
        if not isinstance(value, str):
            return str(value) if value is not None else ""
        return html.escape(value, quote=True)

    def sanitize_for_sql(self, value: str) -> str:
        """基础 SQL 清理（建议使用参数化查询而非此方法）"""
        if not isinstance(value, str):
            return str(value) if value is not None else ""
        value = value.replace("'", "''")
        value = value.replace("\\", "\\\\")
        value = value.replace("\x00", "")
        return value

    def validate_filename(self, filename: str) -> Tuple[bool, str]:
        if not filename or not isinstance(filename, str):
            return False, "Filename is required"
        if len(filename) > 255:
            return False, "Filename too long"
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Invalid filename (path traversal detected)"
        if filename.startswith("."):
            return False, "Hidden files not allowed"
        if "\x00" in filename:
            return False, "Null byte in filename"
        return True, ""

    def validate_file_upload(
        self,
        filename: str,
        file_size: int,
        allowed_types: Optional[List[str]] = None,
        max_size: int = 10 * 1024 * 1024,
    ) -> Tuple[bool, str]:
        valid, msg = self.validate_filename(filename)
        if not valid:
            return False, msg

        if file_size <= 0:
            return False, "Empty file"
        if file_size > max_size:
            return False, f"File too large (max {max_size // 1024 // 1024}MB)"

        if allowed_types:
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            if ext not in allowed_types:
                return False, f"File type not allowed. Allowed: {', '.join(allowed_types)}"

        return True, ""

    def validate_path_traversal(self, path: str, base_dir: str) -> Tuple[bool, str]:
        """检查路径是否在 base_dir 内，防止路径遍历"""
        if not path or not isinstance(path, str):
            return False, "Path is required"
        try:
            base = os.path.realpath(base_dir)
            target = os.path.realpath(os.path.join(base, path))
            if not target.startswith(base + os.sep) and target != base:
                return False, "Path traversal detected"
        except (OSError, ValueError):
            return False, "Invalid path"
        return True, ""

    def validate_json_size(self, data: Dict[str, Any], max_keys: int = 100, max_depth: int = 5) -> Tuple[bool, str]:
        """验证 JSON 数据大小和深度"""
        if not isinstance(data, dict):
            return False, "Data must be an object"

        def check_depth(obj, depth):
            if depth > max_depth:
                return False
            if isinstance(obj, dict):
                if len(obj) > max_keys:
                    return False
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        if not check_depth(v, depth + 1):
                            return False
            elif isinstance(obj, list):
                if len(obj) > max_keys:
                    return False
                for item in obj:
                    if isinstance(item, (dict, list)):
                        if not check_depth(item, depth + 1):
                            return False
            return True

        if not check_depth(data, 1):
            return False, f"JSON too large or too deep (max {max_keys} keys, {max_depth} levels)"
        return True, ""


_global_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator
