"""
security/headers.py - 安全响应头部

设置各种安全相关的 HTTP 响应头，包括：
- CSP (Content Security Policy)
- HSTS (HTTP Strict Transport Security)
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Referrer-Policy
- Permissions-Policy
"""
from typing import Dict, List, Optional


class SecurityHeaders:
    """安全响应头部配置"""

    def __init__(
        self,
        csp_enabled: bool = True,
        hsts_enabled: bool = False,
        frame_options: str = "DENY",
        referrer_policy: str = "strict-origin-when-cross-origin",
    ):
        self.csp_enabled = csp_enabled
        self.hsts_enabled = hsts_enabled
        self.frame_options = frame_options
        self.referrer_policy = referrer_policy

    def get_default_headers(self) -> Dict[str, str]:
        """获取默认安全头部"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": self.frame_options,
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": self.referrer_policy,
            "X-Permitted-Cross-Domain-Policies": "none",
            "X-Download-Options": "noopen",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Permissions-Policy": self._get_permissions_policy(),
        }

        if self.csp_enabled:
            headers["Content-Security-Policy"] = self._get_csp_policy()

        if self.hsts_enabled:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return headers

    def _get_csp_policy(self) -> str:
        """获取 CSP 策略（默认严格模式）"""
        directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-src 'none'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
        return "; ".join(directives)

    def _get_permissions_policy(self) -> str:
        """获取浏览器权限策略"""
        policies = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "fullscreen=(self)",
            "payment=()",
            "accelerometer=()",
            "gyroscope=()",
            "magnetometer=()",
        ]
        return ", ".join(policies)

    def get_csp_report_only(self) -> Dict[str, str]:
        """获取仅报告模式的 CSP（不阻断，仅上报）"""
        csp = self._get_csp_policy()
        return {"Content-Security-Policy-Report-Only": csp}

    def get_websocket_headers(self) -> Dict[str, str]:
        """WebSocket 连接的安全头部"""
        return {
            "X-Content-Type-Options": "nosniff",
        }

    def get_file_upload_headers(self) -> Dict[str, str]:
        """文件上传响应的安全头部"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Download-Options": "noopen",
            "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        }


_global_security_headers: Optional[SecurityHeaders] = None


def get_security_headers() -> SecurityHeaders:
    global _global_security_headers
    if _global_security_headers is None:
        _global_security_headers = SecurityHeaders()
    return _global_security_headers
