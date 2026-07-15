import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from .service_registry import ServiceRegistry, ServiceInfo, ServiceStatus, ServiceType


class ServiceRoute:
    def __init__(self, path: str, service_name: str, 
                 method: str = "POST", timeout: float = 30.0,
                 rate_limit: int = 100):
        self.path = path
        self.service_name = service_name
        self.method = method.upper()
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._call_count = 0
        self._last_call_time = 0
        self._lock = threading.Lock()
    
    def can_call(self) -> bool:
        with self._lock:
            now = time.time()
            if now - self._last_call_time < 1.0:
                if self._call_count >= self.rate_limit:
                    return False
                self._call_count += 1
            else:
                self._call_count = 1
            self._last_call_time = now
            return True


class ServiceGateway:
    def __init__(self, registry: ServiceRegistry = None):
        self._registry = registry or ServiceRegistry()
        self._routes: Dict[str, ServiceRoute] = {}
        self._service_handlers: Dict[str, Callable] = {}
        self._middleware: List[Callable] = []
        self._lock = threading.RLock()
    
    def register_route(self, route: ServiceRoute):
        with self._lock:
            self._routes[route.path] = route
    
    def register_handler(self, service_name: str, handler: Callable):
        with self._lock:
            self._service_handlers[service_name] = handler
    
    def add_middleware(self, middleware: Callable):
        self._middleware.append(middleware)
    
    def dispatch(self, path: str, method: str = "POST", 
                 data: Any = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if path not in self._routes:
            return {
                "success": False,
                "error": f"Route not found: {path}",
                "status_code": 404
            }
        
        route = self._routes[path]
        
        if route.method != method.upper():
            return {
                "success": False,
                "error": f"Method not allowed: {method}",
                "status_code": 405
            }
        
        if not route.can_call():
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "status_code": 429
            }
        
        service_info = self._registry.get_service(route.service_name)
        if not service_info:
            return {
                "success": False,
                "error": f"Service not registered: {route.service_name}",
                "status_code": 503
            }
        
        if service_info.status != ServiceStatus.RUNNING:
            return {
                "success": False,
                "error": f"Service unavailable: {service_info.status.value}",
                "status_code": 503
            }
        
        for middleware in self._middleware:
            try:
                result = middleware(path, method, data)
                if result is not None:
                    return result
            except Exception:
                pass
        
        handler = self._service_handlers.get(route.service_name)
        
        try:
            if handler:
                result = handler(data)
            else:
                result = {"success": True, "data": data}
            
            latency = time.time() - start_time
            self._registry.report_health(route.service_name, min(1.0, 1.0 - latency / route.timeout))
            
            return {
                "success": True,
                "result": result,
                "latency": latency,
                "service": route.service_name,
                "status_code": 200
            }
        except Exception as e:
            self._registry.report_error(route.service_name, str(e))
            return {
                "success": False,
                "error": str(e),
                "service": route.service_name,
                "status_code": 500
            }
    
    def list_routes(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "path": route.path,
                    "service": route.service_name,
                    "method": route.method,
                    "timeout": route.timeout,
                    "rate_limit": route.rate_limit
                }
                for route in self._routes.values()
            ]
    
    def get_service_endpoints(self, service_name: str) -> List[str]:
        with self._lock:
            return [
                route.path for route in self._routes.values()
                if route.service_name == service_name
            ]
    
    def get_gateway_status(self) -> Dict[str, Any]:
        return {
            "total_routes": len(self._routes),
            "total_handlers": len(self._service_handlers),
            "middleware_count": len(self._middleware),
            "registry_status": self._registry.get_status_summary()
        }


def get_service_gateway() -> ServiceGateway:
    if not hasattr(get_service_gateway, '_instance'):
        get_service_gateway._instance = ServiceGateway(get_service_registry())
    return get_service_gateway._instance