import os
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import defaultdict


class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    DEGRADED = "degraded"


class ServiceType(Enum):
    CORE = "core"
    COGNITIVE = "cognitive"
    PERCEPTION = "perception"
    MEMORY = "memory"
    DECISION = "decision"
    LEARNING = "learning"
    EVOLUTION = "evolution"
    MULTI_AGENT = "multi_agent"
    PLUGIN = "plugin"
    API = "api"
    UTILITY = "utility"


class ServiceInfo:
    def __init__(self, name: str, service_type: ServiceType, 
                 description: str = "", version: str = "1.0.0",
                 endpoints: List[str] = None, dependencies: List[str] = None):
        self.name = name
        self.type = service_type
        self.description = description
        self.version = version
        self.endpoints = endpoints or []
        self.dependencies = dependencies or []
        self.status = ServiceStatus.STOPPED
        self.start_time = None
        self.health_score = 0.0
        self.error_count = 0
        self.last_error = None
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "version": self.version,
            "endpoints": self.endpoints,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "start_time": self.start_time,
            "health_score": self.health_score,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }
    
    def update_health(self, score: float):
        self.health_score = score
        if score >= 0.9:
            self.status = ServiceStatus.RUNNING
        elif score >= 0.5:
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.ERROR


class ServiceRegistry:
    def __init__(self, registry_file: str = None):
        self._services: Dict[str, ServiceInfo] = {}
        self._type_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._registry_file = registry_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "service_registry.json"
        )
        self._event_callbacks: Dict[str, List[Callable]] = {}
        
        os.makedirs(os.path.dirname(self._registry_file), exist_ok=True)
        self._load_registry()
    
    def register(self, info: ServiceInfo) -> bool:
        with self._lock:
            if info.name in self._services:
                return False
            
            self._services[info.name] = info
            self._type_index[info.type.value].append(info.name)
            self._save_registry()
            self._emit_event("service_registered", info.to_dict())
            return True
    
    def deregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._services:
                return False
            
            info = self._services[name]
            self._type_index[info.type.value].remove(name)
            del self._services[name]
            self._save_registry()
            self._emit_event("service_deregistered", {"name": name})
            return True
    
    def get_service(self, name: str) -> Optional[ServiceInfo]:
        return self._services.get(name)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        with self._lock:
            names = self._type_index.get(service_type.value, [])
            return [self._services[name] for name in names if name in self._services]
    
    def get_all_services(self) -> List[ServiceInfo]:
        return list(self._services.values())
    
    def update_status(self, name: str, status: ServiceStatus):
        with self._lock:
            if name in self._services:
                self._services[name].status = status
                if status == ServiceStatus.RUNNING:
                    self._services[name].start_time = time.time()
                self._save_registry()
                self._emit_event("service_status_changed", {
                    "name": name,
                    "status": status.value
                })
    
    def report_health(self, name: str, health_score: float):
        with self._lock:
            if name in self._services:
                self._services[name].update_health(health_score)
                self._save_registry()
    
    def report_error(self, name: str, error: str):
        with self._lock:
            if name in self._services:
                service = self._services[name]
                service.error_count += 1
                service.last_error = error
                if service.error_count > 10:
                    service.status = ServiceStatus.ERROR
                self._save_registry()
                self._emit_event("service_error", {
                    "name": name,
                    "error": error,
                    "error_count": service.error_count
                })
    
    def get_status_summary(self) -> Dict[str, Any]:
        with self._lock:
            summary = {}
            for status in ServiceStatus:
                count = sum(1 for s in self._services.values() if s.status == status)
                summary[status.value] = count
            
            total = len(self._services)
            avg_health = sum(s.health_score for s in self._services.values()) / max(total, 1)
            
            return {
                "total_services": total,
                "status_distribution": summary,
                "average_health": avg_health,
                "types": {
                    t: len(names) for t, names in self._type_index.items()
                }
            }
    
    def on_event(self, event_name: str, callback: Callable):
        if event_name not in self._event_callbacks:
            self._event_callbacks[event_name] = []
        self._event_callbacks[event_name].append(callback)
    
    def _emit_event(self, event_name: str, data: Any = None):
        for callback in self._event_callbacks.get(event_name, []):
            try:
                callback(event_name, data)
            except Exception:
                pass
    
    def _save_registry(self):
        try:
            data = {name: info.to_dict() for name, info in self._services.items()}
            with open(self._registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _load_registry(self):
        try:
            if os.path.exists(self._registry_file):
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                
                for name, info_dict in data.items():
                    info = ServiceInfo(
                        name=name,
                        service_type=ServiceType(info_dict.get("type", "utility")),
                        description=info_dict.get("description", ""),
                        version=info_dict.get("version", "1.0.0"),
                        endpoints=info_dict.get("endpoints", []),
                        dependencies=info_dict.get("dependencies", [])
                    )
                    info.status = ServiceStatus(info_dict.get("status", "stopped"))
                    info.start_time = info_dict.get("start_time")
                    info.health_score = info_dict.get("health_score", 0.0)
                    info.error_count = info_dict.get("error_count", 0)
                    info.last_error = info_dict.get("last_error")
                    info.metadata = info_dict.get("metadata", {})
                    self._services[name] = info
                    self._type_index[info.type.value].append(name)
        except Exception:
            pass


def get_service_registry() -> ServiceRegistry:
    """Get singleton service registry instance."""
    if not hasattr(get_service_registry, '_instance'):
        get_service_registry._instance = ServiceRegistry()
    return get_service_registry._instance