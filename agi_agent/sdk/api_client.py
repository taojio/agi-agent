import json
import requests
from typing import Dict, List, Optional, Any


class AGIAgentAPIClient:
    def __init__(self, base_url: str = "http://localhost:8090", 
                 api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def _request(self, method: str, endpoint: str, 
                 data: Any = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self._session.request(
                method, url, json=data, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e), "status_code": response.status_code if 'response' in dir() else 0}
    
    def health_check(self) -> Dict[str, Any]:
        return self._request("GET", "/health")
    
    def get_agent_status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/agent/status")
    
    def get_memory_tiers(self) -> Dict[str, Any]:
        return self._request("GET", "/api/memory/tiers")
    
    def search_memory(self, query: str, tier: str = None, limit: int = 10) -> Dict[str, Any]:
        params = {"query": query, "limit": limit}
        if tier:
            params["tier"] = tier
        return self._request("GET", "/api/memory/search", params=params)
    
    def add_memory(self, content: str, tier: str = "short_term", 
                   category: str = "default") -> Dict[str, Any]:
        data = {"content": content, "tier": tier, "category": category}
        return self._request("POST", "/api/memory/add", data=data)
    
    def get_soul(self) -> Dict[str, Any]:
        return self._request("GET", "/api/soul")
    
    def update_soul(self, soul_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/soul/update", data=soul_data)
    
    def get_tasks(self, status: str = None) -> Dict[str, Any]:
        params = {}
        if status:
            params["status"] = status
        return self._request("GET", "/api/tasks", params=params)
    
    def create_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/tasks/create", data=task_data)
    
    def get_plugins(self) -> Dict[str, Any]:
        return self._request("GET", "/api/plugins/available")
    
    def load_plugin(self, plugin_name: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/plugins/load?plugin_name={plugin_name}")
    
    def activate_plugin(self, plugin_name: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/plugins/{plugin_name}/activate")
    
    def get_service_registry(self) -> Dict[str, Any]:
        return self._request("GET", "/api/platform/services")
    
    def get_gateway_routes(self) -> Dict[str, Any]:
        return self._request("GET", "/api/platform/routes")
    
    def get_metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/metrics")
    
    def send_message(self, message: str, context: str = "") -> Dict[str, Any]:
        data = {"message": message, "context": context}
        return self._request("POST", "/api/chat/send", data=data)