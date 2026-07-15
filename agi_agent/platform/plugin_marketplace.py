import os
import json
import time
import hashlib
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import defaultdict


class PluginVersion:
    def __init__(self, version: str, url: str, 
                 file_hash: str = "", dependencies: List[str] = None,
                 compatible_system_versions: List[str] = None):
        self.version = version
        self.url = url
        self.file_hash = file_hash
        self.dependencies = dependencies or []
        self.compatible_system_versions = compatible_system_versions or ["1.0.0"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "url": self.url,
            "file_hash": self.file_hash,
            "dependencies": self.dependencies,
            "compatible_system_versions": self.compatible_system_versions
        }


class PluginInfo:
    def __init__(self, name: str, description: str = "", 
                 author: str = "", plugin_type: str = "utility",
                 versions: List[PluginVersion] = None, tags: List[str] = None):
        self.name = name
        self.description = description
        self.author = author
        self.type = plugin_type
        self.versions = versions or []
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "type": self.type,
            "versions": [v.to_dict() for v in self.versions],
            "tags": self.tags
        }
    
    def get_latest_version(self) -> Optional[PluginVersion]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)


class PluginMarketplace:
    def __init__(self, plugins_dir: str = None, 
                 marketplace_url: str = None):
        self._plugins_dir = plugins_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "plugins", "mods"
        )
        self._marketplace_url = marketplace_url
        self._available_plugins: Dict[str, PluginInfo] = {}
        self._installed_plugins: Dict[str, PluginInfo] = {}
        self._lock = threading.RLock()
        
        os.makedirs(self._plugins_dir, exist_ok=True)
        self._scan_installed_plugins()
    
    def _scan_installed_plugins(self):
        if not os.path.exists(self._plugins_dir):
            return
        
        for filename in os.listdir(self._plugins_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                name = os.path.splitext(filename)[0]
                filepath = os.path.join(self._plugins_dir, filename)
                
                try:
                    plugin_info = self._extract_plugin_info(filepath)
                    if plugin_info:
                        self._installed_plugins[name] = plugin_info
                except Exception:
                    pass
    
    def _extract_plugin_info(self, filepath: str) -> Optional[PluginInfo]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            name = os.path.splitext(os.path.basename(filepath))[0]
            description = ""
            author = ""
            
            if '"""description' in content:
                desc_start = content.find('"""description') + len('"""description')
                desc_end = content.find('"""', desc_start)
                if desc_end > desc_start:
                    description = content[desc_start:desc_end].strip()
            
            if '__author__' in content:
                import re
                match = re.search(r"__author__\s*=\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    author = match.group(1)
            
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            return PluginInfo(
                name=name,
                description=description,
                author=author,
                versions=[PluginVersion(
                    version="1.0.0",
                    url=f"local://{name}",
                    file_hash=file_hash
                )]
            )
        except Exception:
            return None
    
    def load_marketplace_index(self, index_url: str = None) -> bool:
        url = index_url or self._marketplace_url
        if not url:
            return False
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode('utf-8')
                index = json.loads(data)
                
                for plugin_data in index.get("plugins", []):
                    versions = []
                    for ver_data in plugin_data.get("versions", []):
                        versions.append(PluginVersion(
                            version=ver_data.get("version", "1.0.0"),
                            url=ver_data.get("url", ""),
                            file_hash=ver_data.get("file_hash", ""),
                            dependencies=ver_data.get("dependencies", []),
                            compatible_system_versions=ver_data.get("compatible_versions", ["1.0.0"])
                        ))
                    
                    plugin_info = PluginInfo(
                        name=plugin_data.get("name", ""),
                        description=plugin_data.get("description", ""),
                        author=plugin_data.get("author", ""),
                        plugin_type=plugin_data.get("type", "utility"),
                        versions=versions,
                        tags=plugin_data.get("tags", [])
                    )
                    
                    self._available_plugins[plugin_info.name] = plugin_info
                
                return True
        except Exception:
            return False
    
    def search_plugins(self, query: str = "", tags: List[str] = None, 
                       plugin_type: str = None) -> List[PluginInfo]:
        results = []
        query_lower = query.lower() if query else ""
        
        for plugin in self._available_plugins.values():
            match = True
            
            if query and query_lower not in plugin.name.lower() and \
               query_lower not in plugin.description.lower():
                match = False
            
            if tags:
                tag_set = set(t.lower() for t in tags)
                plugin_tag_set = set(t.lower() for t in plugin.tags)
                if not tag_set.intersection(plugin_tag_set):
                    match = False
            
            if plugin_type and plugin.type != plugin_type:
                match = False
            
            if match:
                results.append(plugin)
        
        return results
    
    def install_plugin(self, plugin_name: str, version: str = None) -> Dict[str, Any]:
        with self._lock:
            if plugin_name not in self._available_plugins:
                return {"success": False, "error": f"Plugin not found: {plugin_name}"}
            
            plugin_info = self._available_plugins[plugin_name]
            
            if version:
                target_version = next((v for v in plugin_info.versions if v.version == version), None)
            else:
                target_version = plugin_info.get_latest_version()
            
            if not target_version:
                return {"success": False, "error": f"Version not found: {version}"}
            
            try:
                with urllib.request.urlopen(target_version.url, timeout=30) as response:
                    content = response.read()
                
                if target_version.file_hash:
                    computed_hash = hashlib.md5(content).hexdigest()
                    if computed_hash != target_version.file_hash:
                        return {"success": False, "error": "File hash mismatch"}
                
                filepath = os.path.join(self._plugins_dir, f"{plugin_name}.py")
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                self._installed_plugins[plugin_name] = plugin_info
                
                return {
                    "success": True,
                    "name": plugin_name,
                    "version": target_version.version,
                    "filepath": filepath
                }
            except urllib.error.URLError as e:
                return {"success": False, "error": f"Download failed: {str(e)}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    def uninstall_plugin(self, plugin_name: str) -> Dict[str, Any]:
        with self._lock:
            if plugin_name not in self._installed_plugins:
                return {"success": False, "error": f"Plugin not installed: {plugin_name}"}
            
            filepath = os.path.join(self._plugins_dir, f"{plugin_name}.py")
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            self._installed_plugins.pop(plugin_name, None)
            
            return {"success": True, "name": plugin_name}
    
    def get_available_plugins(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._available_plugins.values()]
    
    def get_installed_plugins(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._installed_plugins.values()]
    
    def get_marketplace_status(self) -> Dict[str, Any]:
        return {
            "available_plugins": len(self._available_plugins),
            "installed_plugins": len(self._installed_plugins),
            "plugins_dir": self._plugins_dir,
            "marketplace_url": self._marketplace_url
        }


def get_plugin_marketplace() -> PluginMarketplace:
    if not hasattr(get_plugin_marketplace, '_instance'):
        get_plugin_marketplace._instance = PluginMarketplace()
    return get_plugin_marketplace._instance