import os
import subprocess
import sys
import ctypes
import time
import json
from typing import Dict, Any, Optional, List
from ..plugin_base import PeripheralPlugin, PluginStatus, PluginPriority, PluginHookPoint


class WindowsSystemPlugin(PeripheralPlugin):
    """Windows系统控制插件。
    
    提供智能体与Windows操作系统的深度交互能力：
    - 命令执行：执行CMD/PowerShell命令
    - 文件操作：创建、删除、复制、移动文件和目录
    - 进程管理：启动、停止、查询进程
    - 应用程序控制：启动已安装的应用程序
    - 浏览器控制：打开URL、搜索
    - 系统信息：获取系统状态、硬件信息
    """

    def __init__(self):
        super().__init__(
            name="windows_system",
            version="1.0.0",
            description="Windows系统控制插件 - 提供命令执行、文件操作、进程管理、应用程序启动和浏览器控制功能",
            plugin_type="system",
            priority=PluginPriority.HIGH,
            hook_points=[PluginHookPoint.POST_COGNITION, PluginHookPoint.PERIODIC]
        )
        self._last_command_result = None
        self._system_info = {}
        self._running_processes = {}

    def on_load(self) -> bool:
        try:
            self._update_system_info()
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def on_unload(self) -> bool:
        self._running_processes.clear()
        self._system_info.clear()
        return True

    def on_activate(self) -> bool:
        try:
            self._update_system_info()
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def _update_system_info(self):
        try:
            self._system_info = {
                "os": os.name,
                "platform": sys.platform,
                "windows_version": self._get_windows_version(),
                "username": os.getlogin(),
                "current_directory": os.getcwd(),
                "cpu_count": os.cpu_count(),
                "python_version": sys.version,
                "has_admin": self._check_admin_rights()
            }
        except Exception as e:
            self._system_info = {"error": str(e)}

    def _get_windows_version(self) -> str:
        try:
            result = subprocess.check_output(
                ["ver"], shell=True, text=True, stderr=subprocess.STDOUT
            )
            return result.strip()
        except:
            return "Unknown"

    def _check_admin_rights(self) -> bool:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def execute_command(self, command: str, shell_type: str = "cmd", timeout: int = 60) -> Dict[str, Any]:
        """执行命令。
        
        Args:
            command: 要执行的命令
            shell_type: "cmd" 或 "powershell"
            timeout: 超时时间（秒）
        
        Returns:
            包含输出和状态的字典
        """
        try:
            if shell_type.lower() == "powershell":
                cmd = ["powershell", "-Command", command]
            else:
                cmd = ["cmd", "/c", command]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )
            
            self._last_command_result = {
                "success": result.returncode == 0,
                "command": command,
                "shell_type": shell_type,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            return self._last_command_result
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_python_script(self, script: str) -> Dict[str, Any]:
        """执行Python脚本。
        
        Args:
            script: Python代码字符串
        
        Returns:
            包含输出和状态的字典
        """
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_file(self, path: str, content: str = "") -> Dict[str, Any]:
        """创建文件。
        
        Args:
            path: 文件路径
            content: 文件内容
        
        Returns:
            操作结果
        """
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {"success": True, "path": path, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, path: str) -> Dict[str, Any]:
        """读取文件内容。
        
        Args:
            path: 文件路径
        
        Returns:
            包含文件内容的字典
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {"success": True, "path": path, "content": content, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_file(self, path: str) -> Dict[str, Any]:
        """删除文件。
        
        Args:
            path: 文件路径
        
        Returns:
            操作结果
        """
        try:
            if os.path.isfile(path):
                os.remove(path)
                return {"success": True, "path": path}
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
                return {"success": True, "path": path, "is_directory": True}
            else:
                return {"success": False, "error": "Path does not exist"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """列出目录内容。
        
        Args:
            path: 目录路径
        
        Returns:
            包含文件列表的字典
        """
        try:
            files = []
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                is_dir = os.path.isdir(full_path)
                try:
                    size = os.path.getsize(full_path) if not is_dir else 0
                    mtime = os.path.getmtime(full_path)
                except:
                    size = 0
                    mtime = 0
                
                files.append({
                    "name": entry,
                    "path": full_path,
                    "is_directory": is_dir,
                    "size": size,
                    "modified_time": mtime
                })
            
            return {"success": True, "path": path, "files": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_process(self, executable: str, args: Optional[List[str]] = None, 
                     working_dir: Optional[str] = None) -> Dict[str, Any]:
        """启动进程。
        
        Args:
            executable: 可执行文件路径
            args: 命令行参数
            working_dir: 工作目录
        
        Returns:
            包含进程信息的字典
        """
        try:
            process = subprocess.Popen(
                [executable] + (args or []),
                cwd=working_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self._running_processes[process.pid] = {
                "pid": process.pid,
                "executable": executable,
                "args": args,
                "working_dir": working_dir,
                "start_time": time.time()
            }
            
            return {
                "success": True,
                "pid": process.pid,
                "executable": executable,
                "message": "Process started successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_process(self, pid: int) -> Dict[str, Any]:
        """停止进程。
        
        Args:
            pid: 进程ID
        
        Returns:
            操作结果
        """
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
            
            if pid in self._running_processes:
                del self._running_processes[pid]
            
            return {"success": True, "pid": pid, "message": "Process terminated"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_process_list(self) -> Dict[str, Any]:
        """获取进程列表。
        
        Returns:
            包含进程列表的字典
        """
        try:
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True
            )
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        parts = [p.strip('"') for p in line.split('","')]
                        if len(parts) >= 2:
                            processes.append({
                                "name": parts[0],
                                "pid": int(parts[1])
                            })
                    except:
                        pass
            
            return {"success": True, "processes": processes, "count": len(processes)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_application(self, app_name: str) -> Dict[str, Any]:
        """打开应用程序。
        
        Args:
            app_name: 应用程序名称或路径
        
        Returns:
            操作结果
        """
        try:
            known_apps = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "paint": "mspaint.exe",
                "word": "winword.exe",
                "excel": "excel.exe",
                "powerpoint": "powerpnt.exe",
                "chrome": "chrome.exe",
                "edge": "msedge.exe",
                "firefox": "firefox.exe",
                "explorer": "explorer.exe",
                "cmd": "cmd.exe",
                "powershell": "powershell.exe",
                "vscode": "code.exe",
                "sublime": "sublime_text.exe",
                "git": "git-bash.exe"
            }
            
            executable = known_apps.get(app_name.lower(), app_name)
            
            if os.path.exists(executable):
                process = subprocess.Popen(executable)
            else:
                process = subprocess.Popen(executable, shell=True)
            
            return {
                "success": True,
                "app": app_name,
                "executable": executable,
                "pid": process.pid
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_browser(self, url: str, browser: str = "default") -> Dict[str, Any]:
        """打开浏览器访问URL。
        
        Args:
            url: 网址
            browser: 浏览器类型 ("default", "chrome", "edge", "firefox")
        
        Returns:
            操作结果
        """
        try:
            browsers = {
                "chrome": "chrome.exe",
                "edge": "msedge.exe",
                "firefox": "firefox.exe"
            }
            
            if browser == "default" or browser not in browsers:
                os.startfile(url)
            else:
                subprocess.Popen([browsers[browser], url])
            
            return {"success": True, "url": url, "browser": browser}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_web(self, query: str, browser: str = "default") -> Dict[str, Any]:
        """在浏览器中搜索。
        
        Args:
            query: 搜索关键词
            browser: 浏览器类型
        
        Returns:
            操作结果
        """
        try:
            encoded_query = query.replace(" ", "+")
            url = f"https://www.bing.com/search?q={encoded_query}"
            return self.open_browser(url, browser)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息。
        
        Returns:
            系统信息字典
        """
        self._update_system_info()
        return {"success": True, "info": self._system_info}

    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息。
        
        Returns:
            网络信息字典
        """
        try:
            result = subprocess.run(
                ["ipconfig", "/all"],
                capture_output=True,
                text=True
            )
            
            return {"success": True, "output": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process(self, input_data: Any) -> Any:
        self.process_count += 1
        
        if isinstance(input_data, dict) and "action" in input_data:
            action = input_data["action"]
            
            if action == "execute_command":
                return self.execute_command(
                    input_data.get("command", ""),
                    input_data.get("shell_type", "cmd"),
                    input_data.get("timeout", 60)
                )
            elif action == "run_python":
                return self.run_python_script(input_data.get("script", ""))
            elif action == "create_file":
                return self.create_file(
                    input_data.get("path", ""),
                    input_data.get("content", "")
                )
            elif action == "read_file":
                return self.read_file(input_data.get("path", ""))
            elif action == "delete_file":
                return self.delete_file(input_data.get("path", ""))
            elif action == "list_dir":
                return self.list_directory(input_data.get("path", "."))
            elif action == "start_process":
                return self.start_process(
                    input_data.get("executable", ""),
                    input_data.get("args", []),
                    input_data.get("working_dir", None)
                )
            elif action == "stop_process":
                return self.stop_process(input_data.get("pid", 0))
            elif action == "get_processes":
                return self.get_process_list()
            elif action == "open_app":
                return self.open_application(input_data.get("app", ""))
            elif action == "open_browser":
                return self.open_browser(
                    input_data.get("url", ""),
                    input_data.get("browser", "default")
                )
            elif action == "search_web":
                return self.search_web(
                    input_data.get("query", ""),
                    input_data.get("browser", "default")
                )
            elif action == "get_system_info":
                return self.get_system_info()
            elif action == "get_network_info":
                return self.get_network_info()
        
        return input_data

    def get_data(self) -> Dict[str, Any]:
        return {
            "system_info": self._system_info,
            "last_command_result": self._last_command_result,
            "running_processes": self._running_processes,
            "process_count": self.process_count
        }
