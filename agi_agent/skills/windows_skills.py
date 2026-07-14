"""
Windows系统技能模块

提供智能体与Windows操作系统交互的技能：
- 命令执行
- 文件操作
- 进程管理
- 应用程序控制
- 浏览器操作
"""
import os
import subprocess
import sys
import json
from typing import Dict, Any, Optional, List


class WindowsSkills:
    """Windows系统技能集合"""

    def __init__(self):
        self._name = "windows_system"
        self._description = "Windows系统操作技能 - 提供命令执行、文件操作、进程管理、应用程序启动和浏览器控制功能"
        self._version = "1.0.0"

    def get_info(self) -> Dict[str, Any]:
        """获取技能信息"""
        return {
            "name": self._name,
            "description": self._description,
            "version": self._version,
            "skills": [
                {"name": "execute_command", "description": "执行CMD/PowerShell命令"},
                {"name": "run_python", "description": "执行Python脚本"},
                {"name": "create_file", "description": "创建文件"},
                {"name": "read_file", "description": "读取文件内容"},
                {"name": "delete_file", "description": "删除文件或目录"},
                {"name": "list_directory", "description": "列出目录内容"},
                {"name": "start_process", "description": "启动进程"},
                {"name": "stop_process", "description": "停止进程"},
                {"name": "get_process_list", "description": "获取进程列表"},
                {"name": "open_application", "description": "打开应用程序"},
                {"name": "open_browser", "description": "打开浏览器访问URL"},
                {"name": "search_web", "description": "在浏览器中搜索"},
                {"name": "get_system_info", "description": "获取系统信息"},
                {"name": "get_network_info", "description": "获取网络信息"}
            ]
        }

    def execute_command(self, command: str, shell_type: str = "cmd", timeout: int = 60) -> Dict[str, Any]:
        """执行命令"""
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
            
            return {
                "success": result.returncode == 0,
                "command": command,
                "shell_type": shell_type,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_python(self, script: str) -> Dict[str, Any]:
        """执行Python脚本"""
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
        """创建文件"""
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
        """读取文件内容"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {"success": True, "path": path, "content": content, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_file(self, path: str) -> Dict[str, Any]:
        """删除文件或目录"""
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
        """列出目录内容"""
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
        """启动进程"""
        try:
            process = subprocess.Popen(
                [executable] + (args or []),
                cwd=working_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            return {
                "success": True,
                "pid": process.pid,
                "executable": executable,
                "message": "Process started successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_process(self, pid: int) -> Dict[str, Any]:
        """停止进程"""
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
            
            return {"success": True, "pid": pid, "message": "Process terminated"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_process_list(self) -> Dict[str, Any]:
        """获取进程列表"""
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
        """打开应用程序"""
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
        """打开浏览器访问URL"""
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
        """在浏览器中搜索"""
        try:
            encoded_query = query.replace(" ", "+")
            url = f"https://www.bing.com/search?q={encoded_query}"
            return self.open_browser(url, browser)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            result = subprocess.check_output(["ver"], shell=True, text=True, stderr=subprocess.STDOUT)
            windows_version = result.strip()
            
            return {
                "success": True,
                "info": {
                    "os": os.name,
                    "platform": sys.platform,
                    "windows_version": windows_version,
                    "username": os.getlogin(),
                    "current_directory": os.getcwd(),
                    "cpu_count": os.cpu_count(),
                    "python_version": sys.version
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        try:
            result = subprocess.run(
                ["ipconfig", "/all"],
                capture_output=True,
                text=True
            )
            
            return {"success": True, "output": result.stdout}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_skill(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        """执行指定技能"""
        skill_map = {
            "execute_command": self.execute_command,
            "run_python": self.run_python,
            "create_file": self.create_file,
            "read_file": self.read_file,
            "delete_file": self.delete_file,
            "list_directory": self.list_directory,
            "start_process": self.start_process,
            "stop_process": self.stop_process,
            "get_process_list": self.get_process_list,
            "open_application": self.open_application,
            "open_browser": self.open_browser,
            "search_web": self.search_web,
            "get_system_info": self.get_system_info,
            "get_network_info": self.get_network_info
        }
        
        if skill_name in skill_map:
            return skill_map[skill_name](**kwargs)
        else:
            return {"success": False, "error": f"Unknown skill: {skill_name}"}


_windows_skills = None


def get_windows_skills() -> WindowsSkills:
    """获取Windows技能实例（单例）"""
    global _windows_skills
    if _windows_skills is None:
        _windows_skills = WindowsSkills()
    return _windows_skills


def execute_windows_skill(skill_name: str, **kwargs) -> Dict[str, Any]:
    """执行Windows技能"""
    return get_windows_skills().execute_skill(skill_name, **kwargs)
