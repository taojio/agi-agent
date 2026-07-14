"""
SkillsManager - SkillHub 技能库管理器

封装 skillhub CLI，提供技能搜索、安装、卸载、列表等功能。
在 Windows 上通过 Git Bash 调用 skillhub；在 Linux/macOS 上直接调用。
"""
import os
import sys
import json
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


class SkillsManager:
    """SkillHub 技能库管理器"""

    def __init__(self, skills_dir: Optional[str] = None):
        """
        初始化技能管理器。

        Args:
            skills_dir: 技能安装目录。默认为 agi_agent/skills/
        """
        if skills_dir is None:
            skills_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "skills"
            )
        self.skills_dir = os.path.abspath(skills_dir)
        os.makedirs(self.skills_dir, exist_ok=True)

        self._is_windows = platform.system() == "Windows"
        self._bash_path = self._find_bash() if self._is_windows else None
        self._skillhub_path = self._find_skillhub()
        self._cli_version: Optional[str] = None

    # ------------------------------------------------------------------
    # 环境探测
    # ------------------------------------------------------------------

    def _find_bash(self) -> Optional[str]:
        """在 Windows 上查找 Git Bash 路径"""
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        # 尝试从 PATH 查找
        found = shutil.which("bash")
        return found if found else None

    def _find_skillhub(self) -> Optional[str]:
        """查找 skillhub 可执行路径"""
        if self._is_windows:
            # Windows: 检查 ~/.local/bin/skillhub (bash 脚本)
            home = os.path.expanduser("~")
            wrapper = os.path.join(home, ".local", "bin", "skillhub")
            if os.path.isfile(wrapper):
                return wrapper
        else:
            found = shutil.which("skillhub")
            if found:
                return found
            home = os.path.expanduser("~")
            wrapper = os.path.join(home, ".local", "bin", "skillhub")
            if os.path.isfile(wrapper):
                return wrapper
        return None

    def _to_unix_path(self, win_path: str) -> str:
        """将 Windows 路径转换为 Git Bash 兼容的 Unix 路径"""
        win_path = win_path.replace("\\", "/")
        if len(win_path) >= 2 and win_path[1] == ":":
            drive = win_path[0].lower()
            win_path = f"/{drive}{win_path[2:]}"
        return win_path

    def _build_command(self, args: List[str]) -> List[str]:
        """
        构建调用 skillhub 的命令列表。

        Windows: 通过 Git Bash 调用
        Linux/macOS: 直接调用
        """
        if not self._skillhub_path:
            raise RuntimeError("skillhub CLI 未安装")

        if self._is_windows and self._bash_path:
            # 构建 bash 内联命令
            unix_skillhub = self._to_unix_path(self._skillhub_path)
            # 确保 PATH 包含 ~/.local/bin（用 ; 分隔 export 和命令）
            home_unix = self._to_unix_path(os.path.expanduser("~"))
            shell_cmd = (
                f'export PATH="{home_unix}/.local/bin:$PATH"; '
                f'"{unix_skillhub}" --skip-self-upgrade '
                + " ".join(self._shell_quote(a) for a in args)
            )
            cmd = [self._bash_path, "-c", shell_cmd]
            return cmd
        else:
            return [self._skillhub_path, "--skip-self-upgrade"] + args

    @staticmethod
    def _shell_quote(s: str) -> str:
        """对 shell 参数进行安全引用"""
        if not s:
            return "''"
        # 简单转义：如果包含特殊字符则用单引号包裹
        if all(c.isalnum() or c in "-_./:" for c in s):
            return s
        escaped = s.replace("'", "'\"'\"'")
        return f"'{escaped}'"

    def _run(self, args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        执行 skillhub 命令并返回结果。

        Returns:
            dict with keys: success, stdout, stderr, returncode
        """
        cmd = self._build_command(args)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip() if proc.stdout else "",
                "stderr": proc.stderr.strip() if proc.stderr else "",
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "命令执行超时", "returncode": -1}
        except FileNotFoundError as e:
            return {"success": False, "stdout": "", "stderr": f"命令未找到: {e}", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """检查 skillhub CLI 是否可用"""
        return self._skillhub_path is not None

    def get_version(self) -> Optional[str]:
        """获取 skillhub CLI 版本"""
        if self._cli_version:
            return self._cli_version
        result = self._run(["--version"], timeout=10)
        if result["success"]:
            # 输出格式: "skillhub 2026.6.27"
            ver = result["stdout"].strip().split()[-1] if result["stdout"] else ""
            self._cli_version = ver
            return ver
        return None

    def get_status(self) -> Dict[str, Any]:
        """获取技能库系统状态"""
        version = self.get_version()
        installed = self.list_installed_skills()
        return {
            "available": self.is_available(),
            "cli_version": version,
            "skills_dir": self.skills_dir,
            "skills_dir_unix": self._to_unix_path(self.skills_dir) if self._is_windows else self.skills_dir,
            "installed_count": len(installed),
            "installed_skills": installed,
            "bash_path": self._bash_path,
            "platform": platform.system(),
        }

    def search(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        在 SkillHub 商店搜索技能。

        Args:
            query: 搜索关键词
            limit: 返回结果数量上限

        Returns:
            dict: {"success": bool, "results": [...], "count": int, "error": str}
        """
        if not self.is_available():
            return {"success": False, "results": [], "count": 0, "error": "skillhub CLI 未安装"}

        query = (query or "").strip()
        if not query:
            return {"success": False, "results": [], "count": 0, "error": "搜索关键词不能为空"}

        args = ["search", query, "--json", "--search-limit", str(limit)]
        result = self._run(args, timeout=30)

        if not result["success"]:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "error": result["stderr"] or "搜索失败",
            }

        try:
            data = json.loads(result["stdout"])
            results = data.get("results", [])
            # 标记是否已安装
            installed_slugs = {s["slug"] for s in self.list_installed_skills()}
            for r in results:
                r["installed"] = r.get("slug") in installed_slugs
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query,
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "results": [],
                "count": 0,
                "error": "解析搜索结果失败",
            }

    def install(self, slug: str, force: bool = False) -> Dict[str, Any]:
        """
        从 SkillHub 商店安装技能到本地 skills 目录。

        Args:
            slug: 技能唯一标识
            force: 是否覆盖已有安装

        Returns:
            dict: {"success": bool, "slug": str, "path": str, "error": str}
        """
        if not self.is_available():
            return {"success": False, "slug": slug, "error": "skillhub CLI 未安装"}

        slug = (slug or "").strip()
        if not slug:
            return {"success": False, "slug": slug, "error": "技能 slug 不能为空"}

        dir_arg = self.skills_dir
        if self._is_windows:
            dir_arg = self._to_unix_path(self.skills_dir)

        args = ["install", slug, "--dir", dir_arg, "--json"]
        if force:
            args.append("--force")

        result = self._run(args, timeout=120)

        if not result["success"]:
            return {
                "success": False,
                "slug": slug,
                "error": result["stderr"] or result["stdout"] or "安装失败",
            }

        # 尝试解析 JSON 结果
        install_path = os.path.join(self.skills_dir, slug)
        try:
            data = json.loads(result["stdout"])
            install_path = data.get("path", install_path)
            return {
                "success": True,
                "slug": slug,
                "path": install_path,
                "detail": data,
            }
        except (json.JSONDecodeError, ValueError):
            # 非 JSON 输出，根据返回码判断成功
            return {
                "success": True,
                "slug": slug,
                "path": install_path,
                "detail": result["stdout"],
            }

    def uninstall(self, slug: str) -> Dict[str, Any]:
        """
        卸载已安装的技能（删除技能目录）。

        Args:
            slug: 技能唯一标识

        Returns:
            dict: {"success": bool, "slug": str, "error": str}
        """
        slug = (slug or "").strip()
        if not slug:
            return {"success": False, "slug": slug, "error": "技能 slug 不能为空"}

        skill_path = os.path.join(self.skills_dir, slug)
        if not os.path.isdir(skill_path):
            return {"success": False, "slug": slug, "error": f"技能未安装: {slug}"}

        try:
            shutil.rmtree(skill_path)
            return {"success": True, "slug": slug, "path": skill_path}
        except PermissionError as e:
            return {"success": False, "slug": slug, "error": f"权限不足，无法删除: {e}"}
        except Exception as e:
            return {"success": False, "slug": slug, "error": str(e)}

    def list_installed_skills(self) -> List[Dict[str, Any]]:
        """
        列出本地已安装的技能。

        扫描 skills 目录，每个子目录包含 SKILL.md 的视为已安装技能。
        """
        skills = []
        if not os.path.isdir(self.skills_dir):
            return skills

        for entry in sorted(os.listdir(self.skills_dir)):
            entry_path = os.path.join(self.skills_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            # 跳过 __pycache__ 等非技能目录
            if entry.startswith("__") or entry.startswith("."):
                continue

            skill_info = self._parse_skill_dir(entry, entry_path)
            skills.append(skill_info)

        return skills

    def _parse_skill_dir(self, slug: str, dir_path: str) -> Dict[str, Any]:
        """解析单个技能目录，提取元信息"""
        info = {
            "slug": slug,
            "name": slug,
            "description": "",
            "version": "",
            "path": dir_path,
            "has_skill_md": False,
            "has_config": False,
        }

        # 解析 SKILL.md
        skill_md = os.path.join(dir_path, "SKILL.md")
        if os.path.isfile(skill_md):
            info["has_skill_md"] = True
            try:
                with open(skill_md, "r", encoding="utf-8") as f:
                    content = f.read()
                # 从 SKILL.md 提取 name 和 description
                lines = content.strip().split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith("# ") and info["name"] == slug:
                        info["name"] = stripped[2:].strip()
                    elif stripped.lower().startswith("description:") or \
                         (stripped.startswith(">") and i < 5):
                        desc = stripped.lstrip("> ").strip()
                        if stripped.lower().startswith("description:"):
                            desc = stripped[len("description:"):].strip()
                        if desc and not info["description"]:
                            info["description"] = desc
            except Exception:
                pass

        # 解析 config.json
        config_path = os.path.join(dir_path, "config.json")
        if os.path.isfile(config_path):
            info["has_config"] = True
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "version" in cfg:
                    info["version"] = cfg["version"]
                if "name" in cfg and info["name"] == slug:
                    info["name"] = cfg["name"]
                if "description" in cfg and not info["description"]:
                    info["description"] = cfg["description"]
            except Exception:
                pass

        return info

    def get_skill_detail(self, slug: str) -> Dict[str, Any]:
        """
        获取已安装技能的详细信息。

        Args:
            slug: 技能唯一标识

        Returns:
            dict: 技能详情
        """
        slug = (slug or "").strip()
        skill_path = os.path.join(self.skills_dir, slug)
        if not os.path.isdir(skill_path):
            return {"success": False, "slug": slug, "error": "技能未安装"}

        info = self._parse_skill_dir(slug, skill_path)

        # 列出技能目录内的文件
        files = []
        for root, dirs, filenames in os.walk(skill_path):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(root, fn), skill_path)
                files.append(rel.replace("\\", "/"))
        info["files"] = sorted(files)
        info["file_count"] = len(files)

        # 读取 SKILL.md 内容
        skill_md = os.path.join(skill_path, "SKILL.md")
        if os.path.isfile(skill_md):
            try:
                with open(skill_md, "r", encoding="utf-8") as f:
                    info["skill_md_content"] = f.read()
            except Exception:
                info["skill_md_content"] = ""

        return {"success": True, **info}
