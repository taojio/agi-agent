"""
soul_parser.py - SOUL.md 解析器

解析 SOUL.md 格式文件，将其转换为 SOULModel 对象。
支持从文件读取、字符串解析、导出分享等操作。
"""
import os
import re
import json
import time
from typing import Dict, Any, Optional
from .soul_model import SOULModel, IdentityAnchor, GoalTree, BehaviorBoundary, PermissionWhitelist, VersionInfo, \
    PersonalityDimension, GoalLevel, PermissionType, ResourceType, GoalNode, BehaviorRule, PermissionEntry


class SOULParser:
    """SOUL.md 解析器"""

    def __init__(self):
        self._section_patterns = {
            "identity": re.compile(r"##\s*[一1]\s*[\u3001、]\s*身份锚点", re.IGNORECASE),
            "goals": re.compile(r"##\s*[二2]\s*[\u3001、]\s*目标树", re.IGNORECASE),
            "boundaries": re.compile(r"##\s*[三3]\s*[\u3001、]\s*行为边界", re.IGNORECASE),
            "permissions": re.compile(r"##\s*[四4]\s*[\u3001、]\s*权限白名单", re.IGNORECASE),
            "version": re.compile(r"##\s*[五5]\s*[\u3001、]\s*版本管理", re.IGNORECASE)
        }

    def parse_file(self, file_path: str) -> SOULModel:
        """
        从文件解析 SOUL.md

        Args:
            file_path: SOUL.md 文件路径

        Returns:
            SOULModel 对象
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse(content)

    def parse(self, content: str) -> SOULModel:
        """
        解析 SOUL.md 内容

        Args:
            content: SOUL.md 字符串内容

        Returns:
            SOULModel 对象
        """
        model = SOULModel()

        self._parse_header(content, model)
        self._parse_identity(content, model)
        self._parse_goals(content, model)
        self._parse_boundaries(content, model)
        self._parse_permissions(content, model)
        self._parse_version(content, model)

        return model

    def _parse_header(self, content: str, model: SOULModel):
        """解析文件头信息"""
        lines = content.split("\n")[:10]
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                model.identity.name = line[2:].strip()
            elif line.startswith(">"):
                match = re.search(r"SOUL\s*ID\s*:\s*(\S+)", line)
                if match:
                    model.soul_id = match.group(1)
                match = re.search(r"Version\s*:\s*(\S+)", line)
                if match:
                    model.version.version = match.group(1)

    def _parse_identity(self, content: str, model: SOULModel):
        """解析身份锚点模块"""
        section = self._extract_section(content, "identity")
        if not section:
            return

        model.identity.persona = self._extract_field(section, "人设定位", "-")
        model.identity.communication_style = self._extract_field(section, "沟通风格", "-")
        model.identity.role_boundary = self._extract_field(section, "角色边界", "-")

        # 解析人格参数
        personality_section = self._extract_subsection(section, "人格参数")
        if personality_section:
            for line in personality_section.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    parts = line[2:].split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().replace("/100", "")
                        try:
                            dim = PersonalityDimension(key)
                            model.identity.personality[dim] = int(value)
                        except (ValueError, KeyError):
                            pass

    def _parse_goals(self, content: str, model: SOULModel):
        """解析目标树模块"""
        section = self._extract_section(content, "goals")
        if not section:
            return

        mission_section = self._extract_subsection(section, "核心使命")
        if mission_section:
            model.goals.mission = mission_section.strip()

        strategic_section = self._extract_subsection(section, "战略目标")
        if strategic_section:
            for line in strategic_section.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    title = line[2:]
                    if "(" in title:
                        title = title[:title.index("(")].strip()
                    title = title.replace("[", "").replace("]", "")
                    if title:
                        model.goals.add_node(title, GoalLevel.STRATEGIC)

    def _parse_boundaries(self, content: str, model: SOULModel):
        """解析行为边界模块"""
        section = self._extract_section(content, "boundaries")
        if not section:
            return

        model.boundaries.unfreeze()

        forbidden_section = self._extract_subsection(section, "禁止事项")
        if forbidden_section:
            for line in forbidden_section.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("❌"):
                    desc = line.replace("- ", "").replace("❌", "").strip()
                    if desc:
                        model.boundaries.add_forbidden_action(desc)

        ethical_section = self._extract_subsection(section, "伦理原则")
        if ethical_section:
            for line in ethical_section.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("⚖️"):
                    principle = line.replace("- ", "").replace("⚖️", "").strip()
                    if principle:
                        model.boundaries.add_ethical_principle(principle)

        safety_section = self._extract_subsection(section, "安全红线")
        if safety_section:
            for line in safety_section.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("🚫"):
                    redline = line.replace("- ", "").replace("🚫", "").strip()
                    if redline:
                        model.boundaries.add_safety_redline(redline)

        model.boundaries.freeze()

    def _parse_permissions(self, content: str, model: SOULModel):
        """解析权限白名单模块"""
        section = self._extract_section(content, "permissions")
        if not section:
            return

        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
                match = re.match(r"\*\*(.+?)\*\*:\s*(.+?)\s*\((.+?)\)", line)
                if match:
                    resource_type_str = match.group(1)
                    resource = match.group(2)
                    perms_str = match.group(3)

                    try:
                        resource_type = ResourceType(resource_type_str.lower())
                    except ValueError:
                        resource_type = ResourceType.FILE

                    permissions = []
                    for p in perms_str.split(","):
                        p = p.strip().lower()
                        try:
                            permissions.append(PermissionType(p))
                        except ValueError:
                            pass

                    model.permissions.add_permission(resource, resource_type, permissions)

    def _parse_version(self, content: str, model: SOULModel):
        """解析版本管理模块"""
        section = self._extract_section(content, "version")
        if not section:
            return

        model.version.version = self._extract_field(section, "当前版本", "-")
        created_str = self._extract_field(section, "创建时间", "-")
        modified_str = self._extract_field(section, "修改时间", "-")

        if created_str:
            try:
                model.version.created_at = time.mktime(time.strptime(created_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass

        if modified_str:
            try:
                model.version.modified_at = time.mktime(time.strptime(modified_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass

        changelog_section = self._extract_subsection(section, "变更记录")
        if changelog_section:
            for line in changelog_section.split("\n"):
                line = line.strip()
                if line.startswith("- ["):
                    match = re.match(r"-\s*\[(.+?)\]\s*(.+)", line)
                    if match:
                        time_str = match.group(1)
                        desc = match.group(2)
                        try:
                            ts = time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))
                            model.version.add_change(desc)
                        except ValueError:
                            pass

    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """提取指定章节内容"""
        pattern = self._section_patterns.get(section_name)
        if not pattern:
            return None

        matches = list(pattern.finditer(content))
        if not matches:
            return None

        start = matches[0].end()
        end = len(content)

        for other_name, other_pattern in self._section_patterns.items():
            if other_name == section_name:
                continue
            other_matches = list(other_pattern.finditer(content))
            for m in other_matches:
                if m.start() > start and m.start() < end:
                    end = m.start()

        return content[start:end].strip()

    def _extract_subsection(self, content: str, subsection_name: str) -> Optional[str]:
        """提取子章节内容"""
        pattern = re.compile(r"###\s*" + re.escape(subsection_name), re.IGNORECASE)
        matches = list(pattern.finditer(content))
        if not matches:
            return None

        start = matches[0].end()
        end = len(content)

        other_pattern = re.compile(r"###\s*")
        other_matches = list(other_pattern.finditer(content))
        for m in other_matches:
            if m.start() > start and m.start() < end:
                end = m.start()
                break

        return content[start:end].strip()

    def _extract_field(self, content: str, field_name: str, prefix: str = "- ") -> str:
        """提取字段值"""
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith(prefix):
                remaining = line[len(prefix):]
                if remaining.startswith(field_name):
                    value = remaining[len(field_name):].strip().lstrip(":").strip()
                    return value
        return ""

    def save_to_file(self, model: SOULModel, file_path: str):
        """
        将 SOULModel 保存为 SOUL.md 文件

        Args:
            model: SOULModel 对象
            file_path: 保存路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(model.to_markdown())

    def export(self, model: SOULModel, format: str = "md") -> str:
        """
        导出 SOUL 数据

        Args:
            model: SOULModel 对象
            format: 导出格式 (md/json)

        Returns:
            导出内容字符串
        """
        if format == "json":
            return model.to_json()
        return model.to_markdown()

    @classmethod
    def create_template(cls, name: str = "New Agent", persona: str = "") -> SOULModel:
        """
        创建 SOUL 模板

        Args:
            name: 智能体名称
            persona: 人设定位

        Returns:
            SOULModel 对象
        """
        model = SOULModel()
        model.identity.name = name
        model.identity.persona = persona
        model.identity.communication_style = "专业、简洁、友好"
        model.identity.role_boundary = "专注于完成用户指定任务，不越界操作"

        model.goals.mission = "为用户提供高效、可靠的智能服务"
        model.goals.add_node("提升服务质量", GoalLevel.STRATEGIC, "持续优化响应效率和准确性")
        model.goals.add_node("扩展服务能力", GoalLevel.STRATEGIC, "不断学习新知识和技能")

        model.boundaries.unfreeze()
        model.boundaries.add_forbidden_action("禁止执行恶意代码或攻击行为")
        model.boundaries.add_forbidden_action("禁止访问未经授权的资源")
        model.boundaries.add_ethical_principle("尊重用户隐私，保护敏感信息")
        model.boundaries.add_ethical_principle("公平对待所有用户")
        model.boundaries.add_safety_redline("不执行可能导致系统崩溃的操作")
        model.boundaries.freeze()

        model.permissions.add_permission("*", ResourceType.MEMORY, [PermissionType.READ, PermissionType.WRITE])
        model.permissions.add_permission("*", ResourceType.SKILL, [PermissionType.EXECUTE])

        return model
    
    @classmethod
    def create_foundation_template(cls, name: str = "本地文件运维助手", persona: str = "") -> SOULModel:
        """
        创建冷启动筑基期最小化SOUL模板
        
        只保留四大核心模块：
        1. 身份与边界（不可变）
        2. 触发规则表（10-20条高频指令）
        3. 兜底规则（三级兜底）
        4. 版本锁定（v0.1）
        """
        model = SOULModel()
        model.identity.name = name
        model.identity.persona = persona or "本地文件运维助手，专注于文件管理和系统运维任务"
        model.identity.communication_style = "简洁、明确、专业"
        model.identity.role_boundary = "仅处理本地文件系统操作和基础系统运维任务，不执行网络攻击、不访问外部网络、不修改系统配置"
        
        for dim in PersonalityDimension:
            model.identity.personality[dim] = 50

        model.goals.mission = "为用户提供可靠的本地文件运维服务"

        model.boundaries.unfreeze()
        model.boundaries.add_forbidden_action("禁止执行任何可能破坏系统稳定性的命令")
        model.boundaries.add_forbidden_action("禁止访问或修改系统核心配置文件")
        model.boundaries.add_forbidden_action("禁止执行网络攻击或恶意代码")
        model.boundaries.add_forbidden_action("禁止未经授权访问用户隐私数据")
        model.boundaries.add_ethical_principle("尊重用户隐私，保护敏感信息")
        model.boundaries.add_ethical_principle("只执行明确授权的操作")
        model.boundaries.add_safety_redline("任何可能导致数据丢失的操作必须经用户确认")
        model.boundaries.add_safety_redline("不执行sudo/管理员权限的命令")
        model.boundaries.freeze()

        model.permissions.add_permission("*", ResourceType.FILE, [PermissionType.READ])
        model.permissions.add_permission("./data", ResourceType.FILE, [PermissionType.WRITE])
        model.permissions.add_permission("*", ResourceType.MEMORY, [PermissionType.READ, PermissionType.WRITE])
        model.permissions.add_permission("core", ResourceType.SKILL, [PermissionType.EXECUTE])

        model.version.version = "v0.1"
        model.version.add_change("初始化冷启动筑基期SOUL配置", author="system")

        return model
