"""
Skills 技能库系统
基于 SkillHub 商店的技能管理模块
"""
from .skills_manager import SkillsManager
from .windows_skills import WindowsSkills, get_windows_skills, execute_windows_skill

__all__ = ["SkillsManager", "WindowsSkills", "get_windows_skills", "execute_windows_skill"]
