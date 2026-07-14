"""
SOUL 灵魂协议包
实现「一个文件复刻一个完整智能体」的标准化智能体定义格式
"""
from .soul_parser import SOULParser
from .soul_model import SOULModel, IdentityAnchor, GoalTree, BehaviorBoundary, PermissionWhitelist, VersionInfo, \
    PersonalityDimension, GoalLevel, PermissionType, ResourceType

__all__ = ["SOULParser", "SOULModel", "IdentityAnchor", "GoalTree", "BehaviorBoundary", "PermissionWhitelist", 
           "VersionInfo", "PersonalityDimension", "GoalLevel", "PermissionType", "ResourceType"]
