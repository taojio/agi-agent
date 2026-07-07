"""
soul_model.py - SOUL 数据模型

五大标准字段模块：
1. IdentityAnchor - 身份锚点：名称、人设定位、沟通风格、人格参数
2. GoalTree - 目标树：核心使命、阶段目标、执行准则
3. BehaviorBoundary - 行为边界：禁止事项、伦理底线、安全红线（不可变）
4. PermissionWhitelist - 权限白名单：资源范围、操作类型、最小权限原则
5. VersionInfo - 版本管理：语义化版本号、变更记录、导出分享
"""
import time
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PersonalityDimension(Enum):
    """人格维度枚举"""
    RIGOROUSNESS = "rigorousness"
    CREATIVITY = "creativity"
    DECISION_TENDENCY = "decision_tendency"
    EMPATHY = "empathy"
    ASSERTIVENESS = "assertiveness"
    CURIOSITY = "curiosity"


class GoalLevel(Enum):
    """目标层级"""
    MISSION = "mission"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"


class PermissionType(Enum):
    """权限类型"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(Enum):
    """资源类型"""
    MEMORY = "memory"
    SKILL = "skill"
    PLUGIN = "plugin"
    FILE = "file"
    NETWORK = "network"
    DATABASE = "database"


@dataclass
class IdentityAnchor:
    """身份锚点模块"""
    name: str = ""
    persona: str = ""
    communication_style: str = ""
    role_boundary: str = ""
    personality: Dict[PersonalityDimension, int] = field(default_factory=dict)
    avatar_url: str = ""
    language: str = "zh"

    def __post_init__(self):
        self.personality.setdefault(PersonalityDimension.RIGOROUSNESS, 50)
        self.personality.setdefault(PersonalityDimension.CREATIVITY, 50)
        self.personality.setdefault(PersonalityDimension.DECISION_TENDENCY, 50)
        self.personality.setdefault(PersonalityDimension.EMPATHY, 50)
        self.personality.setdefault(PersonalityDimension.ASSERTIVENESS, 50)
        self.personality.setdefault(PersonalityDimension.CURIOSITY, 50)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "persona": self.persona,
            "communication_style": self.communication_style,
            "role_boundary": self.role_boundary,
            "personality": {k.value: v for k, v in self.personality.items()},
            "avatar_url": self.avatar_url,
            "language": self.language
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityAnchor":
        personality = {}
        for k, v in data.get("personality", {}).items():
            try:
                personality[PersonalityDimension(k)] = v
            except ValueError:
                pass
        return cls(
            name=data.get("name", ""),
            persona=data.get("persona", ""),
            communication_style=data.get("communication_style", ""),
            role_boundary=data.get("role_boundary", ""),
            personality=personality,
            avatar_url=data.get("avatar_url", ""),
            language=data.get("language", "zh")
        )


@dataclass
class GoalNode:
    """目标节点"""
    goal_id: str
    title: str
    description: str
    level: GoalLevel
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    priority: int = 5
    deadline: Optional[float] = None
    status: str = "pending"
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "priority": self.priority,
            "deadline": self.deadline,
            "status": self.status,
            "metrics": self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalNode":
        return cls(
            goal_id=data["goal_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            level=GoalLevel(data.get("level", "operational")),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            priority=data.get("priority", 5),
            deadline=data.get("deadline"),
            status=data.get("status", "pending"),
            metrics=data.get("metrics", {})
        )


@dataclass
class GoalTree:
    """目标树模块"""
    mission: str = ""
    strategic_goals: List[str] = field(default_factory=list)
    _nodes: Dict[str, GoalNode] = field(default_factory=dict)

    @property
    def nodes(self) -> Dict[str, GoalNode]:
        return self._nodes

    def add_node(self, title: str, level: GoalLevel, description: str = "",
                 parent_id: str = None, priority: int = 5) -> GoalNode:
        """添加目标节点"""
        node = GoalNode(
            goal_id=f"goal_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            level=level,
            parent_id=parent_id,
            priority=priority
        )
        self._nodes[node.goal_id] = node

        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children.append(node.goal_id)

        if level == GoalLevel.STRATEGIC and parent_id is None:
            self.strategic_goals.append(node.goal_id)

        return node

    def get_mission(self) -> str:
        """获取核心使命"""
        return self.mission

    def get_tree(self) -> Dict[str, Any]:
        """获取目标树结构"""
        result = {"mission": self.mission, "nodes": {}}
        for goal_id, node in self._nodes.items():
            result["nodes"][goal_id] = node.to_dict()
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": self.mission,
            "strategic_goals": self.strategic_goals,
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalTree":
        tree = cls(mission=data.get("mission", ""))
        tree.strategic_goals = data.get("strategic_goals", [])
        for goal_id, node_data in data.get("nodes", {}).items():
            node = GoalNode.from_dict(node_data)
            tree._nodes[goal_id] = node
        return tree


@dataclass
class BehaviorRule:
    """行为规则"""
    rule_id: str
    description: str
    type: str = "allow"
    is_frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "type": self.type,
            "is_frozen": self.is_frozen
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorRule":
        return cls(
            rule_id=data["rule_id"],
            description=data.get("description", ""),
            type=data.get("type", "allow"),
            is_frozen=data.get("is_frozen", False)
        )


@dataclass
class BehaviorBoundary:
    """行为边界模块"""
    forbidden_actions: List[BehaviorRule] = field(default_factory=list)
    ethical_principles: List[str] = field(default_factory=list)
    safety_redlines: List[str] = field(default_factory=list)
    _frozen: bool = True

    def add_forbidden_action(self, description: str) -> BehaviorRule:
        """添加禁止事项（不可修改）"""
        if self._frozen:
            raise ValueError("行为边界已冻结，不可修改")
        rule = BehaviorRule(
            rule_id=f"forbid_{uuid.uuid4().hex[:8]}",
            description=description,
            type="forbid",
            is_frozen=True
        )
        self.forbidden_actions.append(rule)
        return rule

    def add_ethical_principle(self, principle: str):
        """添加伦理原则（不可修改）"""
        if self._frozen:
            raise ValueError("行为边界已冻结，不可修改")
        self.ethical_principles.append(principle)

    def add_safety_redline(self, redline: str):
        """添加安全红线（不可修改）"""
        if self._frozen:
            raise ValueError("行为边界已冻结，不可修改")
        self.safety_redlines.append(redline)

    def freeze(self):
        """冻结行为边界"""
        self._frozen = True

    def unfreeze(self):
        """解冻行为边界（仅用户可操作）"""
        self._frozen = False

    def is_frozen(self) -> bool:
        """检查是否已冻结"""
        return self._frozen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forbidden_actions": [r.to_dict() for r in self.forbidden_actions],
            "ethical_principles": self.ethical_principles,
            "safety_redlines": self.safety_redlines,
            "is_frozen": self._frozen
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorBoundary":
        boundary = cls()
        for rule_data in data.get("forbidden_actions", []):
            boundary.forbidden_actions.append(BehaviorRule.from_dict(rule_data))
        boundary.ethical_principles = data.get("ethical_principles", [])
        boundary.safety_redlines = data.get("safety_redlines", [])
        boundary._frozen = data.get("is_frozen", True)
        return boundary


@dataclass
class PermissionEntry:
    """权限条目"""
    resource: str
    resource_type: ResourceType
    permissions: List[PermissionType] = field(default_factory=list)
    scope: str = "*"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "resource_type": self.resource_type.value,
            "permissions": [p.value for p in self.permissions],
            "scope": self.scope
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionEntry":
        permissions = []
        for p in data.get("permissions", []):
            try:
                permissions.append(PermissionType(p))
            except ValueError:
                pass
        return cls(
            resource=data["resource"],
            resource_type=ResourceType(data.get("resource_type", "file")),
            permissions=permissions,
            scope=data.get("scope", "*")
        )


@dataclass
class PermissionWhitelist:
    """权限白名单模块"""
    entries: List[PermissionEntry] = field(default_factory=list)

    def add_permission(self, resource: str, resource_type: ResourceType,
                       permissions: List[PermissionType], scope: str = "*"):
        """添加权限"""
        entry = PermissionEntry(resource, resource_type, permissions, scope)
        self.entries.append(entry)

    def has_permission(self, resource: str, permission_type: PermissionType) -> bool:
        """检查是否有指定权限"""
        for entry in self.entries:
            if entry.resource == resource and permission_type in entry.permissions:
                return True
            if entry.resource == "*" and permission_type in entry.permissions:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionWhitelist":
        whitelist = cls()
        for entry_data in data.get("entries", []):
            whitelist.entries.append(PermissionEntry.from_dict(entry_data))
        return whitelist


@dataclass
class VersionInfo:
    """版本管理机制"""
    version: str = "v1.0.0"
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    changelog: List[Dict[str, Any]] = field(default_factory=list)
    author: str = ""
    tags: List[str] = field(default_factory=list)

    def bump_version(self, level: str = "patch"):
        """版本号升级"""
        major, minor, patch = self._parse_version()
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        self.version = f"v{major}.{minor}.{patch}"
        self.modified_at = time.time()

    def _parse_version(self):
        """解析版本号"""
        v = self.version.lstrip("v")
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    def add_change(self, description: str, author: str = ""):
        """添加变更记录"""
        self.changelog.append({
            "timestamp": time.time(),
            "description": description,
            "author": author or self.author,
            "version": self.version
        })
        self.modified_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "changelog": self.changelog,
            "author": self.author,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionInfo":
        return cls(
            version=data.get("version", "v1.0.0"),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time()),
            changelog=data.get("changelog", []),
            author=data.get("author", ""),
            tags=data.get("tags", [])
        )


@dataclass
class SOULModel:
    """SOUL 灵魂协议模型"""
    soul_id: str = field(default_factory=lambda: f"soul_{uuid.uuid4().hex[:8]}")
    identity: IdentityAnchor = field(default_factory=IdentityAnchor)
    goals: GoalTree = field(default_factory=GoalTree)
    boundaries: BehaviorBoundary = field(default_factory=BehaviorBoundary)
    permissions: PermissionWhitelist = field(default_factory=PermissionWhitelist)
    version: VersionInfo = field(default_factory=VersionInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soul_id": self.soul_id,
            "identity": self.identity.to_dict(),
            "goals": self.goals.to_dict(),
            "boundaries": self.boundaries.to_dict(),
            "permissions": self.permissions.to_dict(),
            "version": self.version.to_dict()
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """转换为 SOUL.md 格式"""
        md = f"# {self.identity.name}\n\n"
        md += f"> **SOUL ID**: {self.soul_id}\n"
        md += f"> **Version**: {self.version.version}\n\n"

        md += "## 一、身份锚点\n\n"
        md += f"- **人设定位**: {self.identity.persona}\n"
        md += f"- **沟通风格**: {self.identity.communication_style}\n"
        md += f"- **角色边界**: {self.identity.role_boundary}\n"
        md += "\n### 人格参数\n"
        for dim, value in self.identity.personality.items():
            md += f"- {dim.value}: {value}/100\n"

        md += "\n## 二、目标树\n\n"
        md += f"### 核心使命\n{self.goals.mission}\n\n"
        if self.goals.strategic_goals:
            md += "### 战略目标\n"
            for gid in self.goals.strategic_goals[:5]:
                node = self.goals.nodes.get(gid)
                if node:
                    md += f"- [{node.title}]({node.description[:30]}...)\n"

        md += "\n## 三、行为边界\n\n"
        md += "### 禁止事项\n"
        for rule in self.boundaries.forbidden_actions:
            md += f"- ❌ {rule.description}\n"
        md += "\n### 伦理原则\n"
        for principle in self.boundaries.ethical_principles:
            md += f"- ⚖️ {principle}\n"
        md += "\n### 安全红线\n"
        for redline in self.boundaries.safety_redlines:
            md += f"- 🚫 {redline}\n"

        md += "\n## 四、权限白名单\n\n"
        for entry in self.permissions.entries:
            perms = ", ".join([p.value for p in entry.permissions])
            md += f"- **{entry.resource_type.value}**: {entry.resource} ({perms})\n"

        md += "\n## 五、版本管理\n\n"
        md += f"- **当前版本**: {self.version.version}\n"
        md += f"- **创建时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.version.created_at))}\n"
        md += f"- **修改时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.version.modified_at))}\n"
        if self.version.changelog:
            md += "\n### 变更记录\n"
            for change in self.version.changelog[:5]:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(change["timestamp"]))
                md += f"- [{time_str}] {change['description']}\n"

        return md

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOULModel":
        return cls(
            soul_id=data.get("soul_id", f"soul_{uuid.uuid4().hex[:8]}"),
            identity=IdentityAnchor.from_dict(data.get("identity", {})),
            goals=GoalTree.from_dict(data.get("goals", {})),
            boundaries=BehaviorBoundary.from_dict(data.get("boundaries", {})),
            permissions=PermissionWhitelist.from_dict(data.get("permissions", {})),
            version=VersionInfo.from_dict(data.get("version", {}))
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SOULModel":
        """从 JSON 字符串加载"""
        return cls.from_dict(json.loads(json_str))
