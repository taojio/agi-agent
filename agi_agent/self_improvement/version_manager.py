import os
import json
import time
import uuid
import shutil
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import deque
from ..utils.logger import setup_logger


class VersionStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    ROLLBACK = "rollback"


class VersionType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    HOTFIX = "hotfix"
    EXPERIMENTAL = "experimental"


@dataclass
class VersionChange:
    change_id: str
    type: str
    description: str
    target_component: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    author: str = "system"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "type": self.type,
            "description": self.description,
            "target_component": self.target_component,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "author": self.author,
            "timestamp": self.timestamp,
        }


@dataclass
class Version:
    version_id: str
    version_number: str
    type: VersionType
    status: VersionStatus
    changes: List[VersionChange] = field(default_factory=list)
    parent_version: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    deactivated_at: Optional[float] = None
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    rollback_target: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "type": self.type.value,
            "status": self.status.value,
            "changes": [c.to_dict() for c in self.changes],
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "deactivated_at": self.deactivated_at,
            "description": self.description,
            "metrics": self.metrics,
            "rollback_target": self.rollback_target,
        }


@dataclass
class VersionComparison:
    source_version: str
    target_version: str
    added_changes: List[VersionChange] = field(default_factory=list)
    removed_changes: List[VersionChange] = field(default_factory=list)
    modified_changes: List[VersionChange] = field(default_factory=list)
    common_changes: List[VersionChange] = field(default_factory=list)
    metrics_difference: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "added_changes": [c.to_dict() for c in self.added_changes],
            "removed_changes": [c.to_dict() for c in self.removed_changes],
            "modified_changes": [c.to_dict() for c in self.modified_changes],
            "common_changes": [c.to_dict() for c in self.common_changes],
            "metrics_difference": self.metrics_difference,
        }


class ImprovementVersionManager:
    def __init__(self, storage_path: str = None):
        self.logger = setup_logger("version_manager")
        self.versions: Dict[str, Version] = {}
        self.version_history: deque = deque(maxlen=100)
        self._current_version_id: Optional[str] = None
        self._rollback_stack: List[str] = []

        if storage_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.storage_path = os.path.join(base_dir, "data", "versions")
        else:
            self.storage_path = storage_path

        os.makedirs(self.storage_path, exist_ok=True)

        self._load_versions()

    def _load_versions(self):
        if os.path.exists(self.storage_path):
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            version = self._dict_to_version(data)
                            self.versions[version.version_id] = version
                            if version.status == VersionStatus.ACTIVE:
                                self._current_version_id = version.version_id
                    except Exception as e:
                        self.logger.warning(f"Error loading version file {filename}: {e}")

        if not self.versions:
            self._create_initial_version()

    def _dict_to_version(self, data: Dict[str, Any]) -> Version:
        changes = []
        for change_data in data.get("changes", []):
            changes.append(VersionChange(
                change_id=change_data["change_id"],
                type=change_data["type"],
                description=change_data["description"],
                target_component=change_data["target_component"],
                old_value=change_data.get("old_value"),
                new_value=change_data.get("new_value"),
                author=change_data.get("author", "system"),
                timestamp=change_data.get("timestamp", time.time()),
            ))

        return Version(
            version_id=data["version_id"],
            version_number=data["version_number"],
            type=VersionType(data["type"]),
            status=VersionStatus(data["status"]),
            changes=changes,
            parent_version=data.get("parent_version"),
            created_at=data.get("created_at", time.time()),
            activated_at=data.get("activated_at"),
            deactivated_at=data.get("deactivated_at"),
            description=data.get("description", ""),
            metrics=data.get("metrics", {}),
            rollback_target=data.get("rollback_target"),
        )

    def _create_initial_version(self):
        initial_version = Version(
            version_id=f"v_{uuid.uuid4().hex[:8]}",
            version_number="1.0.0",
            type=VersionType.MAJOR,
            status=VersionStatus.ACTIVE,
            description="Initial version",
            changes=[VersionChange(
                change_id=f"ch_{uuid.uuid4().hex[:8]}",
                type="initial",
                description="System initialization",
                target_component="system",
            )],
        )

        self.versions[initial_version.version_id] = initial_version
        self._current_version_id = initial_version.version_id
        self._save_version(initial_version)
        self.logger.info("Created initial version: 1.0.0")

    def _save_version(self, version: Version):
        filepath = os.path.join(self.storage_path, f"{version.version_id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving version {version.version_id}: {e}")

    def create_version(self, version_type: VersionType, changes: List[VersionChange],
                       description: str = "", metrics: Dict[str, float] = None) -> Version:
        parent_version = self.get_current_version()
        version_number = self._generate_version_number(parent_version, version_type)

        version = Version(
            version_id=f"v_{uuid.uuid4().hex[:8]}",
            version_number=version_number,
            type=version_type,
            status=VersionStatus.INACTIVE,
            changes=changes,
            parent_version=parent_version.version_id if parent_version else None,
            description=description,
            metrics=metrics or {},
        )

        self.versions[version.version_id] = version
        self._save_version(version)
        self.logger.info(f"Created version: {version_number} ({version.version_id})")

        return version

    def _generate_version_number(self, parent_version: Optional[Version], version_type: VersionType) -> str:
        if not parent_version:
            return "1.0.0"

        parts = parent_version.version_number.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if version_type == VersionType.MAJOR:
            return f"{major + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{major}.{minor + 1}.0"
        elif version_type == VersionType.PATCH or version_type == VersionType.HOTFIX:
            return f"{major}.{minor}.{patch + 1}"
        else:
            return f"{major}.{minor}.{patch}-exp{int(time.time()) % 1000}"

    def activate_version(self, version_id: str) -> bool:
        version = self.versions.get(version_id)
        if not version:
            self.logger.error(f"Version not found: {version_id}")
            return False

        if version.status == VersionStatus.ACTIVE:
            self.logger.warning(f"Version already active: {version_id}")
            return True

        current_version = self.get_current_version()
        if current_version:
            current_version.status = VersionStatus.INACTIVE
            current_version.deactivated_at = time.time()
            self._save_version(current_version)
            self._rollback_stack.append(current_version.version_id)

        version.status = VersionStatus.ACTIVE
        version.activated_at = time.time()
        self._current_version_id = version.version_id
        self._save_version(version)
        self.version_history.append(version)

        self.logger.info(f"Activated version: {version.version_number} ({version_id})")
        return True

    def deactivate_version(self, version_id: str) -> bool:
        version = self.versions.get(version_id)
        if not version:
            self.logger.error(f"Version not found: {version_id}")
            return False

        if version.status != VersionStatus.ACTIVE:
            self.logger.warning(f"Version is not active: {version_id}")
            return True

        version.status = VersionStatus.INACTIVE
        version.deactivated_at = time.time()
        self._save_version(version)

        if self._current_version_id == version_id:
            self._current_version_id = None

        self.logger.info(f"Deactivated version: {version.version_number} ({version_id})")
        return True

    def archive_version(self, version_id: str) -> bool:
        version = self.versions.get(version_id)
        if not version:
            self.logger.error(f"Version not found: {version_id}")
            return False

        if version.status == VersionStatus.ACTIVE:
            self.deactivate_version(version_id)

        version.status = VersionStatus.ARCHIVED
        self._save_version(version)
        self.logger.info(f"Archived version: {version.version_number} ({version_id})")
        return True

    def get_current_version(self) -> Optional[Version]:
        if not self._current_version_id:
            return None
        return self.versions.get(self._current_version_id)

    def get_version(self, version_id: str) -> Optional[Version]:
        return self.versions.get(version_id)

    def get_version_by_number(self, version_number: str) -> Optional[Version]:
        for version in self.versions.values():
            if version.version_number == version_number:
                return version
        return None

    def list_versions(self, status_filter: Optional[VersionStatus] = None) -> List[Version]:
        versions = list(self.versions.values())
        versions.sort(key=lambda v: v.created_at, reverse=True)

        if status_filter:
            return [v for v in versions if v.status == status_filter]
        return versions

    def compare_versions(self, source_version_id: str, target_version_id: str) -> VersionComparison:
        source = self.get_version(source_version_id)
        target = self.get_version(target_version_id)

        if not source or not target:
            return VersionComparison(source_version=source_version_id, target_version=target_version_id)

        source_changes = {c.change_id: c for c in source.changes}
        target_changes = {c.change_id: c for c in target.changes}

        added_changes = [c for c in target.changes if c.change_id not in source_changes]
        removed_changes = [c for c in source.changes if c.change_id not in target_changes]
        modified_changes = []
        common_changes = []

        for change_id, source_change in source_changes.items():
            if change_id in target_changes:
                target_change = target_changes[change_id]
                if source_change.new_value != target_change.new_value:
                    modified_changes.append(target_change)
                else:
                    common_changes.append(source_change)

        metrics_difference = {}
        for metric, value in target.metrics.items():
            source_value = source.metrics.get(metric, 0)
            metrics_difference[metric] = value - source_value

        return VersionComparison(
            source_version=source.version_number,
            target_version=target.version_number,
            added_changes=added_changes,
            removed_changes=removed_changes,
            modified_changes=modified_changes,
            common_changes=common_changes,
            metrics_difference=metrics_difference,
        )

    def rollback_to_version(self, version_id: str) -> bool:
        target_version = self.get_version(version_id)
        if not target_version:
            self.logger.error(f"Rollback target not found: {version_id}")
            return False

        current_version = self.get_current_version()
        if not current_version:
            self.logger.error("No active version to rollback from")
            return False

        rollback_changes = []
        for change in current_version.changes:
            rollback_changes.append(VersionChange(
                change_id=f"rb_{uuid.uuid4().hex[:8]}",
                type="rollback",
                description=f"Rollback from {current_version.version_number}",
                target_component=change.target_component,
                old_value=change.new_value,
                new_value=change.old_value,
                author="system",
            ))

        rollback_version = self.create_version(
            version_type=VersionType.HOTFIX,
            changes=rollback_changes,
            description=f"Rollback to {target_version.version_number}",
            metrics=target_version.metrics,
        )

        rollback_version.rollback_target = version_id
        self._save_version(rollback_version)

        self.activate_version(rollback_version.version_id)

        self.logger.info(f"Rolled back to version: {target_version.version_number}")
        return True

    def quick_switch(self, version_id: str) -> bool:
        version = self.get_version(version_id)
        if not version:
            self.logger.error(f"Version not found: {version_id}")
            return False

        if version.status == VersionStatus.ACTIVE:
            return True

        if version.status == VersionStatus.DEPRECATED:
            self.logger.warning(f"Attempting to switch to deprecated version: {version_id}")

        return self.activate_version(version_id)

    def get_version_history(self, limit: int = 20) -> List[Version]:
        return list(self.version_history)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        total_versions = len(self.versions)
        active_count = sum(1 for v in self.versions.values() if v.status == VersionStatus.ACTIVE)
        inactive_count = sum(1 for v in self.versions.values() if v.status == VersionStatus.INACTIVE)
        archived_count = sum(1 for v in self.versions.values() if v.status == VersionStatus.ARCHIVED)

        type_distribution = {}
        for v in self.versions.values():
            type_distribution[v.type.value] = type_distribution.get(v.type.value, 0) + 1

        return {
            "total_versions": total_versions,
            "active_count": active_count,
            "inactive_count": inactive_count,
            "archived_count": archived_count,
            "type_distribution": type_distribution,
            "current_version": self.get_current_version().version_number if self.get_current_version() else None,
            "rollback_stack_size": len(self._rollback_stack),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "storage_path": self.storage_path,
            **self.get_stats(),
        }


_global_version_manager: Optional[ImprovementVersionManager] = None


def get_version_manager() -> ImprovementVersionManager:
    global _global_version_manager
    if _global_version_manager is None:
        _global_version_manager = ImprovementVersionManager()
    return _global_version_manager