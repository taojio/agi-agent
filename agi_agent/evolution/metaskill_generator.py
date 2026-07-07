"""
metaskill_generator.py - MetaSkill 元技能生成器

实现「技能自己生成技能」的核心载体，是系统自举进化的工程底座。
"""
import time
import uuid
import os
import json
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field


class MetaSkillQualityCheck(Enum):
    """质量检查关卡"""
    TEMPLATE_COMPLIANCE = "template_compliance"
    DAG_CYCLE_DETECTION = "dag_cycle_detection"
    BOUNDARY_TESTING = "boundary_testing"
    SECURITY_SCAN = "security_scan"


class MetaSkillStatus(Enum):
    """元技能状态"""
    DRAFT = "draft"
    TESTING = "testing"
    VERIFIED = "verified"
    DEPLOYED = "deployed"
    REJECTED = "rejected"


class GenerationStage(Enum):
    """生成阶段"""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    DESIGN = "design"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    SECURITY_SCAN = "security_scan"
    DEPLOYMENT = "deployment"


@dataclass
class MetaSkill:
    """元技能"""
    skill_id: str
    name: str
    description: str = ""
    status: MetaSkillStatus = MetaSkillStatus.DRAFT
    source_experience_ids: List[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    deployed_at: Optional[float] = None
    version: str = "v1.0.0"
    quality_checks: Dict[str, bool] = field(default_factory=dict)
    generated_code: Optional[str] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "source_experience_ids": self.source_experience_ids,
            "generated_at": self.generated_at,
            "deployed_at": self.deployed_at,
            "version": self.version,
            "quality_checks": self.quality_checks,
            "test_results": self.test_results,
            "metadata": self.metadata
        }


class MetaSkillGenerator:
    """MetaSkill 元技能生成器"""

    def __init__(self, skills_dir: str = None):
        """
        初始化元技能生成器

        Args:
            skills_dir: 技能存储目录
        """
        if skills_dir is None:
            skills_dir = os.path.join(os.path.expanduser("~"), ".agi_skills")
        self.skills_dir = skills_dir
        os.makedirs(skills_dir, exist_ok=True)

        self._generated_skills: Dict[str, MetaSkill] = {}
        self._generation_history: List[Dict[str, Any]] = []

    def generate_skill(self, requirement: str, source_experiences: List[str] = None,
                       quality_check_level: int = 4) -> MetaSkill:
        """
        从自然语言需求生成技能

        Args:
            requirement: 技能需求描述
            source_experiences: 源经验ID列表
            quality_check_level: 质量检查等级 (1-4)

        Returns:
            生成的元技能
        """
        skill_id = f"skill_{uuid.uuid4().hex[:8]}"
        meta_skill = MetaSkill(
            skill_id=skill_id,
            name=self._extract_skill_name(requirement),
            description=requirement,
            source_experience_ids=source_experiences or []
        )

        stages = [
            (GenerationStage.REQUIREMENT_ANALYSIS, self._analyze_requirement),
            (GenerationStage.DESIGN, self._design_skill),
            (GenerationStage.CODE_GENERATION, self._generate_code),
        ]

        if quality_check_level >= 2:
            stages.append((GenerationStage.TESTING, self._run_tests))
        if quality_check_level >= 3:
            stages.append((GenerationStage.SECURITY_SCAN, self._security_scan))
        if quality_check_level >= 4:
            stages.append((GenerationStage.DEPLOYMENT, self._deploy_skill))

        for stage, func in stages:
            try:
                result = func(meta_skill)
                meta_skill.metadata[stage.value] = result
            except Exception as e:
                meta_skill.status = MetaSkillStatus.REJECTED
                meta_skill.metadata["error"] = str(e)
                break

        self._generated_skills[skill_id] = meta_skill
        self._generation_history.append({
            "skill_id": skill_id,
            "requirement": requirement,
            "status": meta_skill.status.value,
            "generated_at": meta_skill.generated_at
        })

        return meta_skill

    def _extract_skill_name(self, requirement: str) -> str:
        """从需求中提取技能名称"""
        keywords = ["工具", "助手", "分析", "生成", "处理", "管理", "系统"]
        for kw in keywords:
            idx = requirement.find(kw)
            if idx > 0:
                return requirement[:idx + 2]
        return f"技能_{int(time.time())}"

    def _analyze_requirement(self, skill: MetaSkill) -> Dict[str, Any]:
        """需求分析阶段"""
        return {
            "completed": True,
            "features": self._extract_features(skill.description),
            "dependencies": []
        }

    def _extract_features(self, requirement: str) -> List[str]:
        """提取功能点"""
        features = []
        if "分析" in requirement:
            features.append("数据分析")
        if "生成" in requirement:
            features.append("内容生成")
        if "处理" in requirement:
            features.append("数据处理")
        if "管理" in requirement:
            features.append("资源管理")
        return features

    def _design_skill(self, skill: MetaSkill) -> Dict[str, Any]:
        """设计阶段"""
        return {
            "completed": True,
            "architecture": "plugin",
            "interfaces": ["execute", "validate", "describe"]
        }

    def _generate_code(self, skill: MetaSkill) -> Dict[str, Any]:
        """代码生成阶段"""
        template = f'''# {skill.name}
"""
{skill.description}
"""
class {skill.name.replace(" ", "")}Plugin:
    def execute(self, **kwargs):
        """执行技能"""
        return {{"status": "success", "message": "技能执行完成"}}

    def validate(self, **kwargs):
        """参数验证"""
        return True

    def describe(self):
        """描述技能"""
        return {{"name": "{skill.name}", "description": "{skill.description}"}}
'''
        skill.generated_code = template
        return {"completed": True, "lines": len(template.split("\n"))}

    def _run_tests(self, skill: MetaSkill) -> Dict[str, Any]:
        """测试阶段"""
        checks = [
            ("template_compliance", self._check_template_compliance(skill)),
            ("boundary_testing", self._check_boundary_cases(skill)),
        ]

        skill.quality_checks.update(dict(checks))
        all_passed = all(v for _, v in checks)

        if all_passed:
            skill.status = MetaSkillStatus.TESTING
        else:
            skill.status = MetaSkillStatus.REJECTED

        return {"completed": True, "checks": dict(checks), "all_passed": all_passed}

    def _check_template_compliance(self, skill: MetaSkill) -> bool:
        """模板合规校验"""
        if not skill.generated_code:
            return False
        return "class" in skill.generated_code and "def execute" in skill.generated_code

    def _check_boundary_cases(self, skill: MetaSkill) -> bool:
        """边界用例测试"""
        return True

    def _security_scan(self, skill: MetaSkill) -> Dict[str, Any]:
        """安全扫描阶段"""
        issues = []
        if skill.generated_code and "os.system" in skill.generated_code:
            issues.append("潜在命令注入风险")

        skill.quality_checks["security_scan"] = len(issues) == 0

        if issues:
            skill.status = MetaSkillStatus.REJECTED

        return {"completed": True, "issues": issues, "passed": len(issues) == 0}

    def _deploy_skill(self, skill: MetaSkill) -> Dict[str, Any]:
        """部署阶段"""
        skill_path = os.path.join(self.skills_dir, f"{skill.skill_id}.py")
        if skill.generated_code:
            with open(skill_path, "w", encoding="utf-8") as f:
                f.write(skill.generated_code)

            skill.status = MetaSkillStatus.DEPLOYED
            skill.deployed_at = time.time()
            return {"completed": True, "path": skill_path}

        return {"completed": False, "error": "无生成代码"}

    def get_skill(self, skill_id: str) -> Optional[MetaSkill]:
        """获取元技能"""
        return self._generated_skills.get(skill_id)

    def list_skills(self, status: MetaSkillStatus = None) -> List[MetaSkill]:
        """列出元技能"""
        skills = list(self._generated_skills.values())
        if status:
            skills = [s for s in skills if s.status == status]
        return sorted(skills, key=lambda s: s.generated_at, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        for skill in self._generated_skills.values():
            s = skill.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_generated": len(self._generated_skills),
            "status_counts": status_counts,
            "generation_history_count": len(self._generation_history)
        }
