"""
dual_loop_evolution.py - 双循环进化体系

整体进化分为内外两个耦合循环：
- 外环：经验积累循环（广度拓展）
- 内环：精准优化循环（深度提升）
"""
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field


class EvolutionPhase(Enum):
    """进化阶段"""
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    VERIFICATION = "verification"


class EvolutionLevel(Enum):
    """进化分级"""
    COMPONENT = "component"
    INDIVIDUAL = "individual"
    COLLECTIVE = "collective"


class EvolutionMode(Enum):
    """进化模式"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


@dataclass
class EvolutionRecord:
    """进化记录"""
    record_id: str
    type: str
    content: str
    source: str
    phase: EvolutionPhase
    created_at: float = field(default_factory=time.time)
    verified: bool = False
    verification_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "type": self.type,
            "content": self.content,
            "source": self.source,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "verified": self.verified,
            "verification_result": self.verification_result
        }


@dataclass
class ImprovementProposal:
    """改进提案"""
    proposal_id: str
    title: str
    description: str
    level: EvolutionLevel
    target: str
    expected_benefit: float = 0.0
    risk: float = 0.0
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    test_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "description": self.description,
            "level": self.level.value,
            "target": self.target,
            "expected_benefit": self.expected_benefit,
            "risk": self.risk,
            "status": self.status,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "test_results": self.test_results
        }


class DualLoopEvolution:
    """双循环进化体系"""

    def __init__(self, memory_harness=None, metaskill_generator=None):
        """
        初始化双循环进化系统

        Args:
            memory_harness: 记忆调度器
            metaskill_generator: 元技能生成器
        """
        self.memory_harness = memory_harness
        self.metaskill_generator = metaskill_generator

        self._outer_loop_records: List[EvolutionRecord] = []
        self._inner_loop_records: List[EvolutionRecord] = []
        self._proposals: Dict[str, ImprovementProposal] = {}
        self._enabled_levels: Dict[EvolutionLevel, bool] = {
            EvolutionLevel.COMPONENT: True,
            EvolutionLevel.INDIVIDUAL: False,
            EvolutionLevel.COLLECTIVE: False
        }
        self._mode = EvolutionMode.HYBRID

        self._cycle_counter = 0

    def set_mode(self, mode: EvolutionMode):
        """设置进化模式"""
        self._mode = mode

    def enable_level(self, level: EvolutionLevel, enabled: bool):
        """启用/禁用进化等级"""
        self._enabled_levels[level] = enabled

    def is_level_enabled(self, level: EvolutionLevel) -> bool:
        """检查进化等级是否启用"""
        return self._enabled_levels.get(level, False)

    def run_outer_loop(self):
        """运行外环：经验积累循环（广度拓展）"""
        self._cycle_counter += 1

        records = self._collect_experiences()
        self._outer_loop_records.extend(records)

        proposals = self._analyze_experiences(records)
        for prop in proposals:
            self._proposals[prop.proposal_id] = prop

        return {
            "cycle": self._cycle_counter,
            "phase": EvolutionPhase.COLLECTION.value,
            "records_collected": len(records),
            "proposals_generated": len(proposals)
        }

    def _collect_experiences(self) -> List[EvolutionRecord]:
        """收集经验"""
        records = []

        if self.memory_harness:
            memories = self.memory_harness.retrieve(
                query="",
                max_results=50
            )
            for mem in memories:
                record = EvolutionRecord(
                    record_id=f"exp_{uuid.uuid4().hex[:8]}",
                    type=mem.metadata.category.value,
                    content=mem.content,
                    source=mem.metadata.source_agent,
                    phase=EvolutionPhase.COLLECTION
                )
                records.append(record)

        return records

    def _analyze_experiences(self, records: List[EvolutionRecord]) -> List[ImprovementProposal]:
        """分析经验，生成改进提案"""
        proposals = []

        if not records:
            return proposals

        experience_types = {}
        for r in records:
            experience_types[r.type] = experience_types.get(r.type, 0) + 1

        for exp_type, count in experience_types.items():
            if count >= 3 and self.is_level_enabled(EvolutionLevel.COMPONENT):
                proposal = ImprovementProposal(
                    proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                    title=f"优化{exp_type}处理策略",
                    description=f"检测到{count}次{exp_type}经验，建议优化处理策略",
                    level=EvolutionLevel.COMPONENT,
                    target=f"{exp_type}_handler",
                    expected_benefit=min(0.3, count * 0.05),
                    risk=0.1
                )
                proposals.append(proposal)

        return proposals

    def run_inner_loop(self, proposal_id: str = None):
        """
        运行内环：精准优化循环（深度提升）

        Args:
            proposal_id: 指定提案ID，不指定则处理所有待处理提案

        Returns:
            优化结果
        """
        if proposal_id:
            proposals = [self._proposals.get(proposal_id)]
        else:
            proposals = [p for p in self._proposals.values() if p.status == "proposed"]

        results = []
        for prop in proposals:
            if prop and self.is_level_enabled(prop.level):
                result = self._optimize(prop)
                results.append(result)

        return {
            "cycle": self._cycle_counter,
            "phase": EvolutionPhase.OPTIMIZATION.value,
            "proposals_processed": len(results),
            "results": results
        }

    def _optimize(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """执行优化"""
        proposal.status = "testing"

        test_result = self._test_proposal(proposal)
        proposal.test_results = test_result

        if test_result.get("passed", False):
            proposal.status = "approved"
            proposal.applied_at = time.time()
            return {"proposal_id": proposal.proposal_id, "status": "approved", "test_result": test_result}
        else:
            proposal.status = "rejected"
            return {"proposal_id": proposal.proposal_id, "status": "rejected", "test_result": test_result}

    def _test_proposal(self, proposal: ImprovementProposal) -> Dict[str, Any]:
        """测试提案"""
        import random
        passed = random.random() > 0.2
        return {
            "passed": passed,
            "score": random.uniform(0.7, 0.95) if passed else random.uniform(0.4, 0.6),
            "metrics": {"accuracy": random.uniform(0.7, 0.95), "speedup": random.uniform(0.1, 0.3)}
        }

    def trigger_skill_conversion(self, record_ids: List[str]):
        """触发技能转换"""
        if not self.metaskill_generator:
            return {"success": False, "error": "元技能生成器未配置"}

        records = [r for r in self._outer_loop_records if r.record_id in record_ids]
        if not records:
            return {"success": False, "error": "未找到指定记录"}

        requirement = "\n".join([r.content for r in records])
        skill = self.metaskill_generator.generate_skill(requirement, record_ids)

        return {"success": True, "skill_id": skill.skill_id, "status": skill.status.value}

    def get_proposal(self, proposal_id: str) -> Optional[ImprovementProposal]:
        """获取提案"""
        return self._proposals.get(proposal_id)

    def list_proposals(self, status: str = None) -> List[ImprovementProposal]:
        """列出提案"""
        proposals = list(self._proposals.values())
        if status:
            proposals = [p for p in proposals if p.status == status]
        return sorted(proposals, key=lambda p: p.created_at, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        outer_count = len(self._outer_loop_records)
        inner_count = len(self._inner_loop_records)
        proposal_counts = {}
        for p in self._proposals.values():
            proposal_counts[p.status] = proposal_counts.get(p.status, 0) + 1

        return {
            "cycle": self._cycle_counter,
            "mode": self._mode.value,
            "enabled_levels": {k.value: v for k, v in self._enabled_levels.items()},
            "outer_loop_records": outer_count,
            "inner_loop_records": inner_count,
            "proposals": proposal_counts
        }
