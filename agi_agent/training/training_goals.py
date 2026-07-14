import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque


class GoalLayer(Enum):
    FOUNDATION = "foundation"
    PROFICIENCY = "proficiency"
    EVOLUTION = "evolution"
    ALIGNMENT = "alignment"


class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    EXCEEDED = "exceeded"


@dataclass
class TrainingGoal:
    goal_id: str
    name: str
    description: str
    layer: GoalLayer
    target_value: float
    current_value: float = 0.0
    metric_name: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 5
    deadline_step: Optional[int] = None
    achieved_step: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def progress_ratio(self) -> float:
        if self.target_value == 0:
            return 1.0 if self.current_value > 0 else 0.0
        ratio = self.current_value / abs(self.target_value)
        return max(0.0, min(1.0, ratio))

    def update(self, value: float, step: int = None) -> GoalStatus:
        old_value = self.current_value
        self.current_value = value
        self.history.append({
            "step": step,
            "value": value,
            "timestamp": time.time()
        })

        if self.target_value > 0:
            if value >= self.target_value:
                self.status = GoalStatus.ACHIEVED
                self.achieved_step = step
            elif value >= self.target_value * 1.1:
                self.status = GoalStatus.EXCEEDED
            elif value > 0:
                self.status = GoalStatus.IN_PROGRESS
        else:
            if value <= self.target_value:
                self.status = GoalStatus.ACHIEVED
                self.achieved_step = step
            elif value < self.target_value * 0.9:
                self.status = GoalStatus.EXCEEDED
            elif value > 0:
                self.status = GoalStatus.IN_PROGRESS

        return self.status


@dataclass
class GoalCategory:
    category_id: str
    name: str
    description: str
    goals: Dict[str, TrainingGoal] = field(default_factory=dict)
    weight: float = 1.0

    def add_goal(self, goal: TrainingGoal):
        self.goals[goal.goal_id] = goal

    def get_goal(self, goal_id: str) -> Optional[TrainingGoal]:
        return self.goals.get(goal_id)

    def category_progress(self) -> float:
        if not self.goals:
            return 0.0
        total_progress = sum(g.progress_ratio() for g in self.goals.values())
        return total_progress / len(self.goals)

    def achieved_count(self) -> int:
        return sum(1 for g in self.goals.values()
                   if g.status in (GoalStatus.ACHIEVED, GoalStatus.EXCEEDED))


class TrainingGoalManager:
    def __init__(self):
        self.categories: Dict[str, GoalCategory] = {}
        self.goal_achievement_callbacks: List[Callable] = []
        self._init_default_categories()

    def _init_default_categories(self):
        foundation_perception = GoalCategory(
            category_id="foundation_perception",
            name="感知建模能力",
            description="建立稳定的输入特征提取与表示学习能力",
            weight=1.0
        )
        foundation_perception.add_goal(TrainingGoal(
            goal_id="fg_perception_01",
            name="自由能基线达标",
            description="自由能稳定在0.3以下",
            layer=GoalLayer.FOUNDATION,
            target_value=0.3,
            metric_name="free_energy",
            priority=1
        ))
        foundation_perception.add_goal(TrainingGoal(
            goal_id="fg_perception_02",
            name="感知特征区分度",
            description="特征区分度达到0.6以上",
            layer=GoalLayer.FOUNDATION,
            target_value=0.6,
            metric_name="feature_discrimination",
            priority=3
        ))
        self.categories["foundation_perception"] = foundation_perception

        foundation_cognition = GoalCategory(
            category_id="foundation_cognition",
            name="认知推理能力",
            description="构建因果推理、逻辑演绎等基础认知能力",
            weight=1.0
        )
        foundation_cognition.add_goal(TrainingGoal(
            goal_id="fg_cognition_01",
            name="置信度基线达标",
            description="置信度稳定在0.5以上",
            layer=GoalLayer.FOUNDATION,
            target_value=0.5,
            metric_name="confidence",
            priority=1
        ))
        foundation_cognition.add_goal(TrainingGoal(
            goal_id="fg_cognition_02",
            name="基础推理准确率",
            description="简单推理任务准确率达到60%",
            layer=GoalLayer.FOUNDATION,
            target_value=0.6,
            metric_name="reasoning_accuracy",
            priority=2
        ))
        self.categories["foundation_cognition"] = foundation_cognition

        foundation_homeostasis = GoalCategory(
            category_id="foundation_homeostasis",
            name="稳态维持能力",
            description="确保内稳态指标在合理区间",
            weight=0.8
        )
        foundation_homeostasis.add_goal(TrainingGoal(
            goal_id="fg_homeo_01",
            name="能量水平稳定",
            description="能量水平稳定在0.6-0.8区间",
            layer=GoalLayer.FOUNDATION,
            target_value=0.7,
            metric_name="energy_level",
            priority=2
        ))
        self.categories["foundation_homeostasis"] = foundation_homeostasis

        proficiency_knowledge = GoalCategory(
            category_id="proficiency_knowledge",
            name="知识整合能力",
            description="高效吸收、组织、检索多领域知识",
            weight=1.0
        )
        proficiency_knowledge.add_goal(TrainingGoal(
            goal_id="pg_knowledge_01",
            name="知识图谱规模",
            description="知识图谱节点数达到500以上",
            layer=GoalLayer.PROFICIENCY,
            target_value=500,
            metric_name="kg_node_count",
            priority=2
        ))
        proficiency_knowledge.add_goal(TrainingGoal(
            goal_id="pg_knowledge_02",
            name="知识检索准确率",
            description="知识检索准确率达到70%",
            layer=GoalLayer.PROFICIENCY,
            target_value=0.7,
            metric_name="knowledge_retrieval_accuracy",
            priority=3
        ))
        self.categories["proficiency_knowledge"] = proficiency_knowledge

        proficiency_decision = GoalCategory(
            category_id="proficiency_decision",
            name="决策优化能力",
            description="在复杂环境下做出高质量、低风险的决策",
            weight=1.0
        )
        proficiency_decision.add_goal(TrainingGoal(
            goal_id="pg_decision_01",
            name="决策准确率",
            description="决策准确率达到70%以上",
            layer=GoalLayer.PROFICIENCY,
            target_value=0.7,
            metric_name="decision_accuracy",
            priority=1
        ))
        proficiency_decision.add_goal(TrainingGoal(
            goal_id="pg_decision_02",
            name="决策效率提升",
            description="单步决策延迟低于500ms",
            layer=GoalLayer.PROFICIENCY,
            target_value=500,
            metric_name="decision_latency_ms",
            priority=2
        ))
        self.categories["proficiency_decision"] = proficiency_decision

        evolution_architecture = GoalCategory(
            category_id="evolution_architecture",
            name="架构自进化能力",
            description="根据环境复杂度自主优化网络结构",
            weight=1.0
        )
        evolution_architecture.add_goal(TrainingGoal(
            goal_id="eg_arch_01",
            name="进化成功率",
            description="进化优化成功率达到60%",
            layer=GoalLayer.EVOLUTION,
            target_value=0.6,
            metric_name="evolution_success_rate",
            priority=2
        ))
        evolution_architecture.add_goal(TrainingGoal(
            goal_id="eg_arch_02",
            name="性能提升幅度",
            description="进化带来的综合性能提升>30%",
            layer=GoalLayer.EVOLUTION,
            target_value=0.3,
            metric_name="performance_improvement",
            priority=1
        ))
        self.categories["evolution_architecture"] = evolution_architecture

        alignment_safety = GoalCategory(
            category_id="alignment_safety",
            name="安全边界遵守",
            description="所有行为严格遵守安全约束与伦理规范",
            weight=1.5
        )
        alignment_safety.add_goal(TrainingGoal(
            goal_id="ag_safety_01",
            name="安全合规率",
            description="安全合规率达到99%以上",
            layer=GoalLayer.ALIGNMENT,
            target_value=0.99,
            metric_name="safety_compliance_rate",
            priority=1
        ))
        alignment_safety.add_goal(TrainingGoal(
            goal_id="ag_safety_02",
            name="零重大安全事故",
            description="训练期间无红色安全预警",
            layer=GoalLayer.ALIGNMENT,
            target_value=0,
            metric_name="critical_safety_incidents",
            priority=1
        ))
        self.categories["alignment_safety"] = alignment_safety

    def register_goal_callback(self, callback: Callable):
        self.goal_achievement_callbacks.append(callback)

    def update_goal(self, metric_name: str, value: float, step: int = None):
        for category in self.categories.values():
            for goal in category.goals.values():
                if goal.metric_name == metric_name:
                    old_status = goal.status
                    new_status = goal.update(value, step)
                    if new_status in (GoalStatus.ACHIEVED, GoalStatus.EXCEEDED) and \
                       old_status not in (GoalStatus.ACHIEVED, GoalStatus.EXCEEDED):
                        self._on_goal_achieved(goal)

    def _on_goal_achieved(self, goal: TrainingGoal):
        for callback in self.goal_achievement_callbacks:
            try:
                callback(goal)
            except Exception:
                pass

    def get_layer_goals(self, layer: GoalLayer) -> List[TrainingGoal]:
        goals = []
        for category in self.categories.values():
            for goal in category.goals.values():
                if goal.layer == layer:
                    goals.append(goal)
        return goals

    def get_layer_progress(self, layer: GoalLayer) -> float:
        layer_goals = self.get_layer_goals(layer)
        if not layer_goals:
            return 0.0
        return sum(g.progress_ratio() for g in layer_goals) / len(layer_goals)

    def get_overall_progress(self) -> float:
        if not self.categories:
            return 0.0
        total_weight = sum(c.weight for c in self.categories.values())
        if total_weight == 0:
            return 0.0
        weighted_progress = sum(
            c.category_progress() * c.weight
            for c in self.categories.values()
        )
        return weighted_progress / total_weight

    def get_pending_high_priority_goals(self, top_n: int = 5) -> List[TrainingGoal]:
        all_goals = []
        for category in self.categories.values():
            for goal in category.goals.values():
                if goal.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS):
                    all_goals.append(goal)
        all_goals.sort(key=lambda g: g.priority)
        return all_goals[:top_n]

    def get_summary(self) -> Dict[str, Any]:
        total_goals = sum(len(c.goals) for c in self.categories.values())
        achieved_goals = sum(
            sum(1 for g in c.goals.values()
                if g.status in (GoalStatus.ACHIEVED, GoalStatus.EXCEEDED))
            for c in self.categories.values()
        )

        layer_progress = {}
        for layer in GoalLayer:
            layer_progress[layer.value] = self.get_layer_progress(layer)

        return {
            "total_goals": total_goals,
            "achieved_goals": achieved_goals,
            "overall_progress": self.get_overall_progress(),
            "layer_progress": layer_progress,
            "categories": {
                cid: {
                    "name": cat.name,
                    "progress": cat.category_progress(),
                    "achieved": cat.achieved_count(),
                    "total": len(cat.goals)
                }
                for cid, cat in self.categories.items()
            }
        }
