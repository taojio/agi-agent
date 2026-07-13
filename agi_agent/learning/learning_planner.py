import numpy as np
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class LearningGoalType(Enum):
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    SKILL_DEVELOPMENT = "skill_development"
    PROBLEM_SOLVING = "problem_solving"
    CONCEPT_UNDERSTANDING = "concept_understanding"
    CREATIVE_EXPLORATION = "creative_exploration"
    ADAPTATION = "adaptation"
    OPTIMIZATION = "optimization"
    INNOVATION = "innovation"


class LearningPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LearningGoal:
    def __init__(self, goal_id: str, goal_type: LearningGoalType,
                 description: str, priority: LearningPriority = LearningPriority.MEDIUM,
                 target_confidence: float = 0.8, dependencies: Optional[List[str]] = None):
        self.goal_id = goal_id
        self.goal_type = goal_type
        self.description = description
        self.priority = priority
        self.target_confidence = target_confidence
        self.dependencies = dependencies if dependencies is not None else []
        self.current_confidence = 0.0
        self.progress = 0.0
        self.created_at = time.time()
        self.completed_at = None
        self.status = "pending"
        self.sub_goals: List[str] = []
        self.resources: List[str] = []
        self.estimated_time = 0

    def update_progress(self, new_confidence: float):
        self.current_confidence = min(self.target_confidence, new_confidence)
        self.progress = self.current_confidence / self.target_confidence

        if self.progress >= 1.0:
            self.status = "completed"
            self.completed_at = time.time()
        elif self.progress > 0:
            self.status = "in_progress"

    def to_dict(self):
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "description": self.description,
            "priority": self.priority.value,
            "target_confidence": self.target_confidence,
            "current_confidence": self.current_confidence,
            "progress": self.progress,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "dependencies": self.dependencies,
            "sub_goals": self.sub_goals,
            "resources": self.resources,
            "estimated_time": self.estimated_time
        }


class LearningTask:
    def __init__(self, task_id: str, goal_id: str, description: str,
                 type: str = "study", duration: int = 60, prerequisites: Optional[List[str]] = None):
        self.task_id = task_id
        self.goal_id = goal_id
        self.description = description
        self.type = type
        self.duration = duration
        self.prerequisites = prerequisites if prerequisites is not None else []
        self.status = "pending"
        self.completed_at = None
        self.effectiveness = 0.0
        self.attempts = 0
        self.last_attempt = None

    def complete(self, effectiveness: float = 1.0):
        self.status = "completed"
        self.completed_at = time.time()
        self.effectiveness = effectiveness
        self.attempts += 1
        self.last_attempt = time.time()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "type": self.type,
            "duration": self.duration,
            "prerequisites": self.prerequisites,
            "status": self.status,
            "completed_at": self.completed_at,
            "effectiveness": self.effectiveness,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt
        }


class LearningPlan:
    def __init__(self, plan_id: str, goals: Optional[List[LearningGoal]] = None,
                 tasks: Optional[List[LearningTask]] = None):
        self.plan_id = plan_id
        self.goals = goals if goals is not None else []
        self.tasks = tasks if tasks is not None else []
        self.created_at = time.time()
        self.status = "active"
        self.progress = 0.0

    def add_goal(self, goal: LearningGoal):
        self.goals.append(goal)

    def add_task(self, task: LearningTask):
        self.tasks.append(task)

    def update_progress(self):
        if not self.goals:
            self.progress = 0.0
            return

        completed_goals = sum(1 for g in self.goals if g.status == "completed")
        self.progress = completed_goals / len(self.goals)

        if self.progress >= 1.0:
            self.status = "completed"

    def to_dict(self):
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "goal_count": len(self.goals),
            "task_count": len(self.tasks),
            "goals": [g.to_dict() for g in self.goals],
            "tasks": [t.to_dict() for t in self.tasks]
        }


class LearningPlanner:
    def __init__(self):
        self.goals: Dict[str, LearningGoal] = {}
        self.tasks: Dict[str, LearningTask] = {}
        self.plans: Dict[str, LearningPlan] = {}
        self.learning_history = deque(maxlen=500)
        self.exploration_budget = 0.3
        self.max_goals_per_plan = 10
        self.priority_weights = {
            LearningPriority.LOW: 1.0,
            LearningPriority.MEDIUM: 2.0,
            LearningPriority.HIGH: 4.0,
            LearningPriority.CRITICAL: 8.0
        }

    def create_learning_goal(self, goal_type: LearningGoalType, description: str,
                             priority: LearningPriority = LearningPriority.MEDIUM,
                             target_confidence: float = 0.8,
                             dependencies: Optional[List[str]] = None) -> str:
        goal_id = f"goal_{len(self.goals) + 1}"
        goal = LearningGoal(goal_id, goal_type, description, priority, target_confidence, dependencies)
        self.goals[goal_id] = goal

        self._decompose_goal(goal)

        return goal_id

    def _decompose_goal(self, goal: LearningGoal):
        sub_goals = self._generate_sub_goals(goal)
        for sub_goal in sub_goals:
            sub_id = f"sub_{goal.goal_id}_{len(goal.sub_goals) + 1}"
            sub_goal_obj = LearningGoal(sub_id, goal.goal_type, sub_goal, goal.priority,
                                        goal.target_confidence * 0.8)
            self.goals[sub_id] = sub_goal_obj
            goal.sub_goals.append(sub_id)

    def _generate_sub_goals(self, goal: LearningGoal) -> List[str]:
        strategies = {
            LearningGoalType.KNOWLEDGE_ACQUISITION: [
                "理解核心概念",
                "掌握基础理论",
                "学习相关案例",
                "应用实践练习"
            ],
            LearningGoalType.SKILL_DEVELOPMENT: [
                "学习基本操作",
                "掌握进阶技巧",
                "进行实践练习",
                "接受反馈改进"
            ],
            LearningGoalType.PROBLEM_SOLVING: [
                "分析问题本质",
                "寻找解决方案",
                "实施解决方案",
                "评估解决效果"
            ],
            LearningGoalType.CONCEPT_UNDERSTANDING: [
                "定义概念边界",
                "分析概念关系",
                "构建概念模型",
                "验证模型正确性"
            ],
            LearningGoalType.CREATIVE_EXPLORATION: [
                "收集相关灵感",
                "探索多种可能性",
                "生成创意方案",
                "评估创意价值"
            ],
            LearningGoalType.ADAPTATION: [
                "识别变化需求",
                "学习新的方法",
                "适应新环境",
                "优化适应策略"
            ],
            LearningGoalType.OPTIMIZATION: [
                "识别改进空间",
                "分析当前瓶颈",
                "实施优化方案",
                "验证优化效果"
            ],
            LearningGoalType.INNOVATION: [
                "发现创新机会",
                "探索创新方向",
                "开发创新方案",
                "验证创新价值"
            ]
        }

        return strategies.get(goal.goal_type, ["基础学习", "实践应用", "深度理解"])

    def create_learning_plan(self, goal_ids: List[str]) -> str:
        plan_id = f"plan_{len(self.plans) + 1}"
        plan = LearningPlan(plan_id)

        sorted_goals = sorted(goal_ids, key=lambda gid: self._calculate_goal_priority(gid), reverse=True)
        sorted_goals = sorted_goals[:self.max_goals_per_plan]

        for goal_id in sorted_goals:
            if goal_id in self.goals:
                plan.add_goal(self.goals[goal_id])
                self._generate_tasks_for_goal(goal_id, plan)

        self.plans[plan_id] = plan
        return plan_id

    def _calculate_goal_priority(self, goal_id: str) -> float:
        if goal_id not in self.goals:
            return 0.0

        goal = self.goals[goal_id]
        priority_weight = self.priority_weights.get(goal.priority, 1.0)
        urgency = max(0, 1.0 - (time.time() - goal.created_at) / 86400)
        dependency_count = len(goal.dependencies)

        return priority_weight * (1 + urgency + dependency_count * 0.2)

    def _generate_tasks_for_goal(self, goal_id: str, plan: LearningPlan):
        goal = self.goals.get(goal_id)
        if not goal:
            return

        task_types = {
            LearningGoalType.KNOWLEDGE_ACQUISITION: [
                ("阅读学习", 30),
                ("视频教程", 45),
                ("实践练习", 60),
                ("知识测试", 15)
            ],
            LearningGoalType.SKILL_DEVELOPMENT: [
                ("技能演示", 20),
                ("模仿练习", 40),
                ("独立实践", 60),
                ("成果评估", 30)
            ],
            LearningGoalType.PROBLEM_SOLVING: [
                ("问题分析", 30),
                ("方案设计", 45),
                ("方案实施", 60),
                ("效果验证", 30)
            ],
            LearningGoalType.CONCEPT_UNDERSTANDING: [
                ("概念定义", 15),
                ("关系分析", 30),
                ("模型构建", 45),
                ("模型验证", 20)
            ],
            LearningGoalType.CREATIVE_EXPLORATION: [
                ("灵感收集", 30),
                ("发散思考", 45),
                ("方案生成", 60),
                ("方案筛选", 20)
            ],
            LearningGoalType.ADAPTATION: [
                ("环境分析", 20),
                ("策略学习", 30),
                ("实践适应", 45),
                ("策略优化", 30)
            ],
            LearningGoalType.OPTIMIZATION: [
                ("现状分析", 30),
                ("瓶颈识别", 20),
                ("优化实施", 60),
                ("效果评估", 25)
            ],
            LearningGoalType.INNOVATION: [
                ("机会识别", 25),
                ("方向探索", 40),
                ("方案开发", 60),
                ("价值验证", 30)
            ]
        }

        for i, (task_desc, duration) in enumerate(task_types.get(goal.goal_type, [("学习任务", 60)])):
            task_id = f"task_{goal_id}_{i + 1}"
            task = LearningTask(task_id, goal_id, task_desc, type="study", duration=duration)
            self.tasks[task_id] = task
            plan.add_task(task)

    def execute_task(self, task_id: str, effectiveness: float = 1.0) -> Dict[str, Any]:
        if task_id not in self.tasks:
            return {"success": False, "message": "Task not found"}

        task = self.tasks[task_id]
        task.complete(effectiveness)

        goal_id = task.goal_id
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            completed_subtasks = sum(1 for t in self.tasks.values()
                                     if t.goal_id == goal_id and t.status == "completed")
            total_tasks = sum(1 for t in self.tasks.values() if t.goal_id == goal_id)

            if total_tasks > 0:
                task_progress = completed_subtasks / total_tasks
                goal.update_progress(task_progress * goal.target_confidence)

        self.learning_history.append({
            "task_id": task_id,
            "goal_id": goal_id,
            "effectiveness": effectiveness,
            "timestamp": time.time()
        })

        return {
            "success": True,
            "task_id": task_id,
            "goal_id": goal_id,
            "effectiveness": effectiveness,
            "goal_progress": self.goals[goal_id].progress if goal_id in self.goals else 0.0
        }

    def execute_plan(self, plan_id: str, effectiveness: float = 1.0) -> Dict[str, Any]:
        """执行计划中的所有待执行任务"""
        if plan_id not in self.plans:
            return {"success": False, "message": "Plan not found"}

        plan = self.plans[plan_id]
        executed_tasks = []

        for task in plan.tasks:
            if task.status == "pending":
                result = self.execute_task(task.task_id, effectiveness)
                executed_tasks.append(result)

        plan.update_progress()

        return {
            "success": True,
            "plan_id": plan_id,
            "executed_count": len(executed_tasks),
            "plan_progress": plan.progress,
            "plan_status": plan.status
        }

    def get_recommended_goal(self) -> Optional[Dict]:
        active_goals = [g for g in self.goals.values() if g.status not in ("completed", "abandoned")]
        if not active_goals:
            return None

        best_goal = max(active_goals, key=lambda g: self._calculate_goal_priority(g.goal_id))
        return best_goal.to_dict()

    def get_recommended_task(self, goal_id: str = None) -> Optional[Dict]:
        if goal_id:
            tasks_for_goal = [t for t in self.tasks.values()
                              if t.goal_id == goal_id and t.status == "pending"]
        else:
            tasks_for_goal = [t for t in self.tasks.values() if t.status == "pending"]

        if not tasks_for_goal:
            return None

        best_task = min(tasks_for_goal, key=lambda t: len(t.prerequisites))
        return best_task.to_dict()

    def explore_new_topics(self, count: int = 3) -> List[Dict]:
        exploration_topics = []

        topic_categories = [
            {"type": LearningGoalType.KNOWLEDGE_ACQUISITION, "description": "探索新知识领域"},
            {"type": LearningGoalType.CREATIVE_EXPLORATION, "description": "进行创意探索"},
            {"type": LearningGoalType.INNOVATION, "description": "寻找创新机会"},
            {"type": LearningGoalType.CONCEPT_UNDERSTANDING, "description": "深入理解新概念"},
            {"type": LearningGoalType.ADAPTATION, "description": "探索适应新环境的方法"}
        ]

        np.random.shuffle(topic_categories)

        for i, topic in enumerate(topic_categories[:count]):
            goal_id = self.create_learning_goal(
                topic["type"],
                topic["description"],
                LearningPriority.LOW
            )
            exploration_topics.append({
                "goal_id": goal_id,
                "type": topic["type"].value,
                "description": topic["description"],
                "priority": "exploration"
            })

        return exploration_topics

    def get_plan_summary(self, plan_id: str) -> Optional[Dict]:
        if plan_id not in self.plans:
            return None

        plan = self.plans[plan_id]
        plan.update_progress()

        completed_tasks = sum(1 for t in plan.tasks if t.status == "completed")
        avg_effectiveness = np.mean([t.effectiveness for t in plan.tasks
                                      if t.status == "completed"]) if completed_tasks > 0 else 0.0

        return {
            "plan_id": plan_id,
            "status": plan.status,
            "progress": plan.progress,
            "goals_total": len(plan.goals),
            "goals_completed": sum(1 for g in plan.goals if g.status == "completed"),
            "tasks_total": len(plan.tasks),
            "tasks_completed": completed_tasks,
            "avg_effectiveness": float(avg_effectiveness),
            "goals": [g.to_dict() for g in plan.goals],
            "tasks": [t.to_dict() for t in plan.tasks]
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        stats = {
            "total_goals": len(self.goals),
            "completed_goals": sum(1 for g in self.goals.values() if g.status == "completed"),
            "active_goals": sum(1 for g in self.goals.values() if g.status == "in_progress"),
            "pending_goals": sum(1 for g in self.goals.values() if g.status == "pending"),
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == "completed"),
            "total_plans": len(self.plans),
            "completed_plans": sum(1 for p in self.plans.values() if p.status == "completed"),
            "avg_goal_progress": 0.0,
            "avg_task_effectiveness": 0.0,
            "learning_history_count": len(self.learning_history)
        }

        active_goals_list = [g for g in self.goals.values() if g.status in ("in_progress", "completed")]
        if active_goals_list:
            stats["avg_goal_progress"] = float(np.mean([g.progress for g in active_goals_list]))

        completed_tasks_list = [t for t in self.tasks.values() if t.status == "completed"]
        if completed_tasks_list:
            stats["avg_task_effectiveness"] = float(np.mean([t.effectiveness for t in completed_tasks_list]))

        return stats