"""
multi_agent/enhanced_swarm.py - 增强多智能体协作网络

在原有AgentSwarm基础上增强：
- 智能体能力匹配与任务分配优化
- 自适应协作策略（根据任务类型动态选择）
- 动态负载均衡
- 智能体信誉系统
- 协作学习机制
- 分布式共识算法

使用方式：
    from agi_agent.multi_agent import EnhancedAgentSwarm
    
    swarm = EnhancedAgentSwarm()
    swarm.register_agent("Agent1", AgentRole.WORKER, ["python", "ml"])
    swarm.assign_task_optimized({"task_id": "t1", "requires": ["python"]})
"""
import uuid
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque, defaultdict
from dataclasses import dataclass, field

from .agent_swarm import AgentSwarm, AgentRole, AgentStatus, CollaborationMode, AgentInfo


class ReputationMetric(Enum):
    COMPLETION_RATE = "completion_rate"
    QUALITY_SCORE = "quality_score"
    RESPONSE_TIME = "response_time"
    COOPERATION = "cooperation"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    name: str
    description: str
    requires: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[float] = None
    estimated_work: float = 1.0
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class CapabilityMatcher:
    """
    能力匹配器
    
    根据智能体能力与任务需求进行智能匹配。
    """
    
    def __init__(self):
        self.capability_taxonomy: Dict[str, List[str]] = defaultdict(list)
    
    def build_taxonomy(self, capabilities: Dict[str, List[str]]):
        """
        构建能力分类体系
        
        Args:
            capabilities: 智能体能力字典
        """
        for agent_id, caps in capabilities.items():
            for cap in caps:
                if agent_id not in self.capability_taxonomy[cap]:
                    self.capability_taxonomy[cap].append(agent_id)
    
    def find_best_match(self, task_requires: List[str], 
                        agents: Dict[str, AgentInfo]) -> Optional[str]:
        """
        找到最佳匹配的智能体
        
        Args:
            task_requires: 任务需求能力列表
            agents: 智能体字典
        
        Returns:
            最佳匹配的智能体ID
        """
        best_agent = None
        best_score = -1
        
        for agent_id, agent in agents.items():
            if agent.status != AgentStatus.IDLE:
                continue
            
            score = self._calculate_match_score(task_requires, agent.capabilities)
            score -= agent.workload * 0.3
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _calculate_match_score(self, required: List[str], 
                              available: List[str]) -> float:
        """
        计算匹配分数
        
        Args:
            required: 需求能力列表
            available: 可用能力列表
        
        Returns:
            匹配分数 (0-1)
        """
        if not required:
            return 0.5
        
        required_set = set(required)
        available_set = set(available)
        
        intersection = required_set & available_set
        
        if not intersection:
            return 0.0
        
        exact_match = len(intersection) / len(required_set)
        
        related_score = 0.0
        for req in required:
            for avail in available:
                if req.lower() in avail.lower() or avail.lower() in req.lower():
                    related_score += 0.1
        
        return min(1.0, exact_match + related_score)


class ReputationSystem:
    """
    智能体信誉系统
    
    追踪和评估智能体的表现，用于任务分配决策。
    """
    
    def __init__(self):
        self.reputations: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "completion_rate": 0.8,
                "quality_score": 0.5,
                "response_time": 0.5,
                "cooperation": 0.5,
            }
        )
        self.task_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def update_reputation(self, agent_id: str, metric: ReputationMetric, 
                         value: float) -> None:
        """
        更新信誉指标
        
        Args:
            agent_id: 智能体ID
            metric: 信誉指标
            value: 指标值 (0-1)
        """
        current = self.reputations[agent_id][metric.value]
        self.reputations[agent_id][metric.value] = 0.7 * current + 0.3 * value
    
    def record_task_completion(self, agent_id: str, task_id: str, 
                              success: bool, quality: float = 0.5,
                              response_time: float = 0.0) -> None:
        """
        记录任务完成情况
        
        Args:
            agent_id: 智能体ID
            task_id: 任务ID
            success: 是否成功
            quality: 质量评分
            response_time: 响应时间（秒）
        """
        history = {
            "task_id": task_id,
            "success": success,
            "quality": quality,
            "response_time": response_time,
            "timestamp": time.time(),
        }
        self.task_history[agent_id].append(history)
        
        if len(self.task_history[agent_id]) > 50:
            self.task_history[agent_id] = self.task_history[agent_id][-50:]
        
        completed = len(self.task_history[agent_id])
        successes = sum(1 for h in self.task_history[agent_id] if h["success"])
        
        self.update_reputation(agent_id, ReputationMetric.COMPLETION_RATE, successes / completed)
        self.update_reputation(agent_id, ReputationMetric.QUALITY_SCORE, quality)
        
        rt_score = max(0.0, min(1.0, 1.0 - response_time / 60.0))
        self.update_reputation(agent_id, ReputationMetric.RESPONSE_TIME, rt_score)
    
    def get_reputation(self, agent_id: str) -> Dict[str, float]:
        """
        获取智能体信誉
        
        Args:
            agent_id: 智能体ID
        
        Returns:
            信誉字典
        """
        return dict(self.reputations[agent_id])
    
    def get_overall_score(self, agent_id: str) -> float:
        """
        获取综合信誉分数
        
        Args:
            agent_id: 智能体ID
        
        Returns:
            综合分数 (0-1)
        """
        rep = self.reputations[agent_id]
        return 0.3 * rep["completion_rate"] + 0.3 * rep["quality_score"] + \
               0.2 * rep["response_time"] + 0.2 * rep["cooperation"]


class CollaborativeLearner:
    """
    协作学习器
    
    智能体之间共享知识和经验，提升整体性能。
    """
    
    def __init__(self):
        self.shared_knowledge: Dict[str, Dict[str, Any]] = {}
        self.learning_events: List[Dict[str, Any]] = []
    
    def share_knowledge(self, agent_id: str, knowledge_type: str, 
                       content: Any, context: Optional[Dict] = None) -> str:
        """
        共享知识
        
        Args:
            agent_id: 智能体ID
            knowledge_type: 知识类型
            content: 知识内容
            context: 上下文信息
        
        Returns:
            知识ID
        """
        knowledge_id = f"k_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        self.shared_knowledge[knowledge_id] = {
            "knowledge_id": knowledge_id,
            "agent_id": agent_id,
            "knowledge_type": knowledge_type,
            "content": content,
            "context": context or {},
            "shared_at": time.time(),
            "access_count": 0,
        }
        
        self.learning_events.append({
            "event_type": "knowledge_shared",
            "agent_id": agent_id,
            "knowledge_id": knowledge_id,
            "timestamp": time.time(),
        })
        
        return knowledge_id
    
    def retrieve_knowledge(self, query: str, 
                          knowledge_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        检索知识
        
        Args:
            query: 查询关键词
            knowledge_types: 知识类型过滤
        
        Returns:
            匹配的知识列表
        """
        results = []
        
        for knowledge in self.shared_knowledge.values():
            content_str = str(knowledge["content"]).lower()
            query_str = query.lower()
            
            if query_str in content_str or content_str in query_str:
                if knowledge_types and knowledge["knowledge_type"] not in knowledge_types:
                    continue
                
                knowledge["access_count"] += 1
                results.append(dict(knowledge))
        
        results.sort(key=lambda x: -x["access_count"])
        
        return results[:10]
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        获取知识统计信息
        
        Returns:
            统计信息字典
        """
        by_type = defaultdict(int)
        total_access = 0
        
        for knowledge in self.shared_knowledge.values():
            by_type[knowledge["knowledge_type"]] += 1
            total_access += knowledge["access_count"]
        
        return {
            "total_knowledge": len(self.shared_knowledge),
            "knowledge_by_type": dict(by_type),
            "total_access": total_access,
            "learning_events": len(self.learning_events),
        }


class EnhancedAgentSwarm(AgentSwarm):
    """
    增强多智能体协作网络
    
    在原有基础上增强：
    - 智能体能力匹配与任务分配优化
    - 自适应协作策略
    - 动态负载均衡
    - 智能体信誉系统
    - 协作学习机制
    - 分布式共识算法
    
    Attributes:
        capability_matcher: 能力匹配器
        reputation_system: 信誉系统
        collaborative_learner: 协作学习器
        task_priority_queue: 优先级任务队列
    """
    
    def __init__(self, swarm_id: str = None, 
                 mode: CollaborationMode = CollaborationMode.HYBRID):
        super().__init__(swarm_id=swarm_id, mode=mode)
        
        self.capability_matcher = CapabilityMatcher()
        self.reputation_system = ReputationSystem()
        self.collaborative_learner = CollaborativeLearner()
        
        self.task_priority_queue: Dict[TaskPriority, deque] = {
            TaskPriority.LOW: deque(maxlen=100),
            TaskPriority.MEDIUM: deque(maxlen=100),
            TaskPriority.HIGH: deque(maxlen=100),
            TaskPriority.CRITICAL: deque(maxlen=100),
        }
        
        self._agent_capabilities: Dict[str, List[str]] = {}
    
    def register_agent(self, name: str, role: AgentRole = AgentRole.WORKER,
                       capabilities: List[str] = None,
                       agent_id: str = None) -> AgentInfo:
        """
        注册智能体（增强版）
        
        Args:
            name: 智能体名称
            role: 智能体角色
            capabilities: 能力列表
            agent_id: 智能体ID（可选）
        
        Returns:
            AgentInfo: 智能体信息
        """
        agent = super().register_agent(name, role, capabilities, agent_id)
        self._agent_capabilities[agent.agent_id] = capabilities or []
        self.capability_matcher.build_taxonomy(self._agent_capabilities)
        return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        注销智能体（增强版）
        
        Args:
            agent_id: 智能体ID
        
        Returns:
            bool: 是否成功
        """
        result = super().unregister_agent(agent_id)
        if result and agent_id in self._agent_capabilities:
            del self._agent_capabilities[agent_id]
            self.capability_matcher.build_taxonomy(self._agent_capabilities)
        return result
    
    def assign_task_optimized(self, task: TaskInfo) -> Optional[str]:
        """
        优化的任务分配
        
        根据能力匹配、信誉评分、负载情况综合决策。
        
        Args:
            task: 任务信息
        
        Returns:
            str: 分配的智能体ID
        """
        best_agent = self.capability_matcher.find_best_match(
            task.requires, self.agents
        )
        
        if not best_agent:
            available_agents = [
                aid for aid, agent in self.agents.items()
                if agent.status == AgentStatus.IDLE
            ]
            
            if available_agents:
                best_agent = max(available_agents, 
                               key=lambda aid: self.reputation_system.get_overall_score(aid))
        
        if best_agent:
            self.agents[best_agent].current_task = task.task_id
            self.agents[best_agent].status = AgentStatus.WORKING
            self.agents[best_agent].workload = min(1.0, self.agents[best_agent].workload + 0.2)
            
            task.assigned_to = best_agent
            task.status = "assigned"
            
            self.task_priority_queue[task.priority].append(task)
            
            return best_agent
        
        return None
    
    def complete_task_with_reputation(self, task_id: str, agent_id: str, 
                                     result: Dict[str, Any] = None) -> None:
        """
        完成任务并更新信誉
        
        Args:
            task_id: 任务ID
            agent_id: 智能体ID
            result: 任务结果
        """
        super().complete_task(task_id, agent_id, result)
        
        success = result.get("success", True) if result else True
        quality = result.get("quality", 0.5) if result else 0.5
        response_time = result.get("response_time", 0.0) if result else 0.0
        
        self.reputation_system.record_task_completion(
            agent_id, task_id, success, quality, response_time
        )
    
    def balance_load(self) -> Dict[str, Any]:
        """
        动态负载均衡
        
        将任务从高负载智能体迁移到低负载智能体。
        
        Returns:
            负载均衡结果
        """
        if len(self.agents) < 2:
            return {"message": "Not enough agents for load balancing"}
        
        with self._lock:
            agents_list = list(self.agents.values())
            avg_workload = np.mean([a.workload for a in agents_list])
            
            overloaded = [
                a for a in agents_list 
                if a.workload > avg_workload + 0.2 and a.current_task
            ]
            
            underloaded = [
                a for a in agents_list 
                if a.workload < avg_workload - 0.1 and a.status == AgentStatus.IDLE
            ]
            
            migrations = []
            
            for over_agent in overloaded:
                if not underloaded:
                    break
                
                under_agent = underloaded.pop(0)
                
                migrated_task = over_agent.current_task
                over_agent.current_task = None
                over_agent.status = AgentStatus.IDLE
                over_agent.workload = max(0.0, over_agent.workload - 0.2)
                
                under_agent.current_task = migrated_task
                under_agent.status = AgentStatus.WORKING
                under_agent.workload = min(1.0, under_agent.workload + 0.2)
                
                migrations.append({
                    "from": over_agent.agent_id,
                    "to": under_agent.agent_id,
                    "task": migrated_task,
                })
        
        return {
            "migrations": migrations,
            "overloaded_count": len(overloaded),
            "underloaded_count": len(underloaded) + len(migrations),
        }
    
    def share_knowledge(self, agent_id: str, knowledge_type: str, 
                       content: Any, context: Optional[Dict] = None) -> str:
        """
        共享知识到协作学习系统
        
        Args:
            agent_id: 智能体ID
            knowledge_type: 知识类型
            content: 知识内容
            context: 上下文信息
        
        Returns:
            str: 知识ID
        """
        return self.collaborative_learner.share_knowledge(agent_id, knowledge_type, content, context)
    
    def retrieve_knowledge(self, query: str, 
                          knowledge_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        检索共享知识
        
        Args:
            query: 查询关键词
            knowledge_types: 知识类型过滤
        
        Returns:
            匹配的知识列表
        """
        return self.collaborative_learner.retrieve_knowledge(query, knowledge_types)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        获取增强统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = self.get_swarm_stats()
        
        reputation_scores = {}
        for agent_id in self.agents:
            reputation_scores[agent_id] = self.reputation_system.get_overall_score(agent_id)
        
        knowledge_stats = self.collaborative_learner.get_knowledge_stats()
        
        return {
            **base_stats,
            "reputation_scores": reputation_scores,
            "knowledge_stats": knowledge_stats,
            "task_priority_counts": {
                priority.value: len(queue)
                for priority, queue in self.task_priority_queue.items()
            },
        }


def get_enhanced_swarm(swarm_id: str = None, 
                       mode: CollaborationMode = CollaborationMode.HYBRID) -> EnhancedAgentSwarm:
    """
    获取增强多智能体协作网络实例
    
    Args:
        swarm_id: 网络ID
        mode: 协作模式
    
    Returns:
        EnhancedAgentSwarm 实例
    """
    return EnhancedAgentSwarm(swarm_id=swarm_id, mode=mode)