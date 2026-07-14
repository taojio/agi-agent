import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from enum import Enum


class ConflictType(Enum):
    RESOURCE = "resource"
    TASK = "task"
    DECISION = "decision"
    DATA = "data"
    PRIORITY = "priority"


class ResolutionStrategy(Enum):
    VOTING = "voting"
    LEADER_DECIDE = "leader_decide"
    PERFORMANCE_BASED = "performance_based"
    COMPROMISE = "compromise"
    ESCALATE = "escalate"
    RANDOM = "random"


class ConflictResolver:
    def __init__(self, default_strategy: ResolutionStrategy = ResolutionStrategy.VOTING):
        self.default_strategy = default_strategy
        self.conflict_history: deque = deque(maxlen=100)
        self.resolution_log: deque = deque(maxlen=100)

        self._strategies = {
            ResolutionStrategy.VOTING: self._resolve_by_voting,
            ResolutionStrategy.LEADER_DECIDE: self._resolve_by_leader,
            ResolutionStrategy.PERFORMANCE_BASED: self._resolve_by_performance,
            ResolutionStrategy.COMPROMISE: self._resolve_by_compromise,
            ResolutionStrategy.RANDOM: self._resolve_random,
        }

        self._total_conflicts = 0
        self._resolved_conflicts = 0

    def detect_conflicts(self, tasks: List[Dict[str, Any]],
                          agents: List[Any]) -> List[Dict[str, Any]]:
        conflicts = []

        resource_conflicts = self._detect_resource_conflicts(tasks, agents)
        conflicts.extend(resource_conflicts)

        priority_conflicts = self._detect_priority_conflicts(tasks)
        conflicts.extend(priority_conflicts)

        return conflicts

    def _detect_resource_conflicts(self, tasks: List[Dict[str, Any]],
                                    agents: List[Any]) -> List[Dict[str, Any]]:
        conflicts = []

        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i >= j:
                    continue

                caps1 = set(task1.get("required_capabilities", []))
                caps2 = set(task2.get("required_capabilities", []))
                shared_caps = caps1 & caps2

                if shared_caps:
                    agents_with_cap = [
                        a for a in agents
                        if shared_caps.issubset(set(a.capabilities))
                    ]

                    if len(agents_with_cap) < 2:
                        conflicts.append({
                            "conflict_id": f"conflict_res_{i}_{j}",
                            "type": ConflictType.RESOURCE,
                            "parties": [task1.get("task_id"), task2.get("task_id")],
                            "resource": shared_caps,
                            "severity": len(shared_caps) / max(len(caps1 | caps2), 1)
                        })

        return conflicts

    def _detect_priority_conflicts(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts = []

        high_priority_tasks = [
            t for t in tasks
            if t.get("priority", 2) <= 1
        ]

        if len(high_priority_tasks) > 3:
            conflicts.append({
                "conflict_id": "conflict_priority_overload",
                "type": ConflictType.PRIORITY,
                "parties": [t.get("task_id") for t in high_priority_tasks],
                "severity": min(1.0, len(high_priority_tasks) / 5.0)
            })

        return conflicts

    def resolve_conflict(self, conflict: Dict[str, Any],
                         agents: List[Any] = None,
                         leader_id: str = None,
                         strategy: ResolutionStrategy = None) -> Dict[str, Any]:
        self._total_conflicts += 1
        strat = strategy or self.default_strategy
        resolver = self._strategies.get(strat, self._resolve_by_voting)

        result = resolver(conflict, agents or [], leader_id)

        self.conflict_history.append(conflict)
        self.resolution_log.append({
            "conflict_id": conflict.get("conflict_id"),
            "strategy": strat.value,
            "result": result,
            "timestamp": time.time()
        })

        if result.get("resolved"):
            self._resolved_conflicts += 1

        return result

    def _resolve_by_voting(self, conflict: Dict[str, Any],
                            agents: List[Any], leader_id: str) -> Dict[str, Any]:
        parties = conflict.get("parties", [])
        if not parties:
            return {"resolved": True, "winner": None, "reason": "no_parties"}

        votes = {}
        for agent in agents:
            vote = parties[hash(agent.agent_id) % len(parties)]
            votes[vote] = votes.get(vote, 0) + 1

        if votes:
            winner = max(votes, key=votes.get)
            return {
                "resolved": True,
                "winner": winner,
                "votes": votes,
                "method": "voting"
            }

        return {"resolved": False, "reason": "no_votes"}

    def _resolve_by_leader(self, conflict: Dict[str, Any],
                            agents: List[Any], leader_id: str) -> Dict[str, Any]:
        parties = conflict.get("parties", [])

        if not leader_id:
            return {"resolved": False, "reason": "no_leader"}

        if not parties:
            return {"resolved": True, "winner": None, "reason": "no_parties"}

        leader_idx = hash(leader_id) % len(parties)
        winner = parties[leader_idx]

        return {
            "resolved": True,
            "winner": winner,
            "decided_by": leader_id,
            "method": "leader_decide"
        }

    def _resolve_by_performance(self, conflict: Dict[str, Any],
                                 agents: List[Any], leader_id: str) -> Dict[str, Any]:
        parties = conflict.get("parties", [])
        if not parties:
            return {"resolved": True, "winner": None, "reason": "no_parties"}

        scores = {}
        for i, party in enumerate(parties):
            if agents and i < len(agents):
                scores[party] = agents[i].performance_score
            else:
                scores[party] = 0.5

        winner = max(scores, key=scores.get)

        return {
            "resolved": True,
            "winner": winner,
            "scores": scores,
            "method": "performance_based"
        }

    def _resolve_by_compromise(self, conflict: Dict[str, Any],
                                agents: List[Any], leader_id: str) -> Dict[str, Any]:
        parties = conflict.get("parties", [])

        return {
            "resolved": True,
            "winner": "compromise",
            "parties_involved": parties,
            "compromise_ratio": 1.0 / max(len(parties), 1),
            "method": "compromise"
        }

    def _resolve_random(self, conflict: Dict[str, Any],
                         agents: List[Any], leader_id: str) -> Dict[str, Any]:
        import random
        parties = conflict.get("parties", [])

        if not parties:
            return {"resolved": False, "reason": "no_parties"}

        winner = random.choice(parties)

        return {
            "resolved": True,
            "winner": winner,
            "method": "random"
        }

    def get_resolution_stats(self) -> Dict[str, Any]:
        return {
            "total_conflicts": self._total_conflicts,
            "resolved_conflicts": self._resolved_conflicts,
            "resolution_rate": self._resolved_conflicts / max(self._total_conflicts, 1),
            "default_strategy": self.default_strategy.value,
            "available_strategies": [s.value for s in ResolutionStrategy]
        }
