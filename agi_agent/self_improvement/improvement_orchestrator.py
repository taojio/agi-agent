import time
import uuid
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from collections import deque
from ..utils.logger import setup_logger


class WorkflowNodeType(Enum):
    START = "start"
    END = "end"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    TASK = "task"
    SUBWORKFLOW = "subworkflow"


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowEvent(Enum):
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_PAUSED = "workflow_paused"


@dataclass
class WorkflowNode:
    node_id: str
    type: WorkflowNodeType
    name: str
    description: str = ""
    task: Optional[Callable] = None
    condition: Optional[Callable] = None
    next_nodes: List[str] = field(default_factory=list)
    true_node: Optional[str] = None
    false_node: Optional[str] = None
    parallel_nodes: List[str] = field(default_factory=list)
    subworkflow: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "next_nodes": self.next_nodes,
            "true_node": self.true_node,
            "false_node": self.false_node,
            "parallel_nodes": self.parallel_nodes,
            "subworkflow": self.subworkflow,
            "params": self.params,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time - self.start_time) if self.start_time and self.end_time else None,
        }


@dataclass
class WorkflowInstance:
    instance_id: str
    workflow_id: str
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status.value,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "variables": self.variables,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "duration": (self.end_time - self.start_time) if self.start_time and self.end_time else None,
        }


@dataclass
class RollbackTrigger:
    metric_name: str
    operator: str
    threshold: float
    description: str

    def check(self, value: float) -> bool:
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 0.0001
        return False


@dataclass
class RollbackConfig:
    triggers: List[RollbackTrigger]
    rollback_workflow_id: Optional[str] = None
    max_retries: int = 3
    cooldown_period: float = 300.0


class ImprovementWorkflow:
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.nodes: Dict[str, WorkflowNode] = {}
        self.start_node_id: Optional[str] = None
        self.end_node_id: Optional[str] = None

    def add_node(self, node: WorkflowNode):
        self.nodes[node.node_id] = node
        if node.type == WorkflowNodeType.START:
            self.start_node_id = node.node_id
        elif node.type == WorkflowNodeType.END:
            self.end_node_id = node.node_id

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
        }


class ImprovementOrchestrator:
    def __init__(self):
        self.logger = setup_logger("improvement_orchestrator")
        self.workflows: Dict[str, ImprovementWorkflow] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.instance_history: deque = deque(maxlen=100)
        self._event_handlers: Dict[WorkflowEvent, List[Callable]] = {}
        self._rollback_configs: Dict[str, RollbackConfig] = {}

        self._register_default_handlers()
        self._create_default_workflows()

    def _register_default_handlers(self):
        self.register_event_handler(WorkflowEvent.WORKFLOW_COMPLETED, self._on_workflow_completed)
        self.register_event_handler(WorkflowEvent.WORKFLOW_FAILED, self._on_workflow_failed)
        self.register_event_handler(WorkflowEvent.WORKFLOW_CANCELLED, self._on_workflow_cancelled)

    def register_event_handler(self, event: WorkflowEvent, handler: Callable):
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _trigger_event(self, event: WorkflowEvent, **kwargs):
        for handler in self._event_handlers.get(event, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event.value}: {e}")

    def _on_workflow_completed(self, **kwargs):
        self.logger.info(f"Workflow completed: {kwargs.get('instance_id')}")

    def _on_workflow_failed(self, **kwargs):
        self.logger.error(f"Workflow failed: {kwargs.get('instance_id')}, error: {kwargs.get('error')}")

    def _on_workflow_cancelled(self, **kwargs):
        self.logger.warning(f"Workflow cancelled: {kwargs.get('instance_id')}")

    def _create_default_workflows(self):
        self._create_self_improvement_workflow()
        self._create_rollback_workflow()

    def _create_self_improvement_workflow(self):
        workflow = ImprovementWorkflow(
            workflow_id="self_improvement_main",
            name="Self Improvement Main Workflow",
            description="Main workflow for automated self-improvement cycle"
        )

        workflow.add_node(WorkflowNode(
            node_id="start",
            type=WorkflowNodeType.START,
            name="Start",
            description="Begin improvement cycle",
            next_nodes=["diagnosis"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="diagnosis",
            type=WorkflowNodeType.TASK,
            name="Diagnosis",
            description="Run system diagnosis to identify issues",
            next_nodes=["proposal"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="proposal",
            type=WorkflowNodeType.TASK,
            name="Generate Proposals",
            description="Generate improvement proposals based on diagnosis",
            next_nodes=["verification"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="verification",
            type=WorkflowNodeType.TASK,
            name="Verify Proposals",
            description="Verify proposals for safety and validity",
            next_nodes=["has_valid_proposals"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="has_valid_proposals",
            type=WorkflowNodeType.CONDITIONAL,
            name="Check Valid Proposals",
            description="Check if there are valid proposals to proceed",
            true_node="parallel_tests",
            false_node="end_no_proposals"
        ))

        workflow.add_node(WorkflowNode(
            node_id="parallel_tests",
            type=WorkflowNodeType.PARALLEL,
            name="Parallel Testing",
            description="Run parallel tests (unit, integration, security)",
            parallel_nodes=["unit_test", "integration_test", "security_test"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="unit_test",
            type=WorkflowNodeType.TASK,
            name="Unit Tests",
            description="Run unit tests",
            next_nodes=["test_aggregation"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="integration_test",
            type=WorkflowNodeType.TASK,
            name="Integration Tests",
            description="Run integration tests",
            next_nodes=["test_aggregation"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="security_test",
            type=WorkflowNodeType.TASK,
            name="Security Tests",
            description="Run security tests",
            next_nodes=["test_aggregation"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="test_aggregation",
            type=WorkflowNodeType.TASK,
            name="Aggregate Test Results",
            description="Aggregate and analyze test results",
            next_nodes=["tests_passed"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="tests_passed",
            type=WorkflowNodeType.CONDITIONAL,
            name="Check Tests Passed",
            description="Check if all tests passed",
            true_node="apply_changes",
            false_node="rollback_needed"
        ))

        workflow.add_node(WorkflowNode(
            node_id="apply_changes",
            type=WorkflowNodeType.TASK,
            name="Apply Changes",
            description="Apply improvement changes",
            next_nodes=["evaluation"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="evaluation",
            type=WorkflowNodeType.TASK,
            name="Evaluate Results",
            description="Evaluate improvement effectiveness",
            next_nodes=["improvement_successful"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="improvement_successful",
            type=WorkflowNodeType.CONDITIONAL,
            name="Check Improvement Success",
            description="Check if improvement met threshold",
            true_node="end_success",
            false_node="rollback_needed"
        ))

        workflow.add_node(WorkflowNode(
            node_id="rollback_needed",
            type=WorkflowNodeType.TASK,
            name="Trigger Rollback",
            description="Trigger rollback workflow",
            next_nodes=["end_rolled_back"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="end_success",
            type=WorkflowNodeType.END,
            name="Success",
            description="Improvement completed successfully"
        ))

        workflow.add_node(WorkflowNode(
            node_id="end_rolled_back",
            type=WorkflowNodeType.END,
            name="Rolled Back",
            description="Changes rolled back"
        ))

        workflow.add_node(WorkflowNode(
            node_id="end_no_proposals",
            type=WorkflowNodeType.END,
            name="No Proposals",
            description="No valid proposals to process"
        ))

        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Created default workflow: {workflow.name}")

    def _create_rollback_workflow(self):
        workflow = ImprovementWorkflow(
            workflow_id="rollback_workflow",
            name="Rollback Workflow",
            description="Workflow for rolling back changes"
        )

        workflow.add_node(WorkflowNode(
            node_id="start",
            type=WorkflowNodeType.START,
            name="Start Rollback",
            next_nodes=["create_backup"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="create_backup",
            type=WorkflowNodeType.TASK,
            name="Create Backup",
            description="Create backup of current state",
            next_nodes=["revert_changes"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="revert_changes",
            type=WorkflowNodeType.TASK,
            name="Revert Changes",
            description="Revert to previous version",
            next_nodes=["verify_revert"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="verify_revert",
            type=WorkflowNodeType.TASK,
            name="Verify Revert",
            description="Verify system state after revert",
            next_nodes=["rollback_completed"]
        ))

        workflow.add_node(WorkflowNode(
            node_id="rollback_completed",
            type=WorkflowNodeType.END,
            name="Rollback Completed",
            description="Rollback finished successfully"
        ))

        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Created rollback workflow: {workflow.name}")

    def register_workflow(self, workflow: ImprovementWorkflow):
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow.name}")

    def get_workflow(self, workflow_id: str) -> Optional[ImprovementWorkflow]:
        return self.workflows.get(workflow_id)

    def create_instance(self, workflow_id: str, variables: Dict[str, Any] = None) -> WorkflowInstance:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        instance_id = f"inst_{uuid.uuid4().hex[:8]}"
        nodes = {node_id: node.__deepcopy__() for node_id, node in workflow.nodes.items()}

        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            name=workflow.name,
            nodes=nodes,
            variables=variables or {},
            status=WorkflowStatus.PENDING
        )

        self.instances[instance_id] = instance
        return instance

    async def execute_instance(self, instance_id: str) -> WorkflowInstance:
        instance = self.instances.get(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")

        instance.status = WorkflowStatus.RUNNING
        instance.start_time = time.time()

        try:
            await self._execute_node(instance, instance.nodes.get(instance_id), instance.start_node_id)
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            instance.end_time = time.time()
            self._trigger_event(WorkflowEvent.WORKFLOW_FAILED, instance_id=instance_id, error=str(e))

        instance.end_time = time.time()
        self.instance_history.append(instance)

        if instance.status == WorkflowStatus.COMPLETED:
            self._trigger_event(WorkflowEvent.WORKFLOW_COMPLETED, instance_id=instance_id)

        return instance

    async def _execute_node(self, instance: WorkflowInstance, node_id: str):
        node = instance.nodes.get(node_id)
        if not node:
            return

        node.status = WorkflowStatus.RUNNING
        node.start_time = time.time()

        try:
            if node.type == WorkflowNodeType.START:
                await self._execute_start(node, instance)
            elif node.type == WorkflowNodeType.TASK:
                await self._execute_task(node, instance)
            elif node.type == WorkflowNodeType.CONDITIONAL:
                await self._execute_conditional(node, instance)
            elif node.type == WorkflowNodeType.PARALLEL:
                await self._execute_parallel(node, instance)
            elif node.type == WorkflowNodeType.END:
                await self._execute_end(node, instance)
            elif node.type == WorkflowNodeType.SUBWORKFLOW:
                await self._execute_subworkflow(node, instance)

            node.status = WorkflowStatus.COMPLETED
        except Exception as e:
            node.status = WorkflowStatus.FAILED
            node.error = str(e)
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            self._trigger_event(WorkflowEvent.NODE_FAILED, instance_id=instance.instance_id, node_id=node_id, error=str(e))
            return

        node.end_time = time.time()
        self._trigger_event(WorkflowEvent.NODE_COMPLETED, instance_id=instance.instance_id, node_id=node_id)

    async def _execute_start(self, node: WorkflowNode, instance: WorkflowInstance):
        for next_node_id in node.next_nodes:
            await self._execute_node(instance, next_node_id)

    async def _execute_task(self, node: WorkflowNode, instance: WorkflowInstance):
        if node.task:
            try:
                result = node.task(instance.variables)
                if asyncio.iscoroutine(result):
                    result = await result
                node.result = result
                instance.variables[node.node_id] = result
            except Exception as e:
                node.result = None
                node.error = str(e)
                raise

        for next_node_id in node.next_nodes:
            await self._execute_node(instance, next_node_id)

    async def _execute_conditional(self, node: WorkflowNode, instance: WorkflowInstance):
        if node.condition:
            try:
                condition_result = node.condition(instance.variables)
                if asyncio.iscoroutine(condition_result):
                    condition_result = await condition_result
                node.result = condition_result

                if condition_result:
                    if node.true_node:
                        await self._execute_node(instance, node.true_node)
                else:
                    if node.false_node:
                        await self._execute_node(instance, node.false_node)
            except Exception as e:
                node.result = None
                node.error = str(e)
                if node.false_node:
                    await self._execute_node(instance, node.false_node)
        else:
            for next_node_id in node.next_nodes:
                await self._execute_node(instance, next_node_id)

    async def _execute_parallel(self, node: WorkflowNode, instance: WorkflowInstance):
        tasks = []
        for parallel_node_id in node.parallel_nodes:
            tasks.append(self._execute_node(instance, parallel_node_id))

        if tasks:
            await asyncio.gather(*tasks)

        for next_node_id in node.next_nodes:
            await self._execute_node(instance, next_node_id)

    async def _execute_end(self, node: WorkflowNode, instance: WorkflowInstance):
        instance.status = WorkflowStatus.COMPLETED

    async def _execute_subworkflow(self, node: WorkflowNode, instance: WorkflowInstance):
        if node.subworkflow:
            sub_instance = self.create_instance(node.subworkflow, instance.variables)
            await self.execute_instance(sub_instance.instance_id)
            node.result = sub_instance.status.value
            instance.variables[node.node_id] = sub_instance.to_dict()

        for next_node_id in node.next_nodes:
            await self._execute_node(instance, next_node_id)

    def register_rollback_config(self, workflow_id: str, config: RollbackConfig):
        self._rollback_configs[workflow_id] = config

    def check_rollback_triggers(self, workflow_id: str, metrics: Dict[str, float]) -> bool:
        config = self._rollback_configs.get(workflow_id)
        if not config:
            return False

        for trigger in config.triggers:
            value = metrics.get(trigger.metric_name)
            if value is not None and trigger.check(value):
                self.logger.warning(f"Rollback trigger activated: {trigger.description}")
                return True

        return False

    async def trigger_rollback(self, workflow_id: str, metrics: Dict[str, float]) -> bool:
        if not self.check_rollback_triggers(workflow_id, metrics):
            return False

        config = self._rollback_configs.get(workflow_id)
        if not config or not config.rollback_workflow_id:
            return False

        rollback_instance = self.create_instance(config.rollback_workflow_id, {"triggered_by": workflow_id, "metrics": metrics})
        await self.execute_instance(rollback_instance.instance_id)

        return rollback_instance.status == WorkflowStatus.COMPLETED

    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        return self.instances.get(instance_id)

    def get_instance_history(self, limit: int = 20) -> List[WorkflowInstance]:
        return list(self.instance_history)[-limit:]

    def get_status(self) -> Dict[str, Any]:
        running_count = sum(1 for inst in self.instances.values() if inst.status == WorkflowStatus.RUNNING)
        completed_count = sum(1 for inst in self.instance_history if inst.status == WorkflowStatus.COMPLETED)
        failed_count = sum(1 for inst in self.instance_history if inst.status == WorkflowStatus.FAILED)

        return {
            "workflows_count": len(self.workflows),
            "active_instances_count": len(self.instances),
            "running_instances_count": running_count,
            "history_count": len(self.instance_history),
            "completed_count": completed_count,
            "failed_count": failed_count,
        }


_global_orchestrator: Optional[ImprovementOrchestrator] = None


def get_improvement_orchestrator() -> ImprovementOrchestrator:
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ImprovementOrchestrator()
    return _global_orchestrator