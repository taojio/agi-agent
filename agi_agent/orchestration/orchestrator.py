"""
orchestration/orchestrator.py - 编排引擎

统一的任务编排执行引擎
"""
import time
import logging
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from .event_bus import EventBus, Event
from .task_dag import TaskDAG, TaskNode, TaskStatus, TaskResult

logger = logging.getLogger(__name__)


class OrchestratorEngine:
    """编排引擎

    功能：
    - DAG 任务调度执行
    - 支持串行/并行执行
    - 结果收集与聚合
    - 错误处理与重试
    - 执行监控
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_workers: int = 4,
    ):
        self.event_bus = event_bus or EventBus()
        self.dag = TaskDAG(self.event_bus)
        self.max_workers = max_workers
        self._results: Dict[str, Any] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    def add_task(
        self,
        task_id: str,
        func: Callable,
        name: str = "",
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskNode:
        """添加任务"""
        return self.dag.add_task(
            task_id=task_id,
            name=name or task_id,
            func=func,
            dependencies=dependencies,
            priority=priority,
            max_retries=max_retries,
            metadata=metadata,
        )

    def execute_sync(self) -> Dict[str, Any]:
        """同步串行执行所有任务

        按拓扑顺序执行，单个任务失败不会中断其他任务
        """
        self._results.clear()
        order = self.dag.topological_sort()

        for task_id in order:
            task = self.dag.get_task(task_id)
            if not task or not task.func:
                continue
            if task.status == TaskStatus.COMPLETED:
                continue

            try:
                self._execute_task(task)
            except Exception:
                pass

        self._emit("orchestration.completed", {"mode": "sync", "stats": self.dag.get_stats()})
        return dict(self._results)

    def execute_parallel(self) -> Dict[str, Any]:
        """并行执行所有任务

        使用线程池并行执行就绪任务
        """
        self._results.clear()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        futures: Dict[str, Future] = {}

        try:
            while not self.dag.is_complete():
                ready = self.dag.get_ready_tasks()
                ready = [t for t in ready if t.task_id not in futures]

                for task in ready:
                    if task.func:
                        self.dag.mark_running(task.task_id)
                        future = self._executor.submit(self._execute_task, task)
                        futures[task.task_id] = future

                if not ready and futures:
                    done_ids = []
                    for tid, f in list(futures.items()):
                        if f.done():
                            done_ids.append(tid)
                    for tid in done_ids:
                        del futures[tid]

                if not ready and not futures:
                    break

                if futures:
                    time.sleep(0.01)

        finally:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._emit("orchestration.completed", {"mode": "parallel", "stats": self.dag.get_stats()})
        return dict(self._results)

    def _execute_task(self, task: TaskNode) -> Any:
        """执行单个任务"""
        task_id = task.task_id
        start_time = time.time()

        try:
            self.dag.mark_running(task_id)

            if task.func:
                result = task.func()
            else:
                result = None

            self._results[task_id] = result
            self.dag.mark_completed(task_id, result)

            duration = time.time() - start_time
            logger.debug(f"Task {task_id} completed in {duration:.4f}s")
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task_id} failed: {error_msg}")
            self.dag.mark_failed(task_id, error_msg)
            raise

    def reset(self) -> None:
        """重置所有任务状态"""
        self._results.clear()
        self.dag.reset_all()

    def get_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self._results.get(task_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            "results_count": len(self._results),
            "dag_stats": self.dag.get_stats(),
            "max_workers": self.max_workers,
        }

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        self.event_bus.publish(Event(
            event_type=event_type,
            data=data,
            source="orchestrator",
        ))
