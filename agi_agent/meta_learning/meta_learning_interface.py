import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class InterfaceProtocol(Enum):
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"


class DataFormat(Enum):
    NUMPY = "numpy"
    TENSOR = "tensor"
    LIST = "list"
    DICT = "dict"
    JSON = "json"


class MessageType(Enum):
    TASK_REGISTER = "task_register"
    STRATEGY_REQUEST = "strategy_request"
    PARAMETER_UPDATE = "parameter_update"
    PERFORMANCE_REPORT = "performance_report"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    ADAPTATION_COMPLETE = "adaptation_complete"


class MetaLearningMessage:
    def __init__(self, message_type: MessageType, content: Dict[str, Any],
                 sender: str, receiver: str, timestamp: int = None):
        self.message_type = message_type
        self.content = content
        self.sender = sender
        self.receiver = receiver
        self.timestamp = timestamp if timestamp else np.random.randint(1000000)
        self.message_id = f"{self.sender}_{self.receiver}_{self.timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaLearningMessage':
        return cls(
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            sender=data["sender"],
            receiver=data["receiver"],
            timestamp=data.get("timestamp")
        )


class TaskSpecification(ABC):
    @abstractmethod
    def get_task_id(self) -> str:
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def get_meta_features(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class StrategySpecification(ABC):
    @abstractmethod
    def get_strategy_id(self) -> str:
        pass

    @abstractmethod
    def get_strategy_type(self) -> str:
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class AdaptationRequest:
    def __init__(self, task_spec: TaskSpecification,
                 strategy_spec: Optional[StrategySpecification] = None,
                 num_inner_iterations: int = 5,
                 learning_rate: float = 0.01):
        self.task_spec = task_spec
        self.strategy_spec = strategy_spec
        self.num_inner_iterations = num_inner_iterations
        self.learning_rate = learning_rate
        self.request_id = f"adapt_{np.random.randint(1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_spec": self.task_spec.to_dict(),
            "strategy_spec": self.strategy_spec.to_dict() if self.strategy_spec else None,
            "num_inner_iterations": self.num_inner_iterations,
            "learning_rate": self.learning_rate
        }


class AdaptationResponse:
    def __init__(self, request_id: str, success: bool,
                 adapted_model: Optional[Dict[str, Any]] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 adaptation_time_ms: float = 0.0,
                 error_message: str = ""):
        self.request_id = request_id
        self.success = success
        self.adapted_model = adapted_model
        self.metrics = metrics
        self.adaptation_time_ms = adaptation_time_ms
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "adapted_model": self.adapted_model,
            "metrics": self.metrics,
            "adaptation_time_ms": self.adaptation_time_ms,
            "error_message": self.error_message
        }


class ParameterUpdateRequest:
    def __init__(self, parameter_type: str, value: float,
                 reason: str = "", context: Dict[str, Any] = None):
        self.parameter_type = parameter_type
        self.value = value
        self.reason = reason
        self.context = context or {}
        self.update_id = f"param_{np.random.randint(1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "parameter_type": self.parameter_type,
            "value": self.value,
            "reason": self.reason,
            "context": self.context
        }


class PerformanceReport:
    def __init__(self, task_id: str, strategy_id: str,
                 accuracy: float, loss: float,
                 train_time_ms: float, sample_efficiency: float,
                 success: bool, metadata: Dict[str, Any] = None):
        self.task_id = task_id
        self.strategy_id = strategy_id
        self.accuracy = accuracy
        self.loss = loss
        self.train_time_ms = train_time_ms
        self.sample_efficiency = sample_efficiency
        self.success = success
        self.metadata = metadata or {}
        self.report_id = f"perf_{np.random.randint(1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "task_id": self.task_id,
            "strategy_id": self.strategy_id,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "train_time_ms": self.train_time_ms,
            "sample_efficiency": self.sample_efficiency,
            "success": self.success,
            "metadata": self.metadata
        }


class KnowledgeTransferRequest:
    def __init__(self, source_task_id: str, target_task_id: str,
                 transfer_type: str = "direct",
                 confidence_threshold: float = 0.5):
        self.source_task_id = source_task_id
        self.target_task_id = target_task_id
        self.transfer_type = transfer_type
        self.confidence_threshold = confidence_threshold
        self.transfer_id = f"transfer_{np.random.randint(1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "source_task_id": self.source_task_id,
            "target_task_id": self.target_task_id,
            "transfer_type": self.transfer_type,
            "confidence_threshold": self.confidence_threshold
        }


class KnowledgeTransferResponse:
    def __init__(self, transfer_id: str, success: bool,
                 similarity: float, effectiveness: float,
                 transferred_rules: List[str] = None,
                 recommended_adaptation_steps: int = 5):
        self.transfer_id = transfer_id
        self.success = success
        self.similarity = similarity
        self.effectiveness = effectiveness
        self.transferred_rules = transferred_rules or []
        self.recommended_adaptation_steps = recommended_adaptation_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "success": self.success,
            "similarity": self.similarity,
            "effectiveness": self.effectiveness,
            "transferred_rules": self.transferred_rules,
            "recommended_adaptation_steps": self.recommended_adaptation_steps
        }


class MetaLearningInterface(ABC):
    @abstractmethod
    def register_task(self, task_spec: TaskSpecification) -> str:
        pass

    @abstractmethod
    def request_strategy(self, task_id: str,
                        context: Dict[str, Any] = None) -> StrategySpecification:
        pass

    @abstractmethod
    def adapt(self, request: AdaptationRequest) -> AdaptationResponse:
        pass

    @abstractmethod
    def update_parameters(self, request: ParameterUpdateRequest) -> bool:
        pass

    @abstractmethod
    def report_performance(self, report: PerformanceReport) -> bool:
        pass

    @abstractmethod
    def transfer_knowledge(self, request: KnowledgeTransferRequest) -> KnowledgeTransferResponse:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass


class MessageValidator:
    @staticmethod
    def validate_message(message: MetaLearningMessage) -> Tuple[bool, str]:
        required_fields = ["message_type", "content", "sender", "receiver"]
        for field in required_fields:
            if field not in message.to_dict():
                return False, f"Missing required field: {field}"

        if not isinstance(message.content, dict):
            return False, "Content must be a dictionary"

        if len(message.sender) == 0 or len(message.receiver) == 0:
            return False, "Sender and receiver cannot be empty"

        return True, "Valid"

    @staticmethod
    def validate_task_spec(task_spec: TaskSpecification) -> Tuple[bool, str]:
        if not task_spec.get_task_id():
            return False, "Task ID cannot be empty"
        if not task_spec.get_task_type():
            return False, "Task type cannot be empty"
        if task_spec.get_input_dim() < 0:
            return False, "Input dimension must be non-negative"
        if task_spec.get_output_dim() < 0:
            return False, "Output dimension must be non-negative"

        return True, "Valid"

    @staticmethod
    def validate_adaptation_request(request: AdaptationRequest) -> Tuple[bool, str]:
        if not request.task_spec:
            return False, "Task specification is required"
        if request.num_inner_iterations < 1:
            return False, "Number of inner iterations must be at least 1"
        if request.learning_rate <= 0:
            return False, "Learning rate must be positive"

        return True, "Valid"

    @staticmethod
    def validate_performance_report(report: PerformanceReport) -> Tuple[bool, str]:
        if not report.task_id:
            return False, "Task ID is required"
        if not report.strategy_id:
            return False, "Strategy ID is required"
        if report.accuracy < 0 or report.accuracy > 1:
            return False, "Accuracy must be between 0 and 1"
        if report.loss < 0:
            return False, "Loss must be non-negative"
        if report.train_time_ms < 0:
            return False, "Training time must be non-negative"
        if report.sample_efficiency < 0 or report.sample_efficiency > 1:
            return False, "Sample efficiency must be between 0 and 1"

        return True, "Valid"


class DataConverter:
    @staticmethod
    def to_numpy(data: Any, target_dim: int = None) -> np.ndarray:
        if isinstance(data, np.ndarray):
            if target_dim is not None and data.ndim < target_dim:
                return np.expand_dims(data, axis=0)
            return data
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict):
            return np.array(list(data.values()))
        elif isinstance(data, (int, float)):
            return np.array([data])
        else:
            return np.array([])

    @staticmethod
    def from_numpy(arr: np.ndarray, format: DataFormat) -> Any:
        if format == DataFormat.NUMPY:
            return arr
        elif format == DataFormat.LIST:
            return arr.tolist()
        elif format == DataFormat.DICT:
            return {"data": arr.tolist(), "shape": arr.shape}
        elif format == DataFormat.JSON:
            return {"data": arr.tolist(), "shape": arr.shape}
        else:
            return arr

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(data)
        if norm > 0:
            return data / norm
        return data

    @staticmethod
    def resize(data: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        if data.shape == target_shape:
            return data

        if len(target_shape) == 1:
            return np.resize(data, target_shape)
        else:
            return np.resize(data, target_shape)


class ProtocolHandler:
    def __init__(self, protocol: InterfaceProtocol = InterfaceProtocol.SYNC):
        self.protocol = protocol
        self.pending_requests: Dict[str, Any] = {}
        self.response_callbacks: Dict[str, Callable] = {}

    def send_request(self, request: Any, callback: Optional[Callable] = None) -> str:
        request_id = getattr(request, "request_id", f"req_{np.random.randint(1000000)}")
        self.pending_requests[request_id] = request

        if callback:
            self.response_callbacks[request_id] = callback

        if self.protocol == InterfaceProtocol.SYNC:
            return request_id
        elif self.protocol == InterfaceProtocol.ASYNC:
            return request_id
        else:
            return request_id

    def receive_response(self, response: Any) -> bool:
        request_id = getattr(response, "request_id", None)
        if not request_id or request_id not in self.pending_requests:
            return False

        if request_id in self.response_callbacks:
            self.response_callbacks[request_id](response)
            del self.response_callbacks[request_id]

        del self.pending_requests[request_id]
        return True

    def get_pending_requests(self) -> List[str]:
        return list(self.pending_requests.keys())

    def clear_pending(self):
        self.pending_requests.clear()
        self.response_callbacks.clear()