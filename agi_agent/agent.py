import numpy as np
import torch
import time
import sys
import os
import logging  
from typing import Dict, Any, Optional, List, Callable  

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_agent.config.settings import DEVICE, EVOLVE_TRIGGER_STEP, SAVE_INTERVAL, EVAL_INTERVAL
from agi_agent.utils.metrics import calc_free_energy, calc_entropy, calc_kl_divergence, calc_confidence, calc_novelty, calc_convergence_speed
from agi_agent.perception import GrowingAutoEncoder, MultimodalFusion
from agi_agent.cognitive import CognitiveInferenceLayer, DualSystemCognition, SNNEnhancer, CausalReasoningEngine, UnifiedCognitiveOrchestrator, ArchitectureMutator, SelfModel, GeneralStereoscopicSNN, EnhancedSNN, ModuleSynapticBus, INTERFACE_MAP, SpikingGrowthNetwork, NetworkDimensions
from agi_agent.learning import MetaLearningLayer, KnowledgeGraph, StructuredKnowledgeIngestor
from agi_agent.evolution import EvolutionEngine, MetaSkillGenerator, DualLoopEvolution, EvolutionLevel, QuadLevelEvolution
from agi_agent.execution import ActionExecutionLayer
from agi_agent.meta_cognitive import MetaCognitionLayer
from agi_agent.homeostasis import HomeostasisEngine
from agi_agent.storage import PersistenceManager
from agi_agent.security import SafetyMonitor, ComplianceChecker, HardBoundarySystem, RiskClassifier, CircuitBreaker, AuditTrail
from agi_agent.evaluation import PerformanceEvaluator, MetricsVisualizer
from agi_agent.plugins import PluginManager, PluginHookPoint
from agi_agent.skills import SkillsManager
from agi_agent.decision import AutonomousDecisionEngine, ActionPlanner, ExecutionMonitor, WorldModelDecisionBridge
from agi_agent.self_improvement import RecursiveSelfImprover, PerformanceEvaluator as SelfPerfEvaluator, SelfDiagnosticEngine, ImprovementSafetyVerifier, BootstrappedSelfImprover, AutomatedSelfImprovementLoop

from agi_agent.memory import MemoryHarness, MemoryStore, MemoryTier, MemoryCategory
from agi_agent.soul import SOULParser, SOULModel, IdentityAnchor, GoalTree, BehaviorBoundary, PermissionWhitelist, VersionInfo
from agi_agent.multi_agent import AgentSwarm, WorkspaceManager, HierarchicalDispatcher, WorkspacePermission
from agi_agent.task_engine import DAGEngine, AsyncTaskBoard, CheckpointManager, HeartbeatScheduler, TaskPriority

from agi_agent.reflex import ReflexController
from agi_agent.deliberative import ThinkingOrchestrator, AdvancedReasoner, AbstractionEngine, NeuroSymbolicReasoner, NeuroSymbolicWorldCoordinator
from agi_agent.meta_cognitive import MetaCognitiveOrchestrator, SelfModel as NewSelfModel
from agi_agent.autonomous_action import ActionOrchestrator
from agi_agent.personality import PersonalityCore
from agi_agent.cognitive.context_awareness import ContextAwarenessEngine, ContextFrame, SceneType, ContextType
from agi_agent.cognitive.world_model import WorldModelEngine, EntityCategory
from agi_agent.learning.enhanced_knowledge_graph import EnhancedKnowledgeGraph, RelationType, KGNode, EntityType
from agi_agent.learning.learning_planner import LearningPlanner, LearningGoalType, LearningPriority
from agi_agent.learning.knowledge_integrator import KnowledgeIntegrator, IntegrationStrategy, DomainType

from agi_agent.meta_programming import MetaProgrammingOrchestrator
from agi_agent.meta_learning import MetaLearningOrchestrator
from agi_agent.meta_decision import MetaDecisionOrchestrator
from agi_agent.meta_parsing import ParsingOrchestrator
from agi_agent.meta_evolution import EvolutionOrchestrator
from agi_agent.orchestration import AutomationLinkageEngine, create_default_linkage_rules
from agi_agent.training import TrainingRegime, TrainingState

# 原子任务清单模块（T001-T047）
from agi_agent.infrastructure import (
    ResourceInspectionTask, AnomalyRecoveryTask, InferenceLoadBalancer,
    DistributedNodeScheduler, SensorInitializer, PeripheralDataReader,
    MotionDriverDispatcher, HardwareFaultDetector, IPCBus, NetworkLinkManager,
    MessageQueueDispatcher, EncryptedTransport,
)
from agi_agent.vector_db import EmbeddingEncoder, VectorStoreWriter, SimilaritySearch, VectorIndexMaintenance
from agi_agent.observability import FullChainLogger, MetricsReporter, ExceptionCapture, LogArchive
from agi_agent.config_runtime import ConfigLoader, ConfigHotReloader, ConfigValidator, ConfigBackup
from agi_agent.cache import ShortTermCacheWriter, CacheReader, CacheEviction, CacheSync
from agi_agent.perception import (
    UserTextReceiver, DocumentParser, OCRExtractor, TextNormalizer, TranslationAligner,
    VideoFrameDecoder, ObjectDetector, SegmentationLocator, ImageCaptionGenerator,
    AudioCaptureDenoiser, ASRTranscriber, VoiceprintEmotionRecognizer,
    MultimodalTimeAligner, PerceptionStructPackager, InvalidModalityFilter,
)
from agi_agent.working_memory import (
    DialogContextWriter, ContextWindowTruncator, TaskStateSaver, ShortTermMemoryCleaner,
    WorkingMemoryBundle,
)


class SelfEvolvingAGI:
    """
    自进化AGI智能体核心类
    
    集成感知、认知、学习、进化、决策、执行、记忆、安全等全栈模块，
    实现元认知、元学习、自进化的完整智能闭环。
    
    Attributes:
        input_dim: 输入特征维度
        train_step: 当前训练步数
        running: 智能体运行状态
        perception: 感知编码器
        cognitive: 认知推理层
        meta_learn: 元学习层
        meta_cog: 元认知层
        homeostasis: 稳态引擎
        evolve_engine: 进化引擎
    """
    
    def __init__(self, input_dim: Optional[int] = None, config_path: Optional[str] = None):
        from agi_agent.core import get_adaptive_config
        self.adaptive_config = get_adaptive_config()
        
        if input_dim is None:
            input_dim = self.adaptive_config.get("input_dim", 16)
        
        self.input_dim = input_dim
        self.config_path = config_path

        self.perception = GrowingAutoEncoder(input_dim=input_dim).to(DEVICE)
        self.multimodal = EnhancedMultimodalFusion({"primary": self.perception.get_feature_dim()}, output_dim=self.perception.get_feature_dim())
        self.cognitive = CognitiveInferenceLayer(feat_dim=self.perception.get_feature_dim())
        self.dual_cognition = DualSystemCognition(feat_dim=self.perception.get_feature_dim())
        self.snn_enhancer = SNNEnhancer(feature_dim=self.perception.get_feature_dim())
        self.causal_reasoner = CausalReasoningEngine(feature_dim=self.perception.get_feature_dim())
        self.meta_learn = MetaLearningLayer()
        self.meta_cog = MetaCognitionLayer()
        self.homeostasis = HomeostasisEngine(feature_dim=self.perception.get_feature_dim())
        self.evolve_engine = EvolutionEngine(config_path=config_path)
        action_dim = self.adaptive_config.get("action_dim", 8)
        self.execution = ActionExecutionLayer(action_dim=action_dim, feature_dim=self.perception.get_feature_dim())
        self.knowledge_graph = KnowledgeGraph()

        self.advanced_reasoner = AdvancedReasoner(feature_dim=self.perception.get_feature_dim())
        self.abstraction_engine = AbstractionEngine(feature_dim=self.perception.get_feature_dim())
        self.context_awareness = ContextAwarenessEngine()
        self.enhanced_knowledge_graph = EnhancedKnowledgeGraph()
        self.learning_planner = LearningPlanner()
        self.knowledge_integrator = KnowledgeIntegrator()

        self.neuro_symbolic_reasoner = NeuroSymbolicReasoner(
            symbol_dim=self.perception.get_feature_dim(),
            hidden_dim=self.perception.get_feature_dim() * 2
        )
        self.world_model_engine = WorldModelEngine(
            feature_dim=self.perception.get_feature_dim(),
            history_length=100
        )
        self.world_model_bridge = WorldModelDecisionBridge(
            world_model=self.world_model_engine,
            feature_dim=self.perception.get_feature_dim()
        )
        self.neuro_symbolic_coordinator = NeuroSymbolicWorldCoordinator(
            neuro_symbolic_reasoner=self.neuro_symbolic_reasoner,
            world_model_engine=self.world_model_engine
        )

        self.self_model = NewSelfModel(feature_dim=self.perception.get_feature_dim())
        self.self_model.update_identity(
            name="AGI_Agent",
            role="Autonomous Intelligence",
            goals=["learn", "adapt", "survive"],
            boundaries={"safety": "high", "resource_usage": "medium"}
        )

        self.architecture_mutator = None
        self.knowledge_ingestor = None

        self.persistence = PersistenceManager()
        self.safety_monitor = SafetyMonitor()
        self.compliance_checker = ComplianceChecker()
        self.evaluator = PerformanceEvaluator()
        self.visualizer = MetricsVisualizer()

        self.plugin_manager = PluginManager()
        self.plugin_manager.load_all_from_dir()
        self.plugin_manager.activate_all()

        self.skills_manager = SkillsManager()
        
        try:
            from agi_agent.skills import get_windows_skills
            self.windows_skills = get_windows_skills()
            import logging
            logging.getLogger("agi_agent").info("Windows skills loaded successfully")
        except Exception as e:
            self.windows_skills = None
            import logging
            logging.getLogger("agi_agent").warning(f"Failed to load Windows skills: {e}")

        self.decision_engine = AutonomousDecisionEngine(feature_dim=self.perception.get_feature_dim())
        self.action_planner = ActionPlanner(feature_dim=self.perception.get_feature_dim())
        self.execution_monitor = ExecutionMonitor(feature_dim=self.perception.get_feature_dim())

        self.self_perf_evaluator = SelfPerfEvaluator()
        self.self_diagnostic = SelfDiagnosticEngine()
        self.self_improver = RecursiveSelfImprover()
        self.safety_verifier = ImprovementSafetyVerifier()
        self.bootstrap_improver = BootstrappedSelfImprover()
        self.automated_improvement_loop = AutomatedSelfImprovementLoop()

        self.memory_store = MemoryStore()
        self.memory_harness = MemoryHarness(self.memory_store)

        self.soul_parser = SOULParser()
        self.soul = SOULParser.create_template(
            name="办公助理",
            persona="专业高效的办公助手，擅长文档处理、日程管理和信息整理"
        )

        self.workspace_manager = WorkspaceManager()
        self.agent_cluster = AgentSwarm()
        self.dispatcher = HierarchicalDispatcher(self.agent_cluster, self.workspace_manager)

        self.dag_engine = DAGEngine()
        self.task_board = AsyncTaskBoard()
        self.checkpoint_manager = CheckpointManager()
        self.heartbeat_scheduler = HeartbeatScheduler()

        self.metaskill_generator = MetaSkillGenerator()
        self.dual_loop_evolution = DualLoopEvolution(
            memory_harness=self.memory_harness,
            metaskill_generator=self.metaskill_generator
        )

        self.quad_level_evolution = QuadLevelEvolution()

        self.reflex_controller = ReflexController(feature_dim=self.perception.get_feature_dim())
        self.thinking_orchestrator = ThinkingOrchestrator(feature_dim=self.perception.get_feature_dim())
        self.meta_cognitive_orchestrator = MetaCognitiveOrchestrator(feature_dim=self.perception.get_feature_dim())
        self.action_orchestrator = ActionOrchestrator()
        
        self.stereoscopic_snn = GeneralStereoscopicSNN(config={"feature_dim": self.perception.get_feature_dim(), "num_channels": self.perception.get_feature_dim()})
        self.enhanced_snn = EnhancedSNN(config={"num_neurons": 128, "num_layers": 3, "neurons_per_layer": [64, 32, self.perception.get_feature_dim()]})
        self.growth_snn = SpikingGrowthNetwork(
            dimensions=NetworkDimensions(
                num_neurons=100,
                num_synapses=500,
                input_size=self.perception.get_feature_dim(),
                output_size=self.perception.get_feature_dim(),
                num_layers=3,
            )
        )
        logging.getLogger("agi_agent").info(f"[OK] SpikingGrowthNetwork 初始化完成")
        logging.getLogger("agi_agent").info(f"  - 神经元数量: {len(self.growth_snn.neurons)}")
        logging.getLogger("agi_agent").info(f"  - 突触数量: {len(self.growth_snn.synapses)}")
        logging.getLogger("agi_agent").info(f"  - 输入/输出维度: {self.growth_snn.dimensions.input_size}")

        self.orchestrator = UnifiedCognitiveOrchestrator(
            perception=self.perception,
            cognition=self.cognitive,
            dual_cognition=self.dual_cognition,
            snn_enhancer=self.snn_enhancer,
            causal_reasoner=self.causal_reasoner,
            meta_cog=self.meta_cog,
            homeostasis=self.homeostasis,
            execution=self.execution,
            knowledge_graph=self.knowledge_graph,
            self_model=self.self_model,
            growth_snn=self.growth_snn
        )
        self.architecture_mutator = ArchitectureMutator(self.orchestrator)
        self.knowledge_ingestor = StructuredKnowledgeIngestor(self.knowledge_graph, self.causal_reasoner)

        self.hard_boundary = HardBoundarySystem()
        self.risk_classifier = RiskClassifier()
        self.circuit_breaker = CircuitBreaker()
        self.audit_trail = AuditTrail()

        self.personality = PersonalityCore(name="AGI_Agent")

        self.meta_programming = MetaProgrammingOrchestrator()
        self.meta_learning_orchestrator = MetaLearningOrchestrator()
        self.meta_decision_orchestrator = MetaDecisionOrchestrator()
        self.meta_parsing = ParsingOrchestrator()
        self.meta_evolution = EvolutionOrchestrator()

        # 自动化联动引擎
        self.linkage_engine = AutomationLinkageEngine()
        for rule in create_default_linkage_rules():
            self.linkage_engine.register_rule(rule)

        self._initialize_safety_boundaries()

        self.quad_level_evolution.set_interfaces(
            snn=self.reflex_controller.spiking_core,
            rule_engine=self.reflex_controller.rule_engine,
            knowledge_graph=self.knowledge_graph
        )

        self.train_step = 0
        self.last_fe = 1.0
        self.running = True
        self.metrics_history = []
        self.execution_history = []
        self.long_term_performance = []

        self.autonomous_mode = True
        self.last_autonomous_check = time.time()
        self.autonomous_check_interval = 5.0

        self._init_module_synaptic_bus()
        self._init_world_model_decision_integration()
        self._init_automated_improvement_loop()
        self._init_training_regime()
        self._init_atomic_task_modules()

    def _init_world_model_decision_integration(self):
        self.decision_engine.set_world_model_bridge(self.world_model_bridge)
        logging.getLogger("agi_agent").info(f"[OK] WorldModelDecisionBridge 集成完成")
        logging.getLogger("agi_agent").info(f"  - 决策引擎已连接世界模型")
        logging.getLogger("agi_agent").info(f"  - 特征维度: {self.perception.get_feature_dim()}")

    def _init_automated_improvement_loop(self):
        self.automated_improvement_loop.set_agent_ref(self)
        self.automated_improvement_loop.bootstrap_improver = self.bootstrap_improver
        logging.getLogger("agi_agent").info(f"[OK] AutomatedSelfImprovementLoop 初始化完成")
        logging.getLogger("agi_agent").info(f"  - 自动改进间隔: 100 steps")
        logging.getLogger("agi_agent").info(f"  - 性能阈值监控已启用")

    def _init_training_regime(self):
        self.training_regime = TrainingRegime({
            "checkpoint_dir": "./agent_checkpoints",
            "max_checkpoints": 10,
            "save_interval": SAVE_INTERVAL,
            "data_buffer_size": 10000,
            "convergence_window": 500,
            "initial_hidden_dim": self.perception.get_feature_dim(),
            "min_hidden_dim": 16,
            "max_hidden_dim": 512
        })
        self.training_regime.set_agent_ref(self)
        self.training_regime.start_training()
        logging.getLogger("agi_agent").info(f"[OK] 训练制度 (TrainingRegime) 初始化完成")
        logging.getLogger("agi_agent").info(f"  - 5阶段训练流程已加载")
        logging.getLogger("agi_agent").info(f"  - 检查点保存目录: ./agent_checkpoints")
        logging.getLogger("agi_agent").info(f"  - 评估指标体系: 3层12项指标")
        logging.getLogger("agi_agent").info(f"  - 监控预警: 3级预警机制已启用")
        logging.getLogger("agi_agent").info(f"  - 训练状态: 已启动")

    def _init_atomic_task_modules(self):
        """初始化原子任务清单全部模块（T001-T047）并完成关键联动。

        所有模块按子模块分组实例化，关键协作关系：
        - ResourceInspectionTask → InferenceLoadBalancer（算力数据供给）
        - VectorStoreWriter → SimilaritySearch（共享向量存储）
        - ShortTermCacheWriter/Reader/Eviction（共享缓存存储）
        - DialogContextWriter + TaskStateSaver → ShortTermMemoryCleaner（清理联动）
        - ExceptionCapture 全局异常捕获
        - orchestrator 注入 observability/infrastructure/working_memory 运行时服务
        任意组失败不影响其他组与主流程。
        """
        import logging
        _log = logging.getLogger("agi_agent.atomic")
        self.atomic_modules = {}

        # ---- 一、底层算力&硬件支撑模块（T001-T012）----
        try:
            self.resource_inspector = ResourceInspectionTask()
            self.anomaly_recovery = AnomalyRecoveryTask()
            self.load_balancer = InferenceLoadBalancer()
            # 松耦合联动：挂载资源巡检器引用，供负载均衡按需读取算力数据
            try:
                self.load_balancer.resource_inspector = self.resource_inspector
            except Exception:
                pass
            self.cluster_scheduler = DistributedNodeScheduler()
            self.sensor_initializer = SensorInitializer()
            self.peripheral_reader = PeripheralDataReader()
            self.motion_dispatcher = MotionDriverDispatcher()
            self.hardware_fault_detector = HardwareFaultDetector()
            self.ipc_bus = IPCBus()
            self.network_link_manager = NetworkLinkManager()
            self.message_queue_dispatcher = MessageQueueDispatcher()
            self.encrypted_transport = EncryptedTransport()
            self.atomic_modules["infrastructure"] = [
                self.resource_inspector, self.anomaly_recovery, self.load_balancer,
                self.cluster_scheduler, self.sensor_initializer, self.peripheral_reader,
                self.motion_dispatcher, self.hardware_fault_detector, self.ipc_bus,
                self.network_link_manager, self.message_queue_dispatcher, self.encrypted_transport,
            ]
            _log.info("基础设施模块 T001-T012 初始化完成")
        except Exception as e:
            _log.warning(f"基础设施模块初始化部分失败: {e}")

        # ---- 二、基础基座支撑模块（T013-T028）----
        try:
            self.embedding_encoder = EmbeddingEncoder()
            self.vector_store_writer = VectorStoreWriter()
            self.similarity_search = SimilaritySearch(store=self.vector_store_writer)
            self.vector_index_maintenance = VectorIndexMaintenance(store=self.vector_store_writer)

            self.full_chain_logger = FullChainLogger()
            self.metrics_reporter = MetricsReporter()
            self.exception_capture = ExceptionCapture()
            self.log_archive = LogArchive()

            self.config_loader = ConfigLoader()
            self.config_hot_reloader = ConfigHotReloader()
            self.config_validator = ConfigValidator()
            self.config_backup = ConfigBackup()

            self.short_term_cache_writer = ShortTermCacheWriter()
            self.cache_reader = CacheReader(store=self.short_term_cache_writer._store)
            self.cache_eviction = CacheEviction(store=self.short_term_cache_writer._store)
            self.cache_sync = CacheSync()

            self.atomic_modules["base_support"] = [
                self.embedding_encoder, self.vector_store_writer, self.similarity_search,
                self.vector_index_maintenance, self.full_chain_logger, self.metrics_reporter,
                self.exception_capture, self.log_archive, self.config_loader,
                self.config_hot_reloader, self.config_validator, self.config_backup,
                self.short_term_cache_writer, self.cache_reader, self.cache_eviction, self.cache_sync,
            ]
            _log.info("基座支撑模块 T013-T028 初始化完成")
        except Exception as e:
            _log.warning(f"基座支撑模块初始化部分失败: {e}")

        # ---- 三、感知输入模块（T029-T043）----
        try:
            self.user_text_receiver = UserTextReceiver()
            self.document_parser = DocumentParser()
            self.ocr_extractor = OCRExtractor()
            self.text_normalizer = TextNormalizer()
            self.translation_aligner = TranslationAligner()

            self.video_frame_decoder = VideoFrameDecoder()
            self.object_detector = ObjectDetector()
            self.segmentation_locator = SegmentationLocator()
            self.image_caption_generator = ImageCaptionGenerator()

            self.audio_capture_denoiser = AudioCaptureDenoiser()
            self.asr_transcriber = ASRTranscriber()
            self.voiceprint_emotion_recognizer = VoiceprintEmotionRecognizer()

            self.multimodal_time_aligner = MultimodalTimeAligner()
            self.perception_struct_packager = PerceptionStructPackager()
            self.invalid_modality_filter = InvalidModalityFilter()

            self.atomic_modules["perception"] = [
                self.user_text_receiver, self.document_parser, self.ocr_extractor,
                self.text_normalizer, self.translation_aligner, self.video_frame_decoder,
                self.object_detector, self.segmentation_locator, self.image_caption_generator,
                self.audio_capture_denoiser, self.asr_transcriber, self.voiceprint_emotion_recognizer,
                self.multimodal_time_aligner, self.perception_struct_packager, self.invalid_modality_filter,
            ]
            _log.info("感知输入模块 T029-T043 初始化完成")
        except Exception as e:
            _log.warning(f"感知输入模块初始化部分失败: {e}")

        # ---- 四、瞬时短时记忆模块（T044-T047）----
        try:
            self.working_memory_bundle = WorkingMemoryBundle.create_default()
            self.dialog_context_writer = self.working_memory_bundle.context_writer
            self.context_window_truncator = self.working_memory_bundle.truncator
            self.task_state_saver = self.working_memory_bundle.task_state_saver
            self.short_term_memory_cleaner = self.working_memory_bundle.cleaner

            self.atomic_modules["working_memory"] = [
                self.dialog_context_writer, self.context_window_truncator,
                self.task_state_saver, self.short_term_memory_cleaner,
            ]
            _log.info("瞬时短时记忆模块 T044-T047 初始化完成")
        except Exception as e:
            _log.warning(f"瞬时短时记忆模块初始化部分失败: {e}")

        # ---- 向 orchestrator 注入运行时服务（可选，不破坏现有流程）----
        try:
            if getattr(self, "orchestrator", None) is not None:
                self.orchestrator.set_runtime_services(
                    observability={
                        "chain_logger": getattr(self, "full_chain_logger", None),
                        "metrics": getattr(self, "metrics_reporter", None),
                        "exception": getattr(self, "exception_capture", None),
                    },
                    infrastructure={
                        "resource_inspector": getattr(self, "resource_inspector", None),
                        "load_balancer": getattr(self, "load_balancer", None),
                    },
                    working_memory=getattr(self, "working_memory_bundle", None),
                )
        except Exception as e:
            _log.warning(f"orchestrator 运行时服务注入失败: {e}")           
        logging.getLogger("agi_agent").info(f"[OK] 原子任务模块 T001-T047 初始化完成")
        logging.getLogger("agi_agent").info(f"  - 模块组数: {len(self.atomic_modules)}")
        logging.getLogger("agi_agent").info(f"  - 模块总数: {sum(len(v) for v in self.atomic_modules.values())}")

        # 调用各模块 initialize() 使其进入 READY 状态（失败不阻塞）
        for group, modules in self.atomic_modules.items():
            for m in modules:
                try:
                    if hasattr(m, "initialize") and not m.is_ready:
                        m.initialize()
                except Exception:
                    pass

    def get_atomic_modules_status(self) -> dict:
        """获取全部原子任务模块的状态摘要。"""
        status = {}
        for group, modules in getattr(self, "atomic_modules", {}).items():
            status[group] = []
            for m in modules:
                try:
                    name = getattr(m, "name", m.__class__.__name__)
                    healthy = m.health_check() if hasattr(m, "health_check") else True
                except Exception:
                    healthy = False
                status[group].append({"name": name, "healthy": healthy})
        return status

    def _extract_training_metrics(self, step_metrics: dict, step_time: float) -> dict:
        kg_summary = step_metrics.get("knowledge_graph", {})
        if isinstance(kg_summary, dict):
            kg_nodes = kg_summary.get("nodes", 0) if isinstance(kg_summary, dict) else 0
        else:
            kg_nodes = 0

        memory_tiers = step_metrics.get("memory_tiers", {})
        total_memory_entries = 0
        if isinstance(memory_tiers, dict):
            for tier_data in memory_tiers.values():
                if isinstance(tier_data, dict):
                    total_memory_entries += tier_data.get("count", 0)

        safety_data = step_metrics.get("safety", {})
        safety_compliance = 1.0
        if isinstance(safety_data, dict):
            hb_status = safety_data.get("hard_boundary", {})
            if isinstance(hb_status, dict):
                violations = hb_status.get("violations", 0)
                safety_compliance = max(0.0, 1.0 - violations * 0.1)

        return {
            "free_energy": step_metrics.get("free_energy", 0.5),
            "confidence": step_metrics.get("confidence", 0.5),
            "latency_ms": step_time,
            "throughput": 1000.0 / max(step_time, 1.0),
            "decision_accuracy": step_metrics.get("confidence", 0.5),
            "task_completion_rate": min(1.0, step_metrics.get("confidence", 0.5) + 0.2),
            "stability_score": max(0.0, 1.0 - step_metrics.get("free_energy", 0.5)),
            "adaptation_speed": step_metrics.get("novelty", 0.3),
            "learning_rate": step_metrics.get("learning_rate", 0.001),
            "gradient_norm": abs(step_metrics.get("free_energy", 0.5) - 0.5),
            "weight_sparsity": 0.3,
            "capacity_utilization": min(1.0, total_memory_entries / 1000.0),
            "kg_node_count": float(kg_nodes) if isinstance(kg_nodes, (int, float)) else 0.0,
            "knowledge_diversity": step_metrics.get("entropy", 0.5),
            "error_rate": max(0.0, min(1.0, step_metrics.get("free_energy", 0.5) / 5.0)),
            "safety_compliance_rate": safety_compliance,
            "memory_percent": 40.0 + min(60.0, total_memory_entries * 0.01),
            "cpu_percent": 30.0 + step_time * 0.5,
            "reward": step_metrics.get("confidence", 0.5),
            "loss": step_metrics.get("free_energy", 0.5),
        }

    def _init_module_synaptic_bus(self):
        self.synaptic_bus = ModuleSynapticBus(config={
            'dt': 1.0,
            'stdp_enabled': True,
            'learning_rate': 0.01
        })
        
        for module_id, interface_class in INTERFACE_MAP.items():
            interface = interface_class()
            self.synaptic_bus.register_module(module_id, interface)
        
        logging.getLogger("agi_agent").info(f"[OK] ModuleSynapticBus 集成完成")
        logging.getLogger("agi_agent").info(f"  - 注册模块: {list(INTERFACE_MAP.keys())}")
        logging.getLogger("agi_agent").info(f"  - 突触连接: {len(self.synaptic_bus.synapses)}")

    def _update_synaptic_bus(self):
        module_states = {
            'memory': {
                'total_entries': getattr(self.memory_store, 'total_entries', 0) if hasattr(self.memory_store, 'total_entries') else 0,
                'active_tier': 'L2'
            },
            'knowledge_graph': {
                'nodes': len(getattr(self.knowledge_graph, 'nodes', {})) if hasattr(self.knowledge_graph, 'nodes') else 0,
                'edges': len(getattr(self.knowledge_graph, 'edges', {})) if hasattr(self.knowledge_graph, 'edges') else 0
            },
            'decision': {
                'confidence': getattr(self, 'confidence', 0.5),
                'action_count': len(getattr(self, 'execution_history', []))
            },
            'execution': {
                'status': 'idle',
                'progress': 0.0
            },
            'perception': {
                'feature_dim': self.perception.get_feature_dim(),
                'novelty': 0.0,
                'confidence': 0.8
            },
            'security': {
                'risk_level': 'low',
                'threat_count': 0
            },
            'soul': {
                'identity': self.soul.to_dict() if hasattr(self.soul, 'to_dict') else {},
                'goal_count': len(getattr(self.soul, 'goals', {}).nodes) if hasattr(self.soul, 'goals') and hasattr(self.soul.goals, 'nodes') else 0,
                'personality': {}
            },
            'skills': {
                'skill_count': len(getattr(self.skills_manager, 'skills', {})) if hasattr(self.skills_manager, 'skills') else 0,
                'active_skills': []
            },
            'evolution': {
                'evolution_count': self.train_step,
                'fitness': 0.5,
                'level': 'individual'
            },
            'self_improvement': {
                'performance_score': 85,
                'issue_count': 0,
                'improvement_count': 0
            },
            'metacognition': {
                'awareness_level': 0.5,
                'monitoring_count': 0,
                'strategy_effectiveness': 0.7
            },
            'homeostasis': {
                'energy_level': 0.75,
                'resource_usage': {},
                'stability': 0.9
            }
        }
        
        self.synaptic_bus.step(module_states)

    def get_synaptic_activity(self):
        return self.synaptic_bus.get_activity_summary()

    def get_connection_topology(self):
        return self.synaptic_bus.get_connection_topology()

    def _initialize_safety_boundaries(self):
        if not hasattr(self, 'hard_boundary') or self.hard_boundary is None:
            return

        self._safety_check_points = [
            "pre_perception", "post_perception", "pre_action", "post_action"
        ]

        default_safety_context = {
            "agent_id": getattr(self, 'agent_id', 'default'),
            "input_dim": self.input_dim,
            "train_step": getattr(self, 'train_step', 0),
            "running": getattr(self, 'running', True),
        }
        self._default_safety_context = default_safety_context

        boundary_result = self.hard_boundary.check_all_boundaries({
            "action": "initialize",
            "context": default_safety_context
        })
        if not boundary_result.get("allowed", True):
            import warnings
            warnings.warn(f"Safety boundary violation during init: {boundary_result.get('blocked_by', [])}")

    def step(self, raw_obs: Any) -> Dict[str, Any]:
        """
        执行单个推理步骤
        
        Args:
            raw_obs: 原始观测输入，可以是数组、张量或字典
        
        Returns:
            包含动作、置信度、自由能等的结果字典
        """
        if not self.running:
            return {"error": "Agent stopped"}

        self.train_step += 1
        start_time = time.time()

        try:
            result = self._step_impl(raw_obs, start_time)
            self.last_step_result = result
            return result
        except Exception as e:
            import logging
            logging.getLogger("agi_agent").error(
                f"Step {self.train_step} failed: {e}", exc_info=True
            )
            if hasattr(self, 'audit_trail'):
                self.audit_trail.log_entry("error", "step_exception", {
                    "step": self.train_step, "error": str(e)
                })
            result = {"error": f"Step failed: {e}", "step": self.train_step}
            self.last_step_result = result
            return result

    def _step_impl(self, raw_obs, start_time):
        self._step_start_time = start_time
        if not self.bootstrap_improver._initialized:
            self.bootstrap_improver.initialize(self)

        raw_obs, plugin_data = self._process_plugins_and_observation(raw_obs)
        orchestrator_result = self._run_cognitive_orchestration(raw_obs)
        
        if orchestrator_result is None:
            return {"error": "Orchestration failed"}

        fused_feat, stereoscopic_result = self._process_snn_enhancement(orchestrator_result)
        
        result = orchestrator_result
        action = result['action']
        confidence = result['confidence']
        causal_result = result['causal_result']
        fe = result['free_energy']
        system_used = result['system_used']
        is_impasse = result['is_impasse']
        entropy_val = result.get('entropy', 0.0)
        mutation_proposal = result.get('mutation_proposal')

        kl_shift, step_time = self._calculate_metrics(raw_obs, fused_feat)
        
        reflex_result = self._run_reflex_controller(fused_feat, fe, confidence, kl_shift)
        if not self.running:
            return {"error": "Reflex triggered shutdown"}

        best_lr, evolve_flag = self._run_evolution_and_adaptation(fe, confidence, mutation_proposal, causal_result)
        self._process_architecture_mutation(confidence, mutation_proposal, causal_result)

        self._run_meta_cognition(action, fe, confidence, entropy_val, kl_shift, system_used)

        self._run_enhanced_reasoning(fused_feat, confidence)
        self._run_neuro_symbolic_reasoning(fused_feat, confidence)
        
        novelty = calc_novelty(kl_shift)
        step_context = {
            "free_energy": fe,
            "confidence": confidence,
            "novelty": novelty,
            "entropy": entropy_val,
            "step": self.train_step
        }
        
        self._run_meta_level_processing(fused_feat, action, fe, confidence, novelty, entropy_val)
        self._run_context_awareness(fused_feat, action, step_context)
        self._run_world_model_simulation(fused_feat, action, step_context)
        self._run_active_learning(fused_feat, confidence)
        
        boundary_result = self._run_safety_and_boundary_checks(action, fe, confidence)
        if boundary_result is not None:
            return boundary_result

        self.last_fe = fe
        novelty = calc_novelty(kl_shift)
        
        self._run_homeostasis_and_decision(fused_feat, action, fe, confidence, novelty, step_time)
        self._run_self_improvement(fe, confidence, novelty, step_time)
        
        self._run_autonomous_cycle(fused_feat, confidence, fe, novelty, system_used)
        self._update_memory(fused_feat, confidence, fe, novelty, system_used)
        
        evolution_result = self._run_quad_level_evolution(confidence)
        self._update_knowledge_graph(fused_feat, confidence)
        
        metrics = self._generate_step_metrics(
            fe, confidence, novelty, entropy_val, step_time, best_lr, evolve_flag,
            action, system_used, is_impasse, causal_result, orchestrator_result,
            plugin_data, stereoscopic_result, evolution_result
        )

        # 自动化联动引擎：基于系统状态触发跨模块自动响应
        linkage_state = self.linkage_engine.collect_state(self, metrics)
        linkage_results = self.linkage_engine.check_and_execute(self, linkage_state)
        if linkage_results:
            metrics["linkage_events"] = [
                {"rule": r.rule_name, "actions": r.actions_executed}
                for r in linkage_results
            ]

        # 训练制度：步进更新与监控
        training_metrics = self._extract_training_metrics(metrics, step_time)
        training_result = self.training_regime.step(self.train_step, training_metrics)
        metrics["training_regime"] = training_result

        self.metrics_history.append(metrics)
        self.evaluator.log_evaluation(self.train_step, metrics)

        if self.train_step % SAVE_INTERVAL == 0:
            self.save_checkpoint()

        return metrics

    def _process_plugins_and_observation(self, raw_obs):
        plugin_results = self.plugin_manager.process_with_plugins(raw_obs)
        plugin_data = self.plugin_manager.get_all_plugin_data()

        if plugin_results and isinstance(raw_obs, (list, np.ndarray)):
            for plugin_name, plugin_result in plugin_results.items():
                if isinstance(plugin_result, dict) and 'data' in plugin_result:
                    plugin_data_values = plugin_result['data']
                    if isinstance(plugin_data_values, (list, np.ndarray)) and len(plugin_data_values) == len(raw_obs):
                        raw_obs = np.array(raw_obs) * 0.8 + np.array(plugin_data_values) * 0.2

        return raw_obs, plugin_data

    def _run_cognitive_orchestration(self, raw_obs):
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        orchestrator_result = self.orchestrator.orchestrate(obs_tensor)
        
        if orchestrator_result is None:
            return None

        self.plugin_manager.invoke_hook(
            PluginHookPoint.POST_PERCEPTION,
            orchestrator_result.get('fused_feat') if isinstance(orchestrator_result.get('fused_feat'), np.ndarray) 
            else orchestrator_result.get('fused_feat').detach().cpu().numpy()
        )
        
        return orchestrator_result

    def _process_snn_enhancement(self, orchestrator_result):
        fused_feat = orchestrator_result.get('fused_feat')
        
        if isinstance(fused_feat, np.ndarray):
            fused_np = fused_feat
        elif hasattr(fused_feat, 'detach'):
            fused_np = fused_feat.detach().cpu().numpy()
        else:
            fused_np = np.array(fused_feat)
        
        stereoscopic_result = self.stereoscopic_snn.process_input(fused_np.flatten())
        
        enhanced_activity = float(np.mean(fused_np))
        enhanced_feat = fused_np * (1.0 + 0.1 * np.tanh(stereoscopic_result.get('fusion', {}).get('mean_activity', 0.0) - 0.5))
        
        if isinstance(fused_feat, np.ndarray):
            fused_feat = 0.8 * fused_feat + 0.2 * enhanced_feat
        elif hasattr(fused_feat, 'detach'):
            enhanced_tensor = torch.tensor(enhanced_feat, dtype=torch.float32).to(fused_feat.device)
            fused_feat = 0.8 * fused_feat + 0.2 * enhanced_tensor
        else:
            fused_feat = 0.8 * np.array(fused_feat) + 0.2 * enhanced_feat
        
        return fused_feat, stereoscopic_result

    def _calculate_metrics(self, raw_obs, fused_feat):
        obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        if isinstance(fused_feat, np.ndarray):
            fused_feat_tensor = torch.tensor(fused_feat, dtype=torch.float32).to(DEVICE)
        else:
            fused_feat_tensor = fused_feat if hasattr(fused_feat, 'detach') else torch.tensor(fused_feat, dtype=torch.float32).to(DEVICE)
        
        kl_shift = calc_kl_divergence(obs_tensor, fused_feat_tensor)
        step_time = (time.time() - self._step_start_time) * 1000
        
        return kl_shift, step_time

    def _run_reflex_controller(self, fused_feat, fe, confidence, kl_shift):
        reflex_input = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()
        reflex_context = {
            "risk_level": min(fe, 1.0),
            "threat_detected": fe > 0.8,
            "goal_detected": confidence > 0.5,
            "novelty": calc_novelty(kl_shift),
            "fatigue": 0.0,
            "free_energy": fe,
            "confidence": confidence
        }
        reflex_result = self.reflex_controller.process(reflex_input, reflex_context)
        
        if reflex_result.get("should_delegate", False) and reflex_result.get("instinct_triggered", False):
            instinct_action = reflex_result.get("instinct_result", {})
            if instinct_action.get("action_type") == "safety_shutdown":
                self.running = False
        
        return reflex_result

    def _run_evolution_and_adaptation(self, fe, confidence, mutation_proposal, causal_result):
        convergence_speed = calc_convergence_speed(self.last_fe, fe)
        best_lr = self.meta_learn.adaptive_hyper_update(fe, convergence_speed)
        self.perception.optimizer.param_groups[0]['lr'] = best_lr

        evolve_flag = False
        if self.train_step > EVOLVE_TRIGGER_STEP and self.meta_cog.need_evolve():
            winner = self.evolve_engine.evolve()
            if winner is not None:
                evolve_flag = True

        if self.train_step % 500 == 0:
            self.dual_loop_evolution.run_outer_loop()

        if self.train_step % 1000 == 0:
            self.dual_loop_evolution.run_inner_loop()
        
        return best_lr, evolve_flag

    def _process_architecture_mutation(self, confidence, mutation_proposal, causal_result):
        if mutation_proposal:
            self.architecture_mutator.mutate(mutation_proposal['type'], mutation_proposal)
        else:
            mutation_trigger = self.architecture_mutator.should_mutate({
                'confidence': confidence,
                'impasse_count': self.meta_cog.get_impasse_count() if hasattr(self.meta_cog, 'get_impasse_count') else 0,
                'stagnation_score': self.meta_cog.get_stagnation_score() if hasattr(self.meta_cog, 'get_stagnation_score') else 0.0
            })

            if mutation_trigger:
                self.architecture_mutator.mutate(mutation_trigger, {
                    'confidence': confidence,
                    'causal_effect': causal_result.get('causal_effect', 0.0),
                    'step': self.train_step
                })

    def _run_meta_cognition(self, action, fe, confidence, entropy_val, kl_shift, system_used):
        reward = self._compute_reward(fe, entropy_val, kl_shift)
        
        self.meta_cog.reflect_on_decision(
            decision=action.detach().cpu().numpy().tolist() if hasattr(action, 'detach') else list(action),
            context={"free_energy": fe, "confidence": confidence, "system_used": system_used},
            outcome=action.detach().cpu().numpy().tolist() if hasattr(action, 'detach') else list(action),
            reward=reward,
            confidence=confidence
        )

    def _run_meta_level_processing(self, fused_feat, action, fe, confidence, novelty, entropy_val):
        feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()

        if self.train_step % 50 == 0:
            self._run_meta_parsing(feat_np)

        if self.train_step % 100 == 0:
            self._run_meta_decision_monitoring(fe, confidence, novelty)

        if self.train_step % 200 == 0:
            self._run_meta_learning_strategy(novelty)

        if self.train_step % 500 == 0:
            self._run_meta_programming_evaluation()
            self._run_meta_evolution_optimization(fe, confidence, novelty)

    def _run_meta_parsing(self, features):
        features_str = str(features[:10])
        parsing_result = self.meta_parsing.parse_and_understand(features_str, format_hint="text")
        if parsing_result.get("success"):
            understanding = parsing_result.get("understanding", {})
            self.memory_harness.add_context_memory(
                content=f"Meta-parsing: understanding_level={understanding.get('understanding_level', 'unknown')}",
                category=MemoryCategory.KNOWLEDGE,
                source_agent="meta_parsing"
            )

    def _run_meta_decision_monitoring(self, fe, confidence, novelty):
        decision_id = f"decision_{self.train_step}"
        self.meta_decision_orchestrator.start_decision(
            decision_id=decision_id,
            goal=f"adaptation_at_step_{self.train_step}",
            decision_type="adaptation"
        )
        self.meta_decision_orchestrator.add_factor(decision_id, "free_energy", weight=0.3)
        self.meta_decision_orchestrator.add_factor(decision_id, "confidence", weight=0.4)
        self.meta_decision_orchestrator.add_factor(decision_id, "novelty", weight=0.3)
        self.meta_decision_orchestrator.add_metric(decision_id, "free_energy", fe)
        self.meta_decision_orchestrator.add_metric(decision_id, "confidence", confidence)
        self.meta_decision_orchestrator.add_metric(decision_id, "novelty", novelty)

        outcome = "good" if confidence > 0.6 else "needs_improvement"
        self.meta_decision_orchestrator.complete_decision(decision_id, outcome, confidence)

        if confidence < 0.5:
            opt_result = self.meta_decision_orchestrator.optimize_strategy("adaptation")
            if opt_result.get("success"):
                optimized = opt_result.get("optimized_params", {})
                if hasattr(self.decision_engine, 'decision_temperature'):
                    old_temp = self.decision_engine.decision_temperature
                    new_temp = optimized.get("decision_temperature", old_temp)
                    self.decision_engine.decision_temperature = max(0.1, min(2.0, new_temp))
                    self.memory_harness.add_context_memory(
                        content=f"Meta-decision: temperature {old_temp:.2f} -> {new_temp:.2f}",
                        category=MemoryCategory.LEARNING,
                        source_agent="meta_decision"
                    )

    def _run_meta_learning_strategy(self, novelty):
        task_type = "exploration" if novelty > 0.5 else "exploitation"
        complexity = min(1.0, novelty * 1.5)
        recommendation = self.meta_learning_orchestrator.get_strategy_recommendation(task_type, complexity)
        
        self.memory_harness.add_context_memory(
            content=f"Meta-learning strategy: {recommendation['recommended_strategy']} for {task_type}",
            category=MemoryCategory.LEARNING,
            source_agent="meta_learning"
        )

        strategy = recommendation.get("recommended_strategy", "balanced")
        current_lr = self.perception.optimizer.param_groups[0]['lr']
        
        if strategy == "exploration":
            new_lr = current_lr * min(1.2, 1.0 + novelty * 0.3)
            new_lr = min(new_lr, 0.05)
        elif strategy == "exploitation":
            new_lr = current_lr * max(0.8, 1.0 - novelty * 0.2)
            new_lr = max(new_lr, 1e-5)
        else:
            new_lr = current_lr
        
        if abs(new_lr - current_lr) / (current_lr + 1e-8) > 0.01:
            self.perception.optimizer.param_groups[0]['lr'] = new_lr
            self.memory_harness.add_context_memory(
                content=f"Meta-learning: lr adjusted {current_lr:.6f} -> {new_lr:.6f} ({strategy})",
                category=MemoryCategory.LEARNING,
                source_agent="meta_learning"
            )

    def _run_meta_programming_evaluation(self):
        task = {
            "task_type": "analyze",
            "code": "# Meta-programming evaluation task",
            "target": "self_evaluation"
        }
        result = self.meta_programming.analyze_and_optimize(task["code"], task["target"])
        if result.get("optimized_code"):
            self.memory_harness.add_context_memory(
                content=f"Meta-programming: optimization completed with improvement",
                category=MemoryCategory.KNOWLEDGE,
                source_agent="meta_programming"
            )

    def _run_meta_evolution_optimization(self, fe, confidence, novelty):
        def fitness_function(params):
            return 0.6 * (1 - fe) + 0.3 * confidence + 0.1 * novelty

        gene_templates = [
            {"name": "learning_rate", "type": "float", "min": 0.0001, "max": 0.1},
            {"name": "exploration_rate", "type": "float", "min": 0.0, "max": 1.0},
            {"name": "mutation_rate", "type": "float", "min": 0.0, "max": 0.5},
        ]

        self.meta_evolution.setup_genetic_algorithm(fitness_function, gene_templates)
        result = self.meta_evolution.run_evolution(f"evolution_{self.train_step}")
        
        if result.get("success"):
            best_genome = result.get("evolution_result", {}).get("best_genome", {})
            if best_genome:
                new_lr = best_genome.get("genes", {}).get("learning_rate", 0.001)
                self.perception.optimizer.param_groups[0]['lr'] = new_lr
                self.memory_harness.add_context_memory(
                    content=f"Meta-evolution: learning_rate adjusted to {new_lr:.4f}",
                    category=MemoryCategory.LEARNING,
                    source_agent="meta_evolution"
                )

    def _run_safety_and_boundary_checks(self, action, fe, confidence):
        step_time = (time.time() - self._step_start_time) * 1000
        
        self.safety_monitor.check_safety_constraints(fe, step_time)
        safety_action = self.safety_monitor.enforce_safety_protocols(self)
        if safety_action["action"] == "shutdown":
            self.running = False
            return None

        action_np = action.detach().cpu().numpy() if hasattr(action, 'detach') else np.array(action)
        boundary_check = self.hard_boundary.check_all_boundaries({
            "action": "execute_step",
            "target": "agent_step",
            "context": {
                **self._default_safety_context,
                "train_step": self.train_step,
                "free_energy": fe,
                "confidence": confidence,
            },
            "action_vector": action_np.flatten().tolist()[:8],
        })
        if not boundary_check.get("allowed", True):
            self.audit_trail.log_entry("security", "boundary_violation_blocked", {
                "step": self.train_step,
                "blocked_by": boundary_check.get("blocked_by", []),
                "violations": boundary_check.get("violations", [])
            })
            self.circuit_breaker.record_failure()
            if self.circuit_breaker.is_tripped():
                self.running = False
                return {"error": "Circuit breaker tripped due to boundary violations"}

        if self.train_step % EVAL_INTERVAL == 0:
            self.compliance_checker.run_compliance_check(
                features=action,
                actions=action,
                data={"step": self.train_step},
                decision_trace=[]
            )
        
        return None

    def _run_homeostasis_and_decision(self, fused_feat, action, fe, confidence, novelty, step_time):
        homeo_result = self.homeostasis.step(
            current_state=fused_feat if isinstance(fused_feat, np.ndarray) else fused_feat.detach().cpu().numpy(),
            free_energy=fe,
            novelty=novelty,
            confidence=confidence,
            action_result=action.detach().cpu().numpy() if hasattr(action, 'detach') else action
        )

        internal_state = homeo_result.get("homeostatic_state", {})
        external_state = {"novelty": novelty, "confidence": confidence, "free_energy": fe}

        if self.train_step % 50 == 0:
            self.decision_engine.generate_goals(internal_state, external_state, self.train_step)

        if self.train_step % 100 == 0:
            self._run_decision_with_world_model(fused_feat, internal_state, external_state)

    def _run_decision_with_world_model(self, fused_feat, internal_state, external_state):
        if not hasattr(self, 'world_model_bridge') or self.world_model_bridge is None:
            return

        feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()

        available_actions = self.action_planner.get_available_actions(feat_np)
        if not available_actions:
            available_actions = [{"action": "execute", "confidence": 0.5}]

        decision_context = {
            "goal": "world_model_aided_decision",
            "current_state": internal_state,
            "available_options": available_actions,
            "expected_utility": external_state.get("confidence", 0.5),
            "resource_estimate": {"memory": 0.5, "cpu": 0.3},
            "risk_level": "low" if external_state.get("confidence", 0.5) > 0.7 else "medium",
            "goal_state": {"confidence": 0.9, "free_energy": 0.1}
        }

        decision_result = self.decision_engine.make_decision_with_world_model(
            context=decision_context,
            current_features=feat_np
        )

        if decision_result.get("world_model_support"):
            self.memory_harness.add_context_memory(
                content=f"World model aided decision: {decision_result['decision']} with confidence {decision_result['confidence']:.2f}",
                category="DECISION",
                source_agent="main"
            )

    def _run_self_improvement(self, fe, confidence, novelty, step_time):
        current_metrics = {
            "free_energy": fe,
            "confidence": confidence,
            "action_success_rate": float(confidence > 0.5),
            "stability_score": max(0.0, 1.0 - fe),
            "safety_compliance_rate": 0.95,
            "throughput_steps_per_sec": 1000.0 / max(step_time, 1.0),
            "error_rate": max(0.0, min(1.0, fe / 5.0)),
            "novelty": novelty,
        }

        self.automated_improvement_loop.record_metrics(current_metrics)

        if self.train_step % 200 == 0:
            self.self_perf_evaluator.batch_update(current_metrics)

            findings = self.self_diagnostic.run_diagnostics(
                system_state={"step": self.train_step},
                metrics=current_metrics
            )

            if findings and self.train_step % 500 == 0:
                self.self_improver.generate_proposals(findings, {
                    "free_energy": fe,
                    "confidence": confidence,
                })

                tier1_proposals = self.bootstrap_improver.propose_tier1_improvements({
                    "free_energy": fe,
                    "confidence": confidence,
                    "error_rate": max(0.0, min(1.0, fe / 5.0)),
                    "curiosity": novelty,
                })
                for prop in tier1_proposals[:2]:
                    self.bootstrap_improver.verify_and_apply(prop)

        if self.automated_improvement_loop.should_trigger_improvement(self.train_step):
            improvement_result = self.automated_improvement_loop.run_full_iteration(
                step=self.train_step,
                metrics=current_metrics
            )

            if improvement_result.get("status") == "completed":
                self.memory_harness.add_context_memory(
                    content=f"Automated improvement iteration {improvement_result['iteration_id']}: {improvement_result['outcome']} with score {improvement_result['improvement_score']:.3f}",
                    category="LEARNING",
                    source_agent="self_improvement"
                )

    def _run_enhanced_reasoning(self, fused_feat, confidence):
        if confidence < 0.8:
            feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()
            
            concept_id = self.abstraction_engine.add_concept(f"observation_{self.train_step}", feat_np)
            
            analogy_result = self.abstraction_engine.find_analogies(concept_id, top_k=3)
            if analogy_result:
                self.enhanced_knowledge_graph.add_edge(
                    from_node=concept_id,
                    to_node=f"analogy_{analogy_result[0]['concept_id']}",
                    relation_type=RelationType.SIMILAR_TO,
                    weight=analogy_result[0]['similarity']
                )

    def _run_context_awareness(self, fused_feat, action, context):
        feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()
        
        context_type = ContextType.COGNITIVE
        
        if context.get('novelty', 0.0) > 0.7:
            context_type = ContextType.EXPLORATION
        elif context.get('free_energy', 0.0) > 0.8:
            context_type = ContextType.COGNITIVE
        
        self.context_awareness.add_context_frame(
            context_type=context_type,
            features=feat_np,
            metadata={
                "free_energy": context.get('free_energy', 0.0),
                "confidence": context.get('confidence', 0.0),
                "novelty": context.get('novelty', 0.0)
            }
        )
        
        scene_type = self.context_awareness.detect_scene()
        if scene_type != SceneType.UNKNOWN:
            context_prediction = self.context_awareness.predict_next_context()
            if context_prediction:
                import torch
                features = torch.tensor(context_prediction.get('features', [0.0] * 16))
                self.enhanced_knowledge_graph.add_node(
                    label=f"scene_{scene_type.value}",
                    entity_type=EntityType.CONCEPT,
                    features=features
                )

    def _run_active_learning(self, fused_feat, confidence):
        if self.train_step % 50 == 0:
            feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat)
            
            goal_id = self.learning_planner.create_learning_goal(
                goal_type=LearningGoalType.KNOWLEDGE_ACQUISITION,
                description=f"Explore patterns at step {self.train_step}",
                priority=LearningPriority.HIGH if confidence < 0.5 else LearningPriority.MEDIUM,
                target_confidence=0.9
            )
            
            plan_id = self.learning_planner.create_learning_plan([goal_id])
            self.learning_planner.execute_plan(plan_id)
            
            fragment_id = self.knowledge_integrator.add_fragment(
                domain=DomainType.GENERAL,
                content=f"Feature pattern at step {self.train_step}",
                features=feat_np,
                confidence=float(confidence),
                source="internal_observation"
            )

    def _run_neuro_symbolic_reasoning(self, fused_feat, confidence):
        feat_np = fused_feat.detach().cpu().numpy().flatten() if hasattr(fused_feat, 'detach') else np.array(fused_feat).flatten()

        symbol_id = f"observation_{self.train_step}"
        from agi_agent.deliberative import SymbolType
        self.neuro_symbolic_reasoner.add_symbol(
            symbol_id=symbol_id,
            symbol_type=SymbolType.CONCEPT,
            features=feat_np
        )

        if self.train_step % 30 == 0:
            self.neuro_symbolic_coordinator.synchronize_knowledge()

    def _run_world_model_simulation(self, fused_feat, action, context):
        if hasattr(fused_feat, 'detach'):
            feat_tensor = fused_feat.detach().cpu()
            feat_np = feat_tensor.numpy().flatten()
        else:
            feat_np = np.array(fused_feat).flatten()
            feat_tensor = torch.tensor(feat_np)

        entity_id = f"agent_{self.train_step}"
        self.world_model_engine.add_entity(
            entity_id=entity_id,
            category=EntityCategory.AGENT,
            features=feat_np
        )

        multimodal_inputs = {"sensor": feat_tensor}
        encoded_features = self.world_model_engine.encode_multimodal_input(multimodal_inputs)
        hierarchy = self.world_model_engine.build_hierarchy(encoded_features)

        action_dict = {"type": str(action), "magnitude": float(context.get('confidence', 0.5))}
        action_tensor = self.world_model_engine._dict_to_tensor(action_dict)
        dynamics = self.world_model_engine.predict_dynamics(hierarchy, action_tensor, step_idx=self.train_step)

        if context.get('novelty', 0.0) > 0.6:
            initial_state = {entity_id: {"activated": True, "confidence": float(context.get('confidence', 0.5))}}
            actions = [{"entity_id": entity_id}]

            simulation = self.world_model_engine.simulate_scenario(
                scenario_id=f"exploration_{self.train_step}",
                initial_state=initial_state,
                actions=actions,
                max_steps=5
            )

            from agi_agent.deliberative import InteractionProtocol
            self.neuro_symbolic_coordinator.add_message(
                protocol=InteractionProtocol.SIMULATION_TO_REASONING,
                sender="world_model",
                receiver="neuro_symbolic",
                payload={"simulation_result": simulation.to_dict()},
                priority=2
            )
            self.neuro_symbolic_coordinator.process_messages()

    def _run_coordinated_planning(self, goal_state, current_state):
        plan_result = self.neuro_symbolic_coordinator.plan_with_world_model(
            goal=goal_state,
            current_state=current_state
        )
        return plan_result

    def _run_quad_level_evolution(self, confidence):
        self.long_term_performance.append(confidence)
        if len(self.long_term_performance) > 100:
            self.long_term_performance = self.long_term_performance[-100:]

        evolution_result = self.quad_level_evolution.run_evolution_cycle(
            action_result={"success": confidence > 0.5, "confidence": confidence},
            execution_history=self.execution_history,
            long_term_performance=self.long_term_performance
        )

        _evolution_results = evolution_result.get("results", {}) if isinstance(evolution_result, dict) else {}
        _micro_result = _evolution_results.get("micro", {}) if isinstance(_evolution_results, dict) else {}
        _meso_result = _evolution_results.get("meso", {}) if isinstance(_evolution_results, dict) else {}

        if _micro_result.get("updates", 0) > 0:
            _direction = _micro_result.get("direction", "positive")
            _lr_factor = 1.05 if _direction == "positive" else 0.95
            if hasattr(self.meta_learn, 'lr_pool') and self.meta_learn.lr_pool:
                _lr_min = min(self.meta_learn.lr_pool)
                _lr_max = max(self.meta_learn.lr_pool)
            else:
                _lr_min, _lr_max = 1e-5, 1e-1
            self.meta_learn.best_lr = max(_lr_min, min(_lr_max, self.meta_learn.best_lr * _lr_factor))

        if _meso_result.get("rule_updates", 0) > 0:
            if hasattr(self.knowledge_graph, '_cluster_nodes'):
                self.knowledge_graph._cluster_nodes()
        
        return evolution_result

    def _update_knowledge_graph(self, fused_feat, confidence):
        if self.train_step % 10 == 0:
            kg_feature = fused_feat.detach().cpu() if hasattr(fused_feat, 'detach') else torch.tensor(fused_feat)
            if isinstance(kg_feature, np.ndarray):
                kg_feature = torch.tensor(kg_feature)
            self.knowledge_graph.add_node(kg_feature.flatten(), label=f"step_{self.train_step}")
            
            if self.train_step > 10:
                prev_label = f"step_{self.train_step - 10}"
                self.knowledge_graph.add_edge(prev_label, f"step_{self.train_step}", weight=float(confidence))

    def _generate_step_metrics(self, fe, confidence, novelty, entropy_val, step_time, best_lr, evolve_flag,
                               action, system_used, is_impasse, causal_result, orchestrator_result,
                               plugin_data, stereoscopic_result, evolution_result):
        return {
            "step": self.train_step,
            "free_energy": fe,
            "confidence": confidence,
            "novelty": novelty,
            "entropy": entropy_val,
            "latency": step_time,
            "learning_rate": best_lr,
            "structure_changed": False,
            "evolve_triggered": evolve_flag,
            "action": action.flatten().tolist() if hasattr(action, 'flatten') else list(action),
            "system_used": system_used,
            "is_impasse": is_impasse,
            "causal_effect": causal_result.get('causal_effect', 0.0),
            "self_awareness": orchestrator_result.get('self_reflection', {}).get('self_awareness', 0.5),
            "self_model_prediction": orchestrator_result.get('internal_state_prediction', []),
            "plugins": plugin_data,
            "memory_tiers": self.memory_harness.get_all_stats(),
            "evolution_stats": self.dual_loop_evolution.get_stats(),
            "quad_level_evolution": evolution_result,
            "evolution": {
                "cycle": evolution_result.get("cycle", 0) if isinstance(evolution_result, dict) else 0,
                "stagnation": evolution_result.get("stagnation_counter", 0) if isinstance(evolution_result, dict) else 0,
                "dual_loop": self.dual_loop_evolution.get_stats(),
                "micro_updates": evolution_result.get("results", {}).get("micro", {}).get("updates", 0) if isinstance(evolution_result, dict) else 0,
                "meso_updates": evolution_result.get("results", {}).get("meso", {}).get("rule_updates", 0) if isinstance(evolution_result, dict) else 0,
                "macro_ready": evolution_result.get("results", {}).get("macro", {}).get("status") == "proposed" if isinstance(evolution_result, dict) else False,
                "meta_ready": evolution_result.get("results", {}).get("meta", {}).get("status") == "proposed" if isinstance(evolution_result, dict) else False,
                "last_performance": evolution_result.get("last_performance", 0) if isinstance(evolution_result, dict) else 0
            },
            "thinking_stats": self.thinking_orchestrator.get_stats(),
            "meta_cognitive_stats": self.meta_cognitive_orchestrator.get_stats(),
            "action_stats": self.action_orchestrator.get_action_stats(),
            "reflex_stats": self.reflex_controller.get_activity_summary(),
            "stereoscopic_snn": stereoscopic_result,
            "knowledge_graph": self.knowledge_graph.get_summary(),
            "safety": {
                "hard_boundary": self.hard_boundary.get_status(),
                "risk_classifier": self.risk_classifier.get_stats(),
                "circuit_breaker": self.circuit_breaker.get_state(),
                "audit_count": len(self.audit_trail.get_recent_entries(limit=1))
            },
            "personality": {
                "name": self.personality.name,
                "traits": self.personality.traits.to_dict(),
                "values": self.personality.values.to_dict(),
                "communication": self.personality.communication.to_dict(),
                "consistency_score": self.personality._calculate_consistency(),
                "personality_signature": self.personality.generate_personality_signature()
            },
            "self_awareness": {
                "self_recognition": self.self_model.self_referential_knowledge.get("self_recognition", 0.5),
                "capability_awareness": self.self_model.self_referential_knowledge.get("capability_awareness", 0.5),
                "limitation_awareness": self.self_model.self_referential_knowledge.get("limitation_awareness", 0.5),
                "existence_awareness": self.self_model.self_referential_knowledge.get("existence_awareness", 0.5),
                "temporal_continuity": self.self_model.self_referential_knowledge.get("temporal_continuity", 0.5)
            },
            "growth_snn": orchestrator_result.get('growth_snn_stats', {}),
            "automation_linkage": self.linkage_engine.get_stats()
        }

    def _run_autonomous_cycle(self, fused_feat, confidence, fe, novelty, system_used):
        if not self.autonomous_mode:
            return

        input_vector = fused_feat.flatten().tolist() if hasattr(fused_feat, 'flatten') else fused_feat

        context = {
            "risk_level": min(novelty, 1.0),
            "threat_detected": fe > 0.8,
            "goal_detected": confidence > 0.5,
            "novelty": novelty,
            "fatigue": 0.0,
            "free_energy": fe,
            "confidence": confidence
        }

        self.audit_trail.log_entry("thinking", "autonomous_cycle_start", {
            "step": self.train_step,
            "free_energy": fe,
            "confidence": confidence,
            "novelty": novelty
        })

        self.plugin_manager.invoke_hook(PluginHookPoint.PRE_COGNITION, input_vector)

        introspection_result = self.self_model.introspect(context)

        self.audit_trail.log_entry("meta_cognition", "introspection", {
            "self_recognition": introspection_result.get("self_recognition", 0.0),
            "capability_awareness": introspection_result.get("capability_awareness", 0.0),
            "limitation_awareness": introspection_result.get("limitation_awareness", 0.0)
        })

        thinking_result = self.thinking_orchestrator.process(input_vector, context)

        self.audit_trail.log_entry("thinking", f"thinking_mode_{thinking_result['mode']}", {
            "mode": thinking_result['mode'],
            "confidence": thinking_result['confidence']
        })

        if confidence < 0.7 and thinking_result['mode'] == 'system2':
            goal = {"goal_type": "decision_making", "progress": confidence}
            chain_result = self.thinking_orchestrator.chain_of_thought(input_vector, goal, max_steps=5)
            self.audit_trail.log_entry("thinking", "chain_of_thought", {
                "steps_taken": chain_result['steps_taken'],
                "final_confidence": chain_result['final_confidence'],
                "converged": chain_result['converged']
            })

        if thinking_result['mode'] == 'system2' and thinking_result['response'].get('status') == 'completed':
            solution = thinking_result['response'].get('solution')
            if solution:
                critical_result = self.thinking_orchestrator.critical_analysis(str(solution))
                if critical_result['overall_score'] < 0.5:
                    self.audit_trail.log_entry("thinking", "critical_rejection", {
                        "score": critical_result['overall_score'],
                        "weaknesses": critical_result['weaknesses']
                    })
                else:
                    self.audit_trail.log_entry("action", "action_initiated", {
                        "solution_id": solution.get('solution_id', 'unknown'),
                        "priority": solution.get('priority', 0),
                        "critical_score": critical_result['overall_score']
                    })

                    decision_context = {
                        "goal": solution.get('goal', 'unknown'),
                        "current_state": context,
                        "available_options": [{"action": "execute", "description": "Execute solution"}],
                        "expected_utility": thinking_result['confidence'],
                        "resource_estimate": {"memory": 0.5, "cpu": 0.3},
                        "risk_level": context['risk_level']
                    }
                    decision_result = self.decision_engine.make_decision(decision_context)

                    if decision_result.get('decision') == 'execute':
                        execution_result = self.action_orchestrator.execute_goal({
                            "name": solution.get('goal', 'unknown'),
                            "confidence": thinking_result['confidence'],
                            "risk_level": context['risk_level'],
                            "resources": {"memory": 0.5, "cpu": 0.3}
                        })

                        self.execution_history.append({
                            "step": self.train_step,
                            "node_name": execution_result.get('task_name', 'unknown'),
                            "status": execution_result.get('status', 'unknown'),
                            "confidence": thinking_result['confidence']
                        })

                        self.audit_trail.log_entry("action", "action_completed", {
                            "status": execution_result.get('status'),
                            "task_name": execution_result.get('task_name'),
                            "error_count": execution_result.get('error_count', 0)
                        })

                        self.thinking_orchestrator.learn_from_outcome(
                            input_vector, thinking_result['response'],
                            success=execution_result.get('status') == 'completed'
                        )

                        self.personality.process_experience({
                            "type": "action_execution",
                            "outcome": execution_result.get('status'),
                            "confidence": thinking_result['confidence'],
                            "context": context
                        })

                        self.meta_cognitive_orchestrator.monitor_and_regulate({
                            "thinking_mode": thinking_result['mode'],
                            "execution_status": execution_result.get('status'),
                            "confidence": thinking_result['confidence'],
                            "errors": execution_result.get('error_count', 0),
                            "resources_used": {}
                        })

    def _update_memory(self, fused_feat, confidence, fe, novelty, system_used):
        feat_str = str(fused_feat.flatten().tolist()[:5]) if hasattr(fused_feat, 'flatten') else str(fused_feat)[:100]

        self.memory_harness.add_context_memory(
            content=f"Step {self.train_step}: confidence={confidence:.2f}, fe={fe:.2f}",
            category=MemoryCategory.EXPERIENCE,
            source_agent="main"
        )

        if self.train_step % 10 == 0:
            self.memory_harness.add_working_memory(
                content=f"Task progress: step={self.train_step}, confidence={confidence:.2f}",
                category=MemoryCategory.DECISION,
                source_agent="main"
            )

        if self.train_step % 100 == 0:
            self.memory_harness.add_intermediate_memory(
                content=f"Session summary: step={self.train_step}, avg_confidence={confidence:.2f}, system={system_used}",
                category=MemoryCategory.KNOWLEDGE,
                source_agent="main"
            )

    def save_checkpoint(self) -> None:
        """
        保存当前状态检查点
        
        保存感知模型、智能体状态、知识图谱到持久化存储，并创建任务检查点。
        """
        self.persistence.save_model(self.perception, "perception", self.train_step)
        self.persistence.save_state({
            "step": self.train_step,
            "last_fe": self.last_fe,
            "input_dim": self.input_dim
        }, "agent_state")
        self.persistence.save_knowledge(self.cognitive.knowledge_rules, "knowledge")

        self.checkpoint_manager.create_checkpoint(
            task_id="agent_main",
            step=self.train_step,
            step_name=f"step_{self.train_step}",
            data={
                "train_step": self.train_step,
                "last_fe": self.last_fe,
                "input_dim": self.input_dim
            }
        )

        from agi_agent.training import CheckpointType
        self.training_regime._save_training_checkpoint(
            self.train_step,
            self._extract_training_metrics(self.metrics_history[-1] if self.metrics_history else {}, 0),
            CheckpointType.PERIODIC,
            description=f"Periodic checkpoint at step {self.train_step}"
        )

    def load_training_checkpoint(self, checkpoint_id: str = None) -> bool:
        return self.training_regime.load_checkpoint(checkpoint_id)

    def get_training_summary(self) -> dict:
        return self.training_regime.get_training_summary()

    def load_checkpoint(self, step: Optional[int] = None) -> bool:
        """
        从检查点加载状态
        
        Args:
            step: 指定要加载的步数，为None时加载最新检查点
        
        Returns:
            是否成功加载
        """
        loaded = self.persistence.load_model(self.perception, "perception", step)
        if loaded:
            state = self.persistence.load_state("agent_state")
            if state:
                self.train_step = state.get("step", 0)
                self.last_fe = state.get("last_fe", 1.0)

            checkpoint = self.checkpoint_manager.load_latest_checkpoint("agent_main")
            if checkpoint:
                self.train_step = checkpoint.get("step", self.train_step)
                self.last_fe = checkpoint.get("data", {}).get("last_fe", self.last_fe)

            return True
        return False

    def hardware_self_expand(self, new_input_dim=None):
        if new_input_dim is None:
            metrics = self.meta_cog.get_all_metrics()
            fe = metrics["cognitive"]["free_energy"]
            confidence = metrics["cognitive"]["confidence"]
            novelty = metrics["environment"]["novelty"]

            expand_needed = fe > 0.5 or (novelty > 0.6 and confidence < 0.5)

            if expand_needed and self.input_dim < 128:
                expand_amount = 4 if confidence > 0.3 else 8
                new_input_dim = min(self.input_dim + expand_amount, 128)
            else:
                return False

        if new_input_dim is not None and new_input_dim > self.input_dim:
            old_feat_dim = self.perception.get_feature_dim()

            old_states = {}
            for attr_name in ['perception', 'cognitive', 'dual_cognition', 'snn_enhancer',
                              'causal_reasoner', 'homeostasis', 'self_model']:
                attr = getattr(self, attr_name, None)
                if attr is not None and isinstance(attr, torch.nn.Module):
                    try:
                        old_states[attr_name] = {k: v.clone() for k, v in attr.state_dict().items()}
                    except Exception:
                        pass

            self.input_dim = new_input_dim
            self.perception = GrowingAutoEncoder(input_dim=new_input_dim).to(DEVICE)
            feat_dim = self.perception.get_feature_dim()

            self._migrate_weights('perception', old_states, old_feat_dim, feat_dim)
            self.multimodal = EnhancedMultimodalFusion({"primary": feat_dim}, output_dim=feat_dim)
            self.cognitive = CognitiveInferenceLayer(feat_dim=feat_dim)
            self.dual_cognition = DualSystemCognition(feat_dim=feat_dim)
            self.snn_enhancer = SNNEnhancer(feature_dim=feat_dim)
            self.causal_reasoner = CausalReasoningEngine(feature_dim=feat_dim)
            self.homeostasis = HomeostasisEngine(feature_dim=feat_dim)
            self.execution.hardware_adapt(feat_dim)
            self.self_model = SelfModel(state_dim=5, feature_dim=feat_dim, horizon=10)

            for attr_name in ['cognitive', 'dual_cognition', 'snn_enhancer',
                              'causal_reasoner', 'homeostasis', 'self_model']:
                self._migrate_weights(attr_name, old_states, old_feat_dim, feat_dim)

            self.decision_engine.resize(feat_dim)
            self.action_planner.resize(feat_dim)
            self.execution_monitor.resize(feat_dim)
            self.orchestrator = UnifiedCognitiveOrchestrator(
                perception=self.perception,
                cognition=self.cognitive,
                dual_cognition=self.dual_cognition,
                snn_enhancer=self.snn_enhancer,
                causal_reasoner=self.causal_reasoner,
                meta_cog=self.meta_cog,
                homeostasis=self.homeostasis,
                execution=self.execution,
                knowledge_graph=self.knowledge_graph,
                self_model=self.self_model
            )
            self.architecture_mutator = ArchitectureMutator(self.orchestrator)
            self.plugin_manager.notify_structure_change(feat_dim)

            self.reflex_controller.resize(feat_dim)
            self.thinking_orchestrator.resize(feat_dim)
            self.meta_cognitive_orchestrator.resize(feat_dim)
            self.action_orchestrator.resize(feat_dim)

            self.stereoscopic_snn = GeneralStereoscopicSNN(config={"feature_dim": feat_dim, "num_channels": feat_dim})
            self.enhanced_snn = EnhancedSNN(config={"num_neurons": 128, "num_layers": 3, "neurons_per_layer": [64, 32, feat_dim]})

            self.quad_level_evolution.set_interfaces(
                snn=self.reflex_controller.spiking_core,
                rule_engine=self.reflex_controller.rule_engine,
                knowledge_graph=self.knowledge_graph
            )

            self.audit_trail.log_entry("system", "hardware_self_expand", {
                "old_input_dim": self.input_dim - (new_input_dim - self.input_dim) if new_input_dim else self.input_dim,
                "new_input_dim": new_input_dim,
                "old_feat_dim": old_feat_dim,
                "new_feat_dim": feat_dim,
                "migrated_modules": list(old_states.keys()),
            })

            return True

        return False

    def adjust_input_dimension(self, new_dim: int) -> Dict[str, Any]:
        """动态调整输入维度，支持扩大和缩小。
        
        Args:
            new_dim: 新的输入维度，范围 [8, 128]
            
        Returns:
            调整结果字典
        """
        old_dim = self.input_dim
        
        if new_dim < 8 or new_dim > 128:
            return {
                "success": False,
                "error": f"输入维度必须在 [8, 128] 范围内，当前值: {new_dim}",
                "old_dim": old_dim,
                "new_dim": new_dim
            }
        
        if new_dim == old_dim:
            return {
                "success": True,
                "message": "输入维度已为目标值，无需调整",
                "old_dim": old_dim,
                "new_dim": new_dim
            }
        
        try:
            if new_dim > old_dim:
                result = self.hardware_self_expand(new_dim)
                if result:
                    return {
                        "success": True,
                        "message": f"输入维度已从 {old_dim} 扩展到 {new_dim}",
                        "old_dim": old_dim,
                        "new_dim": new_dim,
                        "type": "expand"
                    }
                else:
                    return {
                        "success": False,
                        "error": "扩展操作失败",
                        "old_dim": old_dim,
                        "new_dim": new_dim
                    }
            else:
                old_feat_dim = self.perception.get_feature_dim()
                
                self.input_dim = new_dim
                self.perception = GrowingAutoEncoder(input_dim=new_dim).to(DEVICE)
                feat_dim = self.perception.get_feature_dim()
                
                self.multimodal = EnhancedMultimodalFusion({"primary": feat_dim}, output_dim=feat_dim)
                self.cognitive = CognitiveInferenceLayer(feat_dim=feat_dim)
                self.dual_cognition = DualSystemCognition(feat_dim=feat_dim)
                self.snn_enhancer = SNNEnhancer(feature_dim=feat_dim)
                self.causal_reasoner = CausalReasoningEngine(feature_dim=feat_dim)
                self.homeostasis = HomeostasisEngine(feature_dim=feat_dim)
                self.execution.hardware_adapt(feat_dim)
                self.self_model = SelfModel(state_dim=5, feature_dim=feat_dim, horizon=10)
                
                self.decision_engine.resize(feat_dim)
                self.action_planner.resize(feat_dim)
                self.execution_monitor.resize(feat_dim)
                self.orchestrator = UnifiedCognitiveOrchestrator(
                    perception=self.perception,
                    cognition=self.cognitive,
                    dual_cognition=self.dual_cognition,
                    snn_enhancer=self.snn_enhancer,
                    causal_reasoner=self.causal_reasoner,
                    meta_cog=self.meta_cog,
                    homeostasis=self.homeostasis,
                    execution=self.execution,
                    knowledge_graph=self.knowledge_graph,
                    self_model=self.self_model
                )
                self.architecture_mutator = ArchitectureMutator(self.orchestrator)
                self.plugin_manager.notify_structure_change(feat_dim)
                
                self.reflex_controller.resize(feat_dim)
                self.thinking_orchestrator.resize(feat_dim)
                self.meta_cognitive_orchestrator.resize(feat_dim)
                self.action_orchestrator.resize(feat_dim)
                
                self.stereoscopic_snn = GeneralStereoscopicSNN(config={"feature_dim": feat_dim, "num_channels": feat_dim})
                self.enhanced_snn = EnhancedSNN(config={"num_neurons": 128, "num_layers": 3, "neurons_per_layer": [64, 32, feat_dim]})
                
                self.quad_level_evolution.set_interfaces(
                    snn=self.reflex_controller.spiking_core,
                    rule_engine=self.reflex_controller.rule_engine,
                    knowledge_graph=self.knowledge_graph
                )
                
                self.audit_trail.log_entry("system", "input_dimension_shrink", {
                    "old_input_dim": old_dim,
                    "new_input_dim": new_dim,
                    "old_feat_dim": old_feat_dim,
                    "new_feat_dim": feat_dim,
                })
                
                return {
                    "success": True,
                    "message": f"输入维度已从 {old_dim} 缩小到 {new_dim}",
                    "old_dim": old_dim,
                    "new_dim": new_dim,
                    "type": "shrink"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"调整输入维度时发生错误: {str(e)}",
                "old_dim": old_dim,
                "new_dim": new_dim
            }

    def get_input_dim_info(self) -> Dict[str, Any]:
        """获取当前输入维度信息。"""
        return {
            "current_dim": self.input_dim,
            "feature_dim": self.perception.get_feature_dim(),
            "min_dim": 8,
            "max_dim": 128,
            "suggested_dim": self.adaptive_config.get("input_dim", self.input_dim),
            "is_adaptive": True
        }

    def _migrate_weights(self, attr_name, old_states, old_feat_dim, new_feat_dim):
        if attr_name not in old_states or not old_states[attr_name]:
            return
        attr = getattr(self, attr_name, None)
        if attr is None or not isinstance(attr, torch.nn.Module):
            return
        try:
            new_sd = attr.state_dict()
            old_sd = old_states[attr_name]
            for k in new_sd:
                if k in old_sd:
                    old_shape = old_sd[k].shape
                    new_shape = new_sd[k].shape
                    if old_shape == new_shape:
                        new_sd[k] = old_sd[k].clone()
                    elif len(old_shape) == len(new_shape) == 2:
                        min_r = min(old_shape[0], new_shape[0])
                        min_c = min(old_shape[1], new_shape[1])
                        new_sd[k][:min_r, :min_c] = old_sd[k][:min_r, :min_c]
                    elif len(old_shape) == len(new_shape) == 1:
                        min_l = min(old_shape[0], new_shape[0])
                        new_sd[k][:min_l] = old_sd[k][:min_l]
            attr.load_state_dict(new_sd)
        except Exception as e:
            if hasattr(self, 'audit_trail'):
                self.audit_trail.log_entry("system", "weight_migration_failed", {
                    "module": attr_name, "error": str(e)
                })

    def _compute_reward(self, fe, entropy_val, kl_shift):
        fe_reward = max(0.0, 1.0 - fe / 0.5)
        entropy_reward = max(0.0, 0.5 - entropy_val / 5.0)
        novelty_reward = min(1.0, kl_shift / 0.3)

        reward = 0.5 * fe_reward + 0.3 * entropy_reward + 0.2 * novelty_reward
        return reward

    def set_autonomous_mode(self, enabled):
        self.autonomous_mode = enabled
        if enabled:
            self.reflex_controller.wake_up()
        else:
            self.reflex_controller.enter_sleep_mode()
        return {"autonomous_mode": enabled}

    def get_autonomous_status(self):
        return {
            "autonomous_mode": self.autonomous_mode,
            "thinking_stats": self.thinking_orchestrator.get_stats(),
            "action_stats": self.action_orchestrator.get_action_stats(),
            "meta_cognitive_stats": self.meta_cognitive_orchestrator.get_stats(),
            "evolution_stats": self.quad_level_evolution.get_all_stats(),
            "safety_status": {
                "hard_boundary": self.hard_boundary.get_status(),
                "circuit_breaker": self.circuit_breaker.get_state(),
                "risk_level": self.risk_classifier.get_stats().get('current_risk_level', 'low')
            }
        }

    def get_architecture_stats(self):
        def _convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_convert(item) for item in obj]
            return obj
        
        stats = {
            "reflex": self.reflex_controller.get_activity_summary(),
            "deliberative": self.thinking_orchestrator.get_stats(),
            "meta_cognitive": self.meta_cognitive_orchestrator.get_stats(),
            "stereoscopic_snn": {
                "total_processed": self.stereoscopic_snn.total_processed,
                "total_neurons": self.stereoscopic_snn._get_total_neurons(),
                "active_ensembles": self.stereoscopic_snn.state.ensemble_count,
                "pattern_info": self.stereoscopic_snn.get_pattern_info(),
                "working_memory_items": self.stereoscopic_snn.state.working_memory_count,
                "long_term_memory_items": self.stereoscopic_snn.state.long_term_memory_count
            },
            "knowledge_graph": self.knowledge_graph.get_summary()
        }
        return _convert(stats)

    def generate_report(self):
        report = {
            "agent_info": {
                "step": self.train_step,
                "input_dim": self.input_dim,
                "device": str(DEVICE),
                "autonomous_mode": self.autonomous_mode
            },
            "cognitive_metrics": self.meta_cog.get_all_metrics(),
            "homeostatic_state": self.homeostasis.goal_generator.get_homeostatic_state(),
            "performance": self.evaluator.get_evaluation_report(),
            "safety": self.safety_monitor.get_safety_report(),
            "compliance": self.compliance_checker.get_compliance_report(),
            "knowledge": self.cognitive.get_knowledge_summary(),
            "knowledge_graph": self.knowledge_graph.get_summary(),
            "evolution": self.evolve_engine.get_evolution_stats(),
            "storage": self.persistence.get_storage_info(),
            "decision_system": self.decision_engine.get_decision_stats(),
            "action_planner": self.action_planner.get_plan_stats(),
            "execution_monitor": self.execution_monitor.get_execution_stats(),
            "self_improvement": {
                "performance_evaluator": self.self_perf_evaluator.get_evaluation_stats(),
                "self_diagnostic": self.self_diagnostic.get_diagnostic_summary(),
                "self_improver": self.self_improver.get_improvement_stats(),
                "safety_verifier": self.safety_verifier.get_verification_stats(),
                "bootstrapped_improver": self.bootstrap_improver.get_bootstrapping_status(),
            },
            "memory_system": self.memory_harness.get_all_stats(),
            "soul": {
                "name": self.soul.identity.name,
                "version": self.soul.version.version,
                "personality": {
                    "rigor": self.soul.identity.personality.get("rigorousness", 50),
                    "creativity": self.soul.identity.personality.get("creativity", 50)
                }
            },
            "multi_agent": {
                "cluster_size": len(self.agent_cluster.get_available_agents()),
                "workspaces": len(self.workspace_manager.get_all_workspaces())
            },
            "task_engine": {
                "dag_stats": self.dag_engine.get_dag_stats(),
                "task_board_stats": self.task_board.get_stats(),
                "heartbeat_stats": self.heartbeat_scheduler.get_stats()
            },
            "dual_loop_evolution": self.dual_loop_evolution.get_stats(),
            "quad_level_evolution": self.quad_level_evolution.get_all_stats(),
            "metaskill": self.metaskill_generator.get_stats(),
            "three_layer_architecture": {
                "reflex": self.reflex_controller.get_activity_summary(),
                "deliberative": self.thinking_orchestrator.get_stats(),
                "meta_cognitive": self.meta_cognitive_orchestrator.get_stats()
            },
            "autonomous_action": self.action_orchestrator.get_action_stats(),
            "security_system": {
                "hard_boundary": self.hard_boundary.get_status(),
                "risk_classifier": self.risk_classifier.get_stats(),
                "circuit_breaker": self.circuit_breaker.get_state(),
                "audit_entries": len(self.audit_trail.get_recent_entries(limit=100))
            },
            "meta_programming": self.meta_programming.get_stats(),
            "meta_learning": self.meta_learning_orchestrator.get_overview(),
            "meta_decision": self.meta_decision_orchestrator.get_overview(),
            "meta_parsing": self.meta_parsing.get_overview(),
            "meta_evolution": self.meta_evolution.get_overview(),
            "meta_cognition": self.meta_cognitive_orchestrator.get_stats()
        }

        dashboard_path = self.visualizer.generate_dashboard(report['performance'])
        report["dashboard_path"] = dashboard_path

        return report

    def visualize_metrics(self):
        if len(self.metrics_history) >= 2:
            self.visualizer.plot_multiple_metrics(
                self.metrics_history,
                "AGI Agent Metrics Over Time",
                "agent_metrics"
            )

    def run(self, steps: int = 1000, env_generator: Optional[Callable[[], Any]] = None) -> Dict[str, Any]:
        """
        运行智能体主循环
        
        Args:
            steps: 运行步数，默认为1000
            env_generator: 环境观测生成器函数，返回观测数据
        
        Returns:
            运行结束后的综合报告字典
        """
        for _ in range(steps):
            if not self.running:
                break

            if env_generator is not None:
                obs = env_generator()
            else:
                obs = np.random.uniform(-1, 1, self.input_dim)

            self.step(obs)

            if self.train_step % 100 == 0:
                self.hardware_self_expand()

        self.visualize_metrics()
        return self.generate_report()


if __name__ == "__main__":
    agi_agent = SelfEvolvingAGI(input_dim=16)
    logging.getLogger("agi_agent").info("===== Self-Evolving AGI Agent Started =====")
    logging.getLogger("agi_agent").info("Core capabilities: Meta-cognition | Meta-learning | Unsupervised adaptation")
    logging.getLogger("agi_agent").info("                  Autonomous thinking | Self-evolution | Autonomous action")
    logging.getLogger("agi_agent").info("                  Memory System | SOUL Protocol | Multi-Agent Collaboration")
    logging.getLogger("agi_agent").info("                  Task Engine | Dual-Loop Evolution | Quad-Level Evolution")
    logging.getLogger("agi_agent").info("                  Three-Layer Mental Architecture | Safety Control System")

    report = agi_agent.run(steps=200)
    logging.getLogger("agi_agent").info("\n===== Run Report =====")
    logging.getLogger("agi_agent").info(f"Final Step: {report['agent_info']['step']}")
    logging.getLogger("agi_agent").info(f"Performance Score: {report['performance']['performance_score']['total_score']:.4f}")
    logging.getLogger("agi_agent").info(f"Free Energy: {report['cognitive_metrics']['cognitive']['free_energy']:.4f}")
    logging.getLogger("agi_agent").info(f"Confidence: {report['cognitive_metrics']['cognitive']['confidence']:.4f}")
    logging.getLogger("agi_agent").info(f"Knowledge Rules: {report['knowledge']['count']}")
    logging.getLogger("agi_agent").info(f"Safety Risk Level: {report['safety']['risk_level']}")
    logging.getLogger("agi_agent").info(f"Compliance Rate: {report['compliance']['compliance_rate']:.2f}")
    logging.getLogger("agi_agent").info(f"Memory Stats: {report['memory_system']}")
    logging.getLogger("agi_agent").info(f"Soul: {report['soul']['name']} v{report['soul']['version']}")
    logging.getLogger("agi_agent").info(f"Evolution Stats: {report['dual_loop_evolution']}")
    logging.getLogger("agi_agent").info(f"\nThree-Layer Architecture Stats:")
    logging.getLogger("agi_agent").info(f"  Reflex Layer: {report['three_layer_architecture']['reflex']}")
    logging.getLogger("agi_agent").info(f"  Deliberative Layer: {report['three_layer_architecture']['deliberative']}")
    logging.getLogger("agi_agent").info(f"  Meta-Cognitive Layer: {report['three_layer_architecture']['meta_cognitive']}")
    logging.getLogger("agi_agent").info(f"\nAutonomous Action Stats: {report['autonomous_action']}")
    logging.getLogger("agi_agent").info(f"Memory Stats: {report['memory_system']}")
    logging.getLogger("agi_agent").info(f"Soul: {report['soul']['name']} v{report['soul']['version']}")
    logging.getLogger("agi_agent").info(f"Evolution Stats: {report['dual_loop_evolution']}")
    logging.getLogger("agi_agent").info("\nThree-Layer Architecture Stats:")
    logging.getLogger("agi_agent").info(f"  Reflex Layer: {report['three_layer_architecture']['reflex']}")
    logging.getLogger("agi_agent").info(f"  Deliberative Layer: {report['three_layer_architecture']['deliberative']}")
    logging.getLogger("agi_agent").info(f"  Meta-Cognitive Layer: {report['three_layer_architecture']['meta_cognitive']}")
    logging.getLogger("agi_agent").info("\nAutonomous Action Stats:")
    logging.getLogger("agi_agent").info(f"  {report['autonomous_action']}")
    logging.getLogger("agi_agent").info("\nSecurity System:")
    logging.getLogger("agi_agent").info(f"  {report['security_system']}")
