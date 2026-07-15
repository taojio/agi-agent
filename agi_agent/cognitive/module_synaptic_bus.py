#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块突触总线 (Module Synaptic Bus)

实现类似于大脑功能区的模块间深度联动架构：

大脑功能区映射：
┌─────────────────────────────────────────────────────────────────────┐
│                        高级认知中枢                                  │
│  [Decision] ←→ [MetaCognition] ←→ [SelfModel] ←→ [Evolution]       │
│       ↑              ↑                ↑              ↑              │
├───────┼──────────────┼────────────────┼──────────────┼──────────────┤
│       │              │                │              │              │
│  [Memory] ←→ [KnowledgeGraph] ←→ [Skills] ←→ [Perception]          │
│       ↑              ↑                ↑              ↑              │
│  [Security] ←→ [SOUL] ←→ [Execution] ←→ [Homeostasis]              │
└─────────────────────────────────────────────────────────────────────┘

关键特性：
1. NeuralInterface: 每个模块的神经接口适配器
2. ModuleSynapse: 模块间的突触连接（支持STDP学习）
3. GlobalOscillator: 全局起搏器协调模块同步
4. SignalRouting: 脉冲信号路由与整合
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class SynapseType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


class ConnectionStrength(Enum):
    STRONG = 0.8
    MEDIUM = 0.5
    WEAK = 0.2


class SignalType(Enum):
    DATA = "data"
    CONTROL = "control"
    ACKNOWLEDGE = "acknowledge"
    REQUEST = "request"


@dataclass
class Spike:
    timestamp: float
    module_id: str
    neuron_index: int
    signal_type: SignalType
    payload: Dict = field(default_factory=dict)
    strength: float = 1.0


@dataclass
class ModuleSynapse:
    source_module: str
    target_module: str
    weight: float
    synapse_type: SynapseType = SynapseType.EXCITATORY
    delay: float = 0.0
    last_pre_spike: float = -float('inf')
    last_post_spike: float = -float('inf')
    stdp_trace: float = 0.0
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.01
    w_max: float = 1.0
    ltp_count: int = 0
    ltd_count: int = 0
    connection_count: int = 0
    last_active_time: float = -1.0


@dataclass
class ModuleActivity:
    module_id: str
    timestamp: float
    spike_count: int
    activation_level: float
    outgoing_signals: int
    incoming_signals: int
    synapses_activated: int


class NeuralInterface:
    def __init__(self, module_id: str, num_neurons: int = 32):
        self.module_id = module_id
        self.num_neurons = num_neurons
        self.neuron_potentials = np.zeros(num_neurons)
        self.last_spike_times = np.full(num_neurons, -float('inf'))
        self.spike_history = deque(maxlen=100)
        self.activation_threshold = 1.0
        self.tau_mem = 20.0
        self.tau_syn = 5.0
        
        self.state_encoding_dim = num_neurons
        self.last_encode_time = 0.0
        self.last_decode_time = 0.0

    def encode_state(self, state: Dict) -> List[Spike]:
        raise NotImplementedError("子类必须实现encode_state方法")

    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        raise NotImplementedError("子类必须实现decode_spikes方法")

    def process_spikes(self, spikes: List[Spike], dt: float = 1.0) -> List[Spike]:
        new_spikes = []
        
        for spike in spikes:
            neuron_idx = spike.neuron_index % self.num_neurons
            self.neuron_potentials[neuron_idx] += spike.strength * 0.3
            
        for i in range(self.num_neurons):
            self.neuron_potentials[i] *= np.exp(-dt / self.tau_mem)
            
            if self.neuron_potentials[i] >= self.activation_threshold:
                new_spikes.append(Spike(
                    timestamp=self.last_encode_time,
                    module_id=self.module_id,
                    neuron_index=i,
                    signal_type=SignalType.DATA,
                    strength=self.neuron_potentials[i],
                    payload={"neuron": i, "potential": float(self.neuron_potentials[i])}
                ))
                self.neuron_potentials[i] = 0.0
                self.last_spike_times[i] = self.last_encode_time
        
        return new_spikes

    def get_activity_summary(self) -> Dict:
        recent_spikes = list(self.spike_history)
        return {
            "module_id": self.module_id,
            "num_neurons": self.num_neurons,
            "active_neurons": int(np.sum(self.neuron_potentials > 0.1)),
            "avg_potential": float(np.mean(self.neuron_potentials)),
            "recent_spikes": len(recent_spikes),
            "spike_rate": len(recent_spikes) / 10.0 if self.last_encode_time > 0 else 0.0
        }


class GlobalOscillator:
    def __init__(self, base_frequency: float = 10.0):
        self.base_frequency = base_frequency
        self.t = 0.0
        self.phase = 0.0
        self.amplitude = 1.0
        
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        self.alpha_phase = 0.0

    def step(self, dt: float = 1.0):
        self.t += dt
        self.phase = (self.phase + 2 * np.pi * self.base_frequency * dt / 1000) % (2 * np.pi)
        
        self.theta_phase = (self.theta_phase + 2 * np.pi * 5.0 * dt / 1000) % (2 * np.pi)
        self.gamma_phase = (self.gamma_phase + 2 * np.pi * 40.0 * dt / 1000) % (2 * np.pi)
        self.alpha_phase = (self.alpha_phase + 2 * np.pi * 10.0 * dt / 1000) % (2 * np.pi)

    def get_theta(self) -> float:
        return self.amplitude * np.sin(self.theta_phase)

    def get_gamma(self) -> float:
        return self.amplitude * np.sin(self.gamma_phase)

    def get_alpha(self) -> float:
        return self.amplitude * np.sin(self.alpha_phase)

    def get_sync_signal(self) -> Dict:
        return {
            "theta": self.get_theta(),
            "gamma": self.get_gamma(),
            "alpha": self.get_alpha(),
            "phase": self.phase,
            "t": self.t
        }


class ModuleSynapticBus:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.modules: Dict[str, NeuralInterface] = {}
        self.synapses: List[ModuleSynapse] = []
        self.oscillator = GlobalOscillator()
        
        self.dt = self.config.get('dt', 1.0)
        self.t = 0.0
        
        self.spike_buffer: List[Spike] = []
        self.delayed_spikes: List[Tuple[float, Spike]] = []
        
        self.activity_history = deque(maxlen=1000)
        self.connection_strength_history = deque(maxlen=100)
        
        self.stdp_enabled = self.config.get('stdp_enabled', True)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        
        self._define_module_topology()
        self._initialize_synapses()
        
        logger.info(f"[OK] ModuleSynapticBus 初始化完成")
        logger.info(f"  - 注册模块数: {len(self.modules)}")
        logger.info(f"  - 突触连接数: {len(self.synapses)}")

    def _define_module_topology(self):
        self.module_list = [
            {'id': 'memory', 'name': '记忆系统', 'neurons': 64, 'region': 'hippocampus'},
            {'id': 'knowledge_graph', 'name': '知识图谱', 'neurons': 64, 'region': 'neocortex'},
            {'id': 'decision', 'name': '决策引擎', 'neurons': 48, 'region': 'prefrontal'},
            {'id': 'execution', 'name': '执行系统', 'neurons': 48, 'region': 'motor'},
            {'id': 'perception', 'name': '感知系统', 'neurons': 64, 'region': 'sensory'},
            {'id': 'security', 'name': '安全系统', 'neurons': 32, 'region': 'amygdala'},
            {'id': 'soul', 'name': 'SOUL系统', 'neurons': 48, 'region': 'cingulate'},
            {'id': 'skills', 'name': '技能系统', 'neurons': 48, 'region': 'striatum'},
            {'id': 'evolution', 'name': '进化引擎', 'neurons': 32, 'region': 'hippocampus'},
            {'id': 'self_improvement', 'name': '自我改进', 'neurons': 32, 'region': 'prefrontal'},
            {'id': 'metacognition', 'name': '元认知', 'neurons': 48, 'region': 'prefrontal'},
            {'id': 'homeostasis', 'name': '稳态系统', 'neurons': 32, 'region': 'brainstem'},
        ]
        
        self.connection_topology = [
            ('memory', 'knowledge_graph', ConnectionStrength.STRONG),
            ('knowledge_graph', 'decision', ConnectionStrength.STRONG),
            ('decision', 'execution', ConnectionStrength.STRONG),
            ('perception', 'knowledge_graph', ConnectionStrength.STRONG),
            ('security', 'execution', ConnectionStrength.STRONG),
            ('security', 'decision', ConnectionStrength.MEDIUM),
            ('soul', 'decision', ConnectionStrength.MEDIUM),
            ('soul', 'metacognition', ConnectionStrength.STRONG),
            ('skills', 'execution', ConnectionStrength.MEDIUM),
            ('skills', 'knowledge_graph', ConnectionStrength.WEAK),
            ('evolution', 'self_improvement', ConnectionStrength.STRONG),
            ('self_improvement', 'metacognition', ConnectionStrength.MEDIUM),
            ('metacognition', 'decision', ConnectionStrength.MEDIUM),
            ('homeostasis', 'security', ConnectionStrength.MEDIUM),
            ('homeostasis', 'execution', ConnectionStrength.WEAK),
            ('memory', 'decision', ConnectionStrength.MEDIUM),
            ('knowledge_graph', 'skills', ConnectionStrength.WEAK),
            ('evolution', 'knowledge_graph', ConnectionStrength.WEAK),
            ('metacognition', 'memory', ConnectionStrength.MEDIUM),
        ]

    def _initialize_synapses(self):
        for src_module, tgt_module, strength in self.connection_topology:
            weight = strength.value
            synapse = ModuleSynapse(
                source_module=src_module,
                target_module=tgt_module,
                weight=weight,
                synapse_type=SynapseType.EXCITATORY,
                delay=np.random.uniform(0.5, 3.0),
                connection_count=np.random.randint(1, 5)
            )
            self.synapses.append(synapse)
            
            reverse_synapse = ModuleSynapse(
                source_module=tgt_module,
                target_module=src_module,
                weight=weight * 0.5,
                synapse_type=SynapseType.INHIBITORY if np.random.random() < 0.3 else SynapseType.EXCITATORY,
                delay=np.random.uniform(0.5, 3.0),
                connection_count=np.random.randint(1, 3)
            )
            self.synapses.append(reverse_synapse)

    def register_module(self, module_id: str, interface: NeuralInterface):
        if module_id in self.modules:
            logger.warning(f"[WARN] 模块 {module_id} 已注册")
            return False
        
        self.modules[module_id] = interface
        logger.info(f"[OK] 模块 {module_id} 注册成功")
        return True

    def get_module(self, module_id: str) -> Optional[NeuralInterface]:
        return self.modules.get(module_id)

    def _route_spikes(self, spikes: List[Spike]) -> Dict[str, List[Spike]]:
        routed = defaultdict(list)
        
        for spike in spikes:
            for synapse in self.synapses:
                if synapse.source_module == spike.module_id:
                    delayed_time = self.t + synapse.delay
                    self.delayed_spikes.append((delayed_time, spike))
        
        return routed

    def _apply_stdp(self, synapse: ModuleSynapse, pre_spike_time: float, post_spike_time: float):
        if not self.stdp_enabled:
            return
        
        delta_t = pre_spike_time - post_spike_time
        
        if delta_t > 0:
            delta_w = synapse.A_minus * np.exp(-delta_t / synapse.tau_minus)
            synapse.ltd_count += 1
        else:
            delta_w = synapse.A_plus * np.exp(-abs(delta_t) / synapse.tau_plus)
            synapse.ltp_count += 1
        
        synapse.weight = max(synapse.w_min, min(synapse.w_max, synapse.weight + delta_w))
        synapse.stdp_trace += delta_w
        synapse.last_active_time = self.t

    def step(self, module_states: Dict[str, Dict] = None):
        self.t += self.dt
        self.oscillator.step(self.dt)
        
        all_spikes = []
        sync_signal = self.oscillator.get_sync_signal()
        
        if module_states:
            for module_id, state in module_states.items():
                interface = self.modules.get(module_id)
                if interface:
                    interface.last_encode_time = self.t
                    spikes = interface.encode_state(state)
                    spikes.extend(interface.process_spikes(spikes, self.dt))
                    all_spikes.extend(spikes)
                    interface.spike_history.extend(spikes)
        
        self._route_spikes(all_spikes)
        
        current_delayed = []
        for delayed_time, spike in self.delayed_spikes:
            if delayed_time <= self.t:
                current_delayed.append(spike)
        
        self.delayed_spikes = [d for d in self.delayed_spikes if d[0] > self.t]
        
        for spike in current_delayed:
            for synapse in self.synapses:
                if synapse.source_module == spike.module_id:
                    synapse.last_pre_spike = self.t
                    target_interface = self.modules.get(synapse.target_module)
                    if target_interface:
                        modified_spike = Spike(
                            timestamp=self.t,
                            module_id=synapse.target_module,
                            neuron_index=spike.neuron_index,
                            signal_type=spike.signal_type,
                            payload=spike.payload,
                            strength=spike.strength * synapse.weight
                        )
                        target_spikes = target_interface.process_spikes([modified_spike], self.dt)
                        target_interface.spike_history.extend(target_spikes)
                        
                        if target_spikes:
                            synapse.last_post_spike = self.t
                            self._apply_stdp(synapse, synapse.last_pre_spike, synapse.last_post_spike)
        
        self._record_activity()

    def _record_activity(self):
        activities = []
        for module_id, interface in self.modules.items():
            activity = ModuleActivity(
                module_id=module_id,
                timestamp=self.t,
                spike_count=len(interface.spike_history),
                activation_level=np.mean(interface.neuron_potentials),
                outgoing_signals=0,
                incoming_signals=0,
                synapses_activated=0
            )
            activities.append(activity)
        
        self.activity_history.append({
            'timestamp': self.t,
            'activities': activities,
            'sync_signal': self.oscillator.get_sync_signal()
        })

    def get_activity_summary(self) -> Dict:
        module_summaries = {}
        for module_id, interface in self.modules.items():
            module_summaries[module_id] = interface.get_activity_summary()
        
        synapse_summary = []
        for synapse in self.synapses:
            synapse_summary.append({
                'source': synapse.source_module,
                'target': synapse.target_module,
                'weight': synapse.weight,
                'type': synapse.synapse_type.value,
                'delay': synapse.delay,
                'ltp_count': synapse.ltp_count,
                'ltd_count': synapse.ltd_count,
                'last_active': synapse.last_active_time
            })
        
        return {
            'timestamp': self.t,
            'modules': module_summaries,
            'synapses': synapse_summary,
            'oscillator': self.oscillator.get_sync_signal(),
            'total_synapses': len(self.synapses),
            'total_modules': len(self.modules),
            'active_synapses': sum(1 for s in self.synapses if s.last_active_time > self.t - 100)
        }

    def get_connection_topology(self) -> Dict:
        nodes = []
        edges = []
        
        for module_info in self.module_list:
            interface = self.modules.get(module_info['id'])
            nodes.append({
                'id': module_info['id'],
                'name': module_info['name'],
                'region': module_info['region'],
                'neurons': module_info['neurons'],
                'active': interface.get_activity_summary() if interface else {}
            })
        
        for synapse in self.synapses:
            edges.append({
                'source': synapse.source_module,
                'target': synapse.target_module,
                'weight': synapse.weight,
                'type': synapse.synapse_type.value,
                'ltp_count': synapse.ltp_count,
                'ltd_count': synapse.ltd_count
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }

    def propagate_signal(self, source_module: str, signal_type: SignalType, payload: Dict = None):
        interface = self.modules.get(source_module)
        if not interface:
            return
        
        spikes = interface.encode_state({
            'signal_type': signal_type.value,
            'payload': payload or {}
        })
        
        for spike in spikes:
            for synapse in self.synapses:
                if synapse.source_module == source_module:
                    target_interface = self.modules.get(synapse.target_module)
                    if target_interface:
                        target_spike = Spike(
                            timestamp=self.t,
                            module_id=synapse.target_module,
                            neuron_index=spike.neuron_index,
                            signal_type=signal_type,
                            payload=payload or {},
                            strength=spike.strength * synapse.weight
                        )
                        target_interface.spike_history.append(target_spike)

    def get_signal_flow(self, recent_ms: float = 1000.0) -> Dict:
        flow_data = defaultdict(lambda: defaultdict(int))
        cutoff_time = self.t - recent_ms
        
        for activity in self.activity_history:
            if activity['timestamp'] < cutoff_time:
                continue
            for act in activity['activities']:
                flow_data[act.module_id]['spikes'] += act.spike_count
        
        return {
            'time_range': [cutoff_time, self.t],
            'signal_flow': dict(flow_data)
        }