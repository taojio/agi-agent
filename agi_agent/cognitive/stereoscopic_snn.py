#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
立体脉冲神经网络（Stereoscopic SNN）v3.0

基于人脑知识存储机制的五个层面实现：

一、底层存储单元：突触可塑性是知识的物理载体
   - 全网络STDP：感知通路、认知通路、交叉连接的所有突触均具备可塑性
   - 精确到毫秒级的LTP/LTD规则

二、知识的编码形式：神经元集群的时空脉冲模式
   - 分布式编码：跨脑区神经元协同活动表征概念
   - 时间维度：毫秒级脉冲同步性、发放顺序编码
   - 空间维度：多脑区特征整合

三、记忆的巩固：短时动态维持 vs 长时结构固化
   - 短时工作记忆：回响活动维持动态模式
   - 长时记忆：STDP驱动权重持久改变
   - 记忆巩固机制：重复激活强化连接

四、知识的提取：联想记忆与吸引子动力学
   - 海马CA3循环连接实现模式补全
   - 吸引子网络特性：抗噪声、抗失真
   - 链式脉冲序列复现时序记忆

五、知识的层级抽象：多层网络对应皮层层级加工
   - 底层：感官特征提取
   - 中层：物体/模式编码
   - 高层：抽象概念表征

版本: 3.0
日期: 2026-07-03
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time


class ConnectionType(Enum):
    FEEDFORWARD = "feedforward"
    FEEDBACK = "feedback"
    LATERAL = "lateral"
    RECURRENT = "recurrent"


@dataclass
class SynapticConnection:
    pre_neuron: int
    post_neuron: int
    weight: float
    connection_type: ConnectionType
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


@dataclass
class CrossConnection:
    source_neuron: int
    target_neuron: int
    weight: float
    connection_type: ConnectionType
    source_path: str
    target_path: str
    source_layer: str
    target_layer: str
    delay: float = 0.0
    last_active_t: float = -1.0
    stdp_trace: float = 0.0


@dataclass
class NeuralEnsemble:
    ensemble_id: str
    neurons: Set[int]
    activation_pattern: List[Tuple[int, float]] = field(default_factory=list)
    coherence_score: float = 0.0
    last_activation_time: float = 0.0
    stability: float = 0.0


@dataclass
class WorkingMemoryItem:
    pattern_id: str
    spike_pattern: List[Tuple[int, float]]
    activation_time: float
    decay_rate: float = 0.95
    strength: float = 1.0


@dataclass
class StereoscopicState:
    perceptual_spikes: Dict[str, int] = field(default_factory=dict)
    cognitive_spikes: int = 0
    cross_connection_count: int = 0
    active_connections: int = 0
    total_spikes: int = 0
    information_flow: Dict[str, float] = field(default_factory=dict)
    fusion_activity: float = 0.0
    working_memory_count: int = 0
    long_term_memory_count: int = 0
    ensemble_count: int = 0


class StereoscopicSNN:
    """
    立体脉冲神经网络 v3.0

    双通道立体架构（神经元数量大幅提升）：
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      高级认知融合区                                  │
    │  海马(500) ←→ 基底节(300) ←→ 认知输出(40) ←→ 储备池(50)            │
    │                       融合层(100神经元)                              │
    └────────▲──────────────────────────▲─────────────────────────────────┘
             │                          │
    ┌────────┴──────────────────────────┴─────────────────────────────────┐
    │                      特征整合层                                      │
    │  听觉皮层(320神经元)    ←→    认知隐藏2层(50神经元)                  │
    └────────▲──────────────────────────▲─────────────────────────────────┘
             │                          │
    ┌────────┴──────────────────────────┴─────────────────────────────────┐
    │                      初级处理层                                      │
    │  耳蜗核(128) ←→ 下丘(80)    ←→    认知隐藏1层(50神经元)              │
    └────────▲──────────────────────────▲─────────────────────────────────┘
             │                          │
    ┌────────┴──────────────────────────┴─────────────────────────────────┐
    │                      感知输入层                                      │
    │  耳蜗换能(128通道)   ←→    认知输入层(60神经元)                      │
    └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.sample_rate = self.config.get('sample_rate', 44100)
        self.num_channels = self.config.get('num_channels', 128)
        self.sparsity = self.config.get('sparsity', 0.15)
        self.dt = 1.0
        self.t = 0.0

        self._init_perceptual_path()
        self._init_cognitive_path()

        self.state = StereoscopicState()

        self.cross_connections: List[CrossConnection] = []
        self._build_cross_connections()

        self._init_intra_path_synapses()

        self._init_hippocampal_attractor_network()

        self._init_working_memory()

        self.neural_ensembles: Dict[str, NeuralEnsemble] = {}
        self._detect_ensembles()

        self.delay_buffer: List[Tuple[float, int, float, str]] = []

        self.learning_enabled = True
        self.stdp_tau_plus = 20.0
        self.stdp_tau_minus = 20.0
        self.stdp_a_plus = 0.01
        self.stdp_a_minus = 0.012

        self._init_fusion_layer()
        self._init_pacemaker_population()
        self._init_tonality_learning()

        self.spike_history = deque(maxlen=500)
        self.rhythm_patterns: Dict[str, List[float]] = {}

        self.total_processing_time = 0.0
        self.total_processed = 0

        print(f"[OK] 立体SNN v3.0 初始化完成")
        print(f"  - 感知通路: 6层, {sum(self.perceptual_layer_sizes.values())} 神经元")
        print(f"  - 认知通路: 5层SNN({sum(self.cognitive_layer_sizes.values())}神经元) + 储备池(50节点)")
        print(f"  - 交叉连接: {len(self.cross_connections)} 个 (稀疏度: {self.sparsity*100:.0f}%)")
        print(f"  - 融合层: {self.fusion_neurons} 神经元")
        print(f"  - 总神经元数: {self._get_total_neurons()}")
        print(f"  - 海马循环连接: {len(self.hippocampus_recurrent_synapses)}")

    def _init_perceptual_path(self):
        from bio_auditory_snn import (
            CochleaModel, CochlearNucleus, InferiorColliculus,
            AuditoryCortex, HippocampalMemory, BasalGanglia
        )

        self.perceptual_config = {
            'num_channels': self.num_channels,
            'sample_rate': self.sample_rate,
        }

        self.cochlea = CochleaModel(self.num_channels, self.sample_rate)
        self.cochlear_nucleus = CochlearNucleus(self.num_channels)
        self.inferior_colliculus = InferiorColliculus(self.num_channels // 2)
        self.auditory_cortex = AuditoryCortex(self.num_channels // 4)
        self.hippocampus = HippocampalMemory(500)
        self.basal_ganglia = BasalGanglia(300)

        self.perceptual_layer_sizes = {
            'cochlea': self.num_channels,
            'cochlear_nucleus': self.num_channels,
            'inferior_colliculus': self.num_channels // 2 + 16,
            'auditory_cortex': (self.num_channels // 4) * 10,
            'hippocampus': 500,
            'basal_ganglia': 300
        }

        self.perceptual_layer_offsets = {
            'cochlea': 0,
            'cochlear_nucleus': self.num_channels,
            'inferior_colliculus': self.num_channels + self.num_channels,
            'auditory_cortex': self.num_channels + self.num_channels + (self.num_channels // 2 + 16),
            'hippocampus': self.num_channels + self.num_channels + (self.num_channels // 2 + 16) + ((self.num_channels // 4) * 10),
            'basal_ganglia': self.num_channels + self.num_channels + (self.num_channels // 2 + 16) + ((self.num_channels // 4) * 10) + 500
        }

    def _init_cognitive_path(self):
        from enhanced_snn import EnhancedSNN, NeuronType, SelfOrganizingChaosReservoir

        cog_input = 60
        cog_h1 = 50
        cog_h2 = 50
        cog_out = 40

        cognitive_config = {
            'num_neurons': cog_input + cog_h1 + cog_h2 + cog_out,
            'num_layers': 4,
            'neurons_per_layer': [cog_input, cog_h1, cog_h2, cog_out],
            'neuron_type': NeuronType.DENDRISTOR,
            'learning_rate': 0.001
        }

        self.cognitive_snn = EnhancedSNN(cognitive_config)
        self.reservoir = SelfOrganizingChaosReservoir({'num_nodes': 50})

        self.cognitive_layer_sizes = {
            'input': cog_input,
            'hidden1': cog_h1,
            'hidden2': cog_h2,
            'output': cog_out,
            'reservoir': 50
        }

        self.cognitive_layer_boundaries = [0, cog_input, cog_input+cog_h1, cog_input+cog_h1+cog_h2, cog_input+cog_h1+cog_h2+cog_out]

    def _build_cross_connections(self):
        layer_mappings = [
            ('cochlea', 'input', ConnectionType.FEEDFORWARD, 1.0),
            ('cochlear_nucleus', 'input', ConnectionType.FEEDFORWARD, 1.2),
            ('inferior_colliculus', 'hidden1', ConnectionType.FEEDFORWARD, 1.5),
            ('auditory_cortex', 'hidden2', ConnectionType.FEEDFORWARD, 1.8),
            ('hippocampus', 'output', ConnectionType.FEEDFORWARD, 1.2),
            ('basal_ganglia', 'output', ConnectionType.FEEDFORWARD, 1.0),
            ('output', 'auditory_cortex', ConnectionType.FEEDBACK, 0.8),
            ('output', 'hippocampus', ConnectionType.FEEDBACK, 0.6),
            ('hidden2', 'inferior_colliculus', ConnectionType.FEEDBACK, 0.6),
            ('hidden1', 'cochlear_nucleus', ConnectionType.FEEDBACK, 0.5),
        ]

        for src_layer, tgt_layer, conn_type, rate_mult in layer_mappings:
            if conn_type == ConnectionType.FEEDFORWARD:
                src_size = self.perceptual_layer_sizes.get(src_layer, 128)
                tgt_size = self.cognitive_layer_sizes.get(tgt_layer, 60)
                src_path = 'perceptual'
                tgt_path = 'cognitive'
            else:
                src_size = self.cognitive_layer_sizes.get(src_layer, 40)
                tgt_size = self.perceptual_layer_sizes.get(tgt_layer, 128)
                src_path = 'cognitive'
                tgt_path = 'perceptual'

            base_connections = int(min(src_size, tgt_size) * self.sparsity * rate_mult)
            num_connections = max(5, base_connections)

            src_neurons = np.random.choice(src_size, min(num_connections, src_size), replace=False)

            for src_neuron in src_neurons:
                tgt_neuron = np.random.randint(0, tgt_size)
                weight = np.random.uniform(0.05, 0.25)
                if conn_type == ConnectionType.FEEDBACK:
                    weight *= 0.5

                delay = np.random.uniform(0.5, 5.0)

                connection = CrossConnection(
                    source_neuron=int(src_neuron),
                    target_neuron=int(tgt_neuron),
                    weight=weight,
                    connection_type=conn_type,
                    source_path=src_path,
                    target_path=tgt_path,
                    source_layer=src_layer,
                    target_layer=tgt_layer,
                    delay=delay
                )

                self.cross_connections.append(connection)

        self.state.cross_connection_count = len(self.cross_connections)

    def _init_intra_path_synapses(self):
        self.perceptual_synapses: Dict[str, List[SynapticConnection]] = {}
        self.cognitive_synapses: List[SynapticConnection] = []

        perceptual_layers = ['cochlear_nucleus', 'inferior_colliculus', 'auditory_cortex', 'hippocampus']
        for layer_name in perceptual_layers:
            layer_size = self.perceptual_layer_sizes[layer_name]
            synapses = []
            for i in range(layer_size):
                for j in range(layer_size):
                    if i != j and np.random.random() < self.sparsity:
                        synapse = SynapticConnection(
                            pre_neuron=i,
                            post_neuron=j,
                            weight=np.random.uniform(0.05, 0.25),
                            connection_type=ConnectionType.LATERAL,
                            delay=np.random.uniform(0.5, 3.0)
                        )
                        synapses.append(synapse)
            self.perceptual_synapses[layer_name] = synapses

        for layer_idx in range(3):
            layer_size = self.cognitive_layer_sizes[['input', 'hidden1', 'hidden2'][layer_idx]]
            next_layer_size = self.cognitive_layer_sizes[['hidden1', 'hidden2', 'output'][layer_idx]]
            offset = self.cognitive_layer_boundaries[layer_idx]
            next_offset = self.cognitive_layer_boundaries[layer_idx + 1]

            for i in range(layer_size):
                for j in range(next_layer_size):
                    if np.random.random() < self.sparsity:
                        synapse = SynapticConnection(
                            pre_neuron=offset + i,
                            post_neuron=next_offset + j,
                            weight=np.random.uniform(0.05, 0.3),
                            connection_type=ConnectionType.FEEDFORWARD,
                            delay=np.random.uniform(0.5, 3.0)
                        )
                        self.cognitive_synapses.append(synapse)

    def _init_hippocampal_attractor_network(self):
        self.hippocampus_recurrent_synapses: List[SynapticConnection] = []
        hipp_size = 500

        for i in range(hipp_size):
            for j in range(hipp_size):
                if i != j and np.random.random() < 0.2:
                    synapse = SynapticConnection(
                        pre_neuron=i,
                        post_neuron=j,
                        weight=np.random.uniform(0.1, 0.35),
                        connection_type=ConnectionType.RECURRENT,
                        delay=np.random.uniform(1.0, 5.0),
                        A_plus=0.015,
                        A_minus=0.018,
                        tau_plus=25.0,
                        tau_minus=25.0
                    )
                    self.hippocampus_recurrent_synapses.append(synapse)

        self.attractor_basins: Dict[str, List[int]] = {}
        self.attractor_activation: Dict[str, float] = {}

    def _init_working_memory(self):
        self.working_memory: Dict[str, WorkingMemoryItem] = {}
        self.working_memory_capacity = 7
        self.working_memory_decay = 0.95
        self.consolidation_threshold = 5

    def _init_fusion_layer(self):
        self.fusion_neurons = 100
        self.fusion_state = np.zeros(self.fusion_neurons)
        self.fusion_threshold = 0.5

        percep_out_size = self.perceptual_layer_sizes['hippocampus'] + self.perceptual_layer_sizes['basal_ganglia']
        cog_out_size = self.cognitive_layer_sizes['output'] + self.cognitive_layer_sizes['reservoir']

        self.percep_to_fusion = np.random.uniform(-0.1, 0.2, (percep_out_size, self.fusion_neurons)) * 0.15
        self.cog_to_fusion = np.random.uniform(-0.1, 0.2, (cog_out_size, self.fusion_neurons)) * 0.15
        self.fusion_bias = np.zeros(self.fusion_neurons)

    def _init_pacemaker_population(self):
        self.pacemaker_neurons = 32
        
        self.pacemaker_phases = np.zeros(self.pacemaker_neurons)
        
        self.pacemaker_frequencies = np.logspace(0.5, 2.5, self.pacemaker_neurons)
        
        self.pacemaker_amplitudes = np.ones(self.pacemaker_neurons)
        
        self.pacemaker_input_weights = np.random.uniform(0.1, 0.3, (self.pacemaker_neurons, 128))
        
        self.pacemaker_output_weights = np.random.uniform(0.1, 0.2, (128, self.pacemaker_neurons))
        
        self.beat_predictions = deque(maxlen=100)
        
        self.current_bpm = 120.0
        self.bpm_history = deque(maxlen=50)
        
        self.beat_strengths = np.zeros(8)
        self.beat_index = 0

    def _update_pacemaker(self, current_t_ms: float, input_spikes: List[Any] = None):
        dt = self.dt / 1000.0
        
        input_influence = np.zeros(self.pacemaker_neurons)
        
        if input_spikes:
            for spike in input_spikes:
                neuron_idx = spike.channel % self.pacemaker_neurons if hasattr(spike, 'channel') else 0
                input_influence += self.pacemaker_input_weights[:, neuron_idx] * 0.5
        
        self.pacemaker_phases += 2 * np.pi * self.pacemaker_frequencies * dt * (self.current_bpm / 120.0)
        
        self.pacemaker_phases = self.pacemaker_phases % (2 * np.pi)
        
        oscillatory_activity = np.sin(self.pacemaker_phases) * self.pacemaker_amplitudes
        
        self.pacemaker_amplitudes = np.clip(
            self.pacemaker_amplitudes + input_influence * dt - 0.01 * dt,
            0.5, 2.0
        )
        
        beat_spikes = []
        for i in range(self.pacemaker_neurons):
            if self.pacemaker_phases[i] < 0.1 and self.pacemaker_phases[i] > -0.1:
                beat_spikes.append(i)
                self.beat_index = (self.beat_index + 1) % 8
                self.beat_strengths[self.beat_index] = self.pacemaker_amplitudes[i]
        
        if input_spikes and beat_spikes:
            self.bpm_history.append(self.current_bpm)
            if len(self.bpm_history) > 10:
                self.current_bpm = np.mean(self.bpm_history)
        
        return beat_spikes

    def _init_tonality_learning(self):
        self.tonality_weights = np.ones((12, 12)) * 0.5
        
        self.pitch_counts = np.zeros(12)
        
        self.transition_counts = np.zeros((12, 12))
        
        self.tonic_activation = np.zeros(12)
        
        self.mode_profiles = {
            'major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            'minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
            'dorian': [6.20, 2.80, 3.50, 5.40, 2.60, 4.20, 2.40, 4.80, 3.90, 2.70, 3.40, 3.00],
            'phrygian': [6.10, 5.50, 3.40, 5.30, 2.50, 4.00, 2.30, 4.70, 3.80, 2.60, 3.30, 2.90],
            'lydian': [6.40, 2.20, 4.50, 2.30, 4.40, 4.10, 2.50, 5.20, 2.40, 3.70, 2.30, 2.90],
            'mixolydian': [6.30, 2.20, 3.50, 2.30, 4.40, 4.10, 2.50, 5.20, 2.40, 3.70, 3.80, 2.90],
            'locrian': [5.00, 5.50, 3.40, 5.30, 2.50, 4.00, 4.50, 4.70, 3.80, 2.60, 3.30, 2.90],
        }
        
        self.current_mode = 'major'
        self.mode_confidence = 0.0
        
        self.key_strengths = np.zeros(12)
        
        self.transition_history = deque(maxlen=1000)
        
        self.pitch_prediction = np.zeros(12)
        
        self.prediction_error = 0.0
        self.error_history = deque(maxlen=100)
        
        self.tonal_gravity = np.zeros(12)
        
        self.leading_tone_strength = 0.0
        self.resolution_quality = 0.0
        
        self.cadence_detection = {
            'authentic': 0.0,
            'plagal': 0.0,
            'half': 0.0,
            'deceptive': 0.0,
        }
        
        self.cadence_history = deque(maxlen=20)
        
        self._init_pitch_rhythm_binding()

    def _init_pitch_rhythm_binding(self):
        self.binding_time_window = 50.0
        
        self.pitch_rhythm_binding_weights = np.random.uniform(0.1, 0.3, (128, 12))
        
        self.active_binding_units = []
        
        self.binding_strengths = deque(maxlen=50)
        
        self.melody_events: List[Dict] = []
        
        self.current_event_id = 0
        
        self.pitch_neurons_active = np.zeros(128, dtype=bool)
        self.rhythm_neurons_active = np.zeros(12, dtype=bool)
        
        self.binding_sync_threshold = 0.7

    def _bind_pitch_rhythm(self, pitch_spikes: List, rhythm_spikes: List, current_time_ms: float):
        self.pitch_neurons_active[:] = False
        self.rhythm_neurons_active[:] = False
        
        for spike in pitch_spikes:
            if hasattr(spike, 'channel') and spike.channel < 128:
                self.pitch_neurons_active[spike.channel] = True
        
        for spike in rhythm_spikes:
            if hasattr(spike, 'channel') and spike.channel < 12:
                self.rhythm_neurons_active[spike.channel] = True
        
        pitch_active_count = np.sum(self.pitch_neurons_active)
        rhythm_active_count = np.sum(self.rhythm_neurons_active)
        
        if pitch_active_count > 0 and rhythm_active_count > 0:
            binding_strength = 0.7 + (pitch_active_count * rhythm_active_count) / (128 * 12) * 0.3
        else:
            binding_strength = 0.0
        
        self.binding_strengths.append(binding_strength)
        
        if binding_strength > self.binding_sync_threshold:
            self.current_event_id += 1
            
            event = {
                'event_id': self.current_event_id,
                'time_ms': current_time_ms,
                'pitch_channels': np.where(self.pitch_neurons_active)[0].tolist(),
                'rhythm_channels': np.where(self.rhythm_neurons_active)[0].tolist(),
                'binding_strength': binding_strength,
                'is_bound': True,
            }
            self.melody_events.append(event)
            self.active_binding_units.append(event)
            
            if len(self.active_binding_units) > 10:
                self.active_binding_units = self.active_binding_units[-10:]
            
            self.pitch_rhythm_binding_weights[self.pitch_neurons_active, :] += 0.01
            self.pitch_rhythm_binding_weights[:, self.rhythm_neurons_active] += 0.01
            self.pitch_rhythm_binding_weights = np.clip(self.pitch_rhythm_binding_weights, 0.01, 1.0)
        
        return binding_strength

    def _predict_next_pitch(self, current_pitch: int) -> Tuple[int, float]:
        if 0 <= current_pitch < 12:
            self.pitch_prediction = self.tonality_weights[current_pitch]
        else:
            self.pitch_prediction = np.zeros(12)
        
        if np.sum(self.pitch_prediction) > 0:
            self.pitch_prediction = self.pitch_prediction / np.sum(self.pitch_prediction)
        
        predicted_pitch = int(np.argmax(self.pitch_prediction))
        prediction_strength = float(self.pitch_prediction[predicted_pitch])
        
        self.tonal_gravity = self.pitch_prediction
        
        if current_pitch == 10:
            self.leading_tone_strength = prediction_strength
        
        return predicted_pitch, prediction_strength

    def _calculate_prediction_error(self, predicted_pitch: int, actual_pitch: int) -> float:
        if predicted_pitch == actual_pitch:
            error = 0.0
            self.resolution_quality = 1.0
        else:
            error = 1.0 - (self.pitch_prediction[actual_pitch] if 0 <= actual_pitch < 12 else 0)
            self.resolution_quality = 1.0 - error
        
        self.prediction_error = error
        self.error_history.append(error)
        
        return error

    def _detect_cadence(self, pitch_sequence: List[int]) -> Dict:
        if len(pitch_sequence) < 2:
            return {}
        
        last_two = pitch_sequence[-2:]
        
        tonic = int(np.argmax(self.key_strengths))
        dominant = (tonic + 7) % 12
        subdominant = (tonic + 5) % 12
        
        self.cadence_detection = {
            'authentic': 0.0,
            'plagal': 0.0,
            'half': 0.0,
            'deceptive': 0.0,
        }
        
        if last_two[0] == dominant and last_two[1] == tonic:
            self.cadence_detection['authentic'] = 1.0 - self.prediction_error
            self.cadence_history.append(('authentic', self.t))
        elif last_two[0] == subdominant and last_two[1] == tonic:
            self.cadence_detection['plagal'] = 0.8 - self.prediction_error * 0.5
            self.cadence_history.append(('plagal', self.t))
        elif last_two[1] == dominant:
            self.cadence_detection['half'] = 0.5
            self.cadence_history.append(('half', self.t))
        elif last_two[0] == dominant and last_two[1] != tonic:
            self.cadence_detection['deceptive'] = 0.7
            self.cadence_history.append(('deceptive', self.t))
        
        return self.cadence_detection

    def _learn_tonality(self, pitch_sequence: List[int]):
        for pitch in pitch_sequence:
            if 0 <= pitch < 12:
                self.pitch_counts[pitch] += 1
        
        for i in range(len(pitch_sequence) - 1):
            from_pitch = pitch_sequence[i]
            to_pitch = pitch_sequence[i + 1]
            if 0 <= from_pitch < 12 and 0 <= to_pitch < 12:
                self.transition_counts[from_pitch, to_pitch] += 1
                self.transition_history.append((from_pitch, to_pitch))
        
        total_counts = np.sum(self.pitch_counts)
        if total_counts > 0:
            self.tonic_activation = self.pitch_counts / total_counts
        
        total_transitions = np.sum(self.transition_counts)
        if total_transitions > 0:
            self.tonality_weights = self.transition_counts / total_transitions
        
        self._detect_key()

    def _detect_key(self):
        if np.sum(self.pitch_counts) == 0:
            return
        
        normalized_counts = self.pitch_counts / np.max(self.pitch_counts)
        
        best_key = 0
        best_mode = 'major'
        best_correlation = 0.0
        
        for mode_name, profile in self.mode_profiles.items():
            profile_np = np.array(profile)
            profile_normalized = profile_np / np.max(profile_np)
            
            for key_shift in range(12):
                shifted_profile = np.roll(profile_normalized, key_shift)
                correlation = np.corrcoef(normalized_counts, shifted_profile)[0, 1]
                
                if correlation > best_correlation and not np.isnan(correlation):
                    best_correlation = correlation
                    best_key = key_shift
                    best_mode = mode_name
        
        self.key_strengths[best_key] = best_correlation
        self.current_mode = best_mode
        self.mode_confidence = best_correlation
    
    def get_tonality_info(self) -> Dict:
        return {
            'key_strengths': self.key_strengths.tolist(),
            'current_mode': self.current_mode,
            'mode_confidence': float(self.mode_confidence),
            'pitch_counts': self.pitch_counts.tolist(),
            'tonic_activation': self.tonic_activation.tolist(),
            'most_active_pitch': int(np.argmax(self.pitch_counts)) if np.sum(self.pitch_counts) > 0 else -1,
        }

    def _get_total_neurons(self) -> int:
        percep_total = sum(self.perceptual_layer_sizes.values())
        cog_total = sum(self.cognitive_layer_sizes.values())
        return percep_total + cog_total + self.fusion_neurons

    def _detect_ensembles(self):
        for layer_name in ['auditory_cortex', 'hippocampus']:
            layer_size = self.perceptual_layer_sizes[layer_name]
            num_ensembles = min(20, layer_size // 25)

            for i in range(num_ensembles):
                ensemble_neurons = set(np.random.choice(layer_size, 25, replace=False))
                ensemble = NeuralEnsemble(
                    ensemble_id=f"{layer_name}_ensemble_{i}",
                    neurons=ensemble_neurons
                )
                self.neural_ensembles[ensemble.ensemble_id] = ensemble

        self.state.ensemble_count = len(self.neural_ensembles)

    def _update_ensemble_activity(self, layer_name: str, spikes: List[Any], current_t: float):
        for ensemble_id, ensemble in self.neural_ensembles.items():
            if layer_name not in ensemble_id:
                continue

            active_count = 0
            for spike in spikes:
                spike_channel = spike.channel if hasattr(spike, 'channel') else spike
                if spike_channel in ensemble.neurons:
                    active_count += 1
                    ensemble.activation_pattern.append((spike_channel, current_t))

            if len(ensemble.activation_pattern) > 50:
                ensemble.activation_pattern = ensemble.activation_pattern[-50:]

            ensemble.coherence_score = active_count / len(ensemble.neurons)
            if ensemble.coherence_score > 0.3:
                ensemble.last_activation_time = current_t
                ensemble.stability = min(1.0, ensemble.stability + 0.1)

    def _apply_global_stdp(self, current_t: float):
        for layer_name, synapses in self.perceptual_synapses.items():
            for synapse in synapses:
                self._apply_stdp_rule(synapse, current_t)

        for synapse in self.cognitive_synapses:
            self._apply_stdp_rule(synapse, current_t)

        for synapse in self.hippocampus_recurrent_synapses:
            self._apply_stdp_rule(synapse, current_t)

        for conn in self.cross_connections:
            self._apply_cross_connection_stdp(conn, current_t)

    def _apply_stdp_rule(self, synapse: SynapticConnection, current_t: float):
        delta_t = synapse.last_post_spike - synapse.last_pre_spike

        if delta_t > 0 and delta_t < 50.0:
            dw = synapse.A_plus * np.exp(-delta_t / synapse.tau_plus)
            synapse.weight += dw
            synapse.ltp_count += 1
        elif delta_t < 0 and delta_t > -50.0:
            dw = -synapse.A_minus * np.exp(delta_t / synapse.tau_minus)
            synapse.weight += dw
            synapse.ltd_count += 1

        synapse.weight = max(synapse.w_min, min(synapse.w_max, synapse.weight))

    def _apply_cross_connection_stdp(self, conn: CrossConnection, current_t: float):
        delta_t = conn.stdp_trace

        if delta_t > 0:
            conn.weight += self.stdp_a_plus * np.exp(-delta_t / self.stdp_tau_plus)
        elif delta_t < 0:
            conn.weight -= self.stdp_a_minus * np.exp(delta_t / self.stdp_tau_minus)

        conn.weight = max(0.01, min(0.5, conn.weight))

    def _update_working_memory(self, pattern_id: str, spike_pattern: List[Tuple[int, float]], current_t: float):
        if pattern_id in self.working_memory:
            item = self.working_memory[pattern_id]
            item.strength = min(1.0, item.strength + 0.1)
            item.activation_time = current_t

            if item.strength >= self.consolidation_threshold:
                self._consolidate_to_long_term(pattern_id)
                del self.working_memory[pattern_id]
        else:
            if len(self.working_memory) >= self.working_memory_capacity:
                oldest_key = min(self.working_memory.keys(), key=lambda k: self.working_memory[k].activation_time)
                del self.working_memory[oldest_key]

            self.working_memory[pattern_id] = WorkingMemoryItem(
                pattern_id=pattern_id,
                spike_pattern=spike_pattern,
                activation_time=current_t
            )

        for key in list(self.working_memory.keys()):
            item = self.working_memory[key]
            item.strength *= self.working_memory_decay
            if item.strength < 0.1:
                del self.working_memory[key]

        self.state.working_memory_count = len(self.working_memory)

    def _consolidate_to_long_term(self, pattern_id: str):
        if pattern_id in self.working_memory:
            item = self.working_memory[pattern_id]
            self.hippocampus.store_sequence(pattern_id)

            for synapse in self.hippocampus_recurrent_synapses:
                synapse.weight *= 1.1
                synapse.weight = min(synapse.w_max, synapse.weight)

            self.state.long_term_memory_count = len(self.hippocampus.memory_traces)

    def _attractor_dynamics(self, partial_pattern: List[int]) -> List[int]:
        if not partial_pattern:
            return []

        activation = np.zeros(500)
        for neuron_id in partial_pattern:
            if neuron_id < 500:
                activation[neuron_id] = 1.0

        for iteration in range(5):
            new_activation = np.copy(activation)

            for synapse in self.hippocampus_recurrent_synapses:
                if activation[synapse.pre_neuron] > 0.5:
                    new_activation[synapse.post_neuron] += synapse.weight * 0.5

            new_activation = np.clip(new_activation, 0, 1)
            activation = new_activation

        return [i for i in range(500) if activation[i] > 0.3]

    def _update_synapse_spike_times(self, layer_name: str, spikes: List[Any], current_t: float):
        if layer_name in self.perceptual_synapses:
            for synapse in self.perceptual_synapses[layer_name]:
                for spike in spikes:
                    spike_channel = spike.channel if hasattr(spike, 'channel') else spike
                    if spike_channel == synapse.pre_neuron:
                        synapse.last_pre_spike = current_t
                    if spike_channel == synapse.post_neuron:
                        synapse.last_post_spike = current_t

        for synapse in self.cognitive_synapses:
            for spike in spikes:
                spike_id = spike if isinstance(spike, int) else (spike.channel if hasattr(spike, 'channel') else 0)
                if spike_id == synapse.pre_neuron:
                    synapse.last_pre_spike = current_t
                if spike_id == synapse.post_neuron:
                    synapse.last_post_spike = current_t

        for synapse in self.hippocampus_recurrent_synapses:
            for spike in spikes:
                spike_channel = spike.channel if hasattr(spike, 'channel') else spike
                if spike_channel == synapse.pre_neuron:
                    synapse.last_pre_spike = current_t
                if spike_channel == synapse.post_neuron:
                    synapse.last_post_spike = current_t

    def process_audio(self, audio_signal: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()

        cochlea_spikes = self.cochlea.process(audio_signal)

        cn_spikes = []
        current_t = 0.0
        for spike in cochlea_spikes:
            t_ms = spike.timestamp * 1000
            while current_t < t_ms:
                current_t += self.dt

            self._apply_feedback_to_perceptual('cochlear_nucleus', current_t)
            layer_spikes = self.cochlear_nucleus.process([spike], current_t, self.dt)
            cn_spikes.extend(layer_spikes)

        ic_spikes = []
        current_t = 0.0
        for spike in cn_spikes:
            t_ms = getattr(spike, 'timestamp', 0) * 1000 if getattr(spike, 'timestamp', 0) > 0 else current_t
            while current_t < t_ms:
                current_t += self.dt

            self._apply_feedback_to_perceptual('inferior_colliculus', current_t)
            layer_spikes = self.inferior_colliculus.process([spike], [spike], current_t, self.dt)
            ic_spikes.extend(layer_spikes)

        cortex_spikes = []
        current_t = 0.0
        for spike in ic_spikes:
            t_ms = getattr(spike, 'timestamp', 0) * 1000 if getattr(spike, 'timestamp', 0) > 0 else current_t
            while current_t < t_ms:
                current_t += self.dt

            self._apply_feedback_to_perceptual('auditory_cortex', current_t)
            layer_spikes = self.auditory_cortex.process([spike], current_t, self.dt)
            cortex_spikes.extend(layer_spikes)

        self._update_ensemble_activity('auditory_cortex', cortex_spikes, current_t)

        hipp_spikes = []
        bg_spikes = []
        current_t = 0.0
        for spike in cortex_spikes:
            t_ms = getattr(spike, 'timestamp', 0) * 1000 if getattr(spike, 'timestamp', 0) > 0 else current_t
            while current_t < t_ms:
                current_t += self.dt

            h_spikes = self.hippocampus.process([spike], current_t, self.dt)
            b_spikes = self.basal_ganglia.process([spike], current_t, self.dt)
            hipp_spikes.extend(h_spikes)
            bg_spikes.extend(b_spikes)

            self._propagate_feedforward(spike, 'hippocampus', current_t)

        self._update_ensemble_activity('hippocampus', hipp_spikes, current_t)

        self._update_synapse_spike_times('hippocampus', hipp_spikes, current_t)

        cognitive_input = self._generate_cognitive_input(cochlea_spikes, cn_spikes, ic_spikes)

        for _ in range(3):
            self.cognitive_snn.update(self.dt, self.t, cognitive_input)
        cognitive_spikes = self.cognitive_snn.get_spikes()

        self._update_synapse_spike_times('cognitive', cognitive_spikes, current_t)

        reservoir_input = np.zeros(50)
        for i, nid in enumerate(cognitive_spikes[:50]):
            reservoir_input[i] = 1.0
        self.reservoir.update(reservoir_input, self.dt)

        self._process_delay_buffer(self.t)

        ff_count = self._propagate_all_feedforward(cortex_spikes, hipp_spikes, bg_spikes)
        fb_count = self._propagate_all_feedback(cognitive_spikes)

        if self.learning_enabled:
            self._apply_global_stdp(current_t)

            pattern_id = f"pattern_{int(current_t)}"
            cortex_pattern = [(s.channel, s.timestamp) for s in cortex_spikes[:20]]
            self._update_working_memory(pattern_id, cortex_pattern, current_t)

        fusion_output = self._fuse_outputs(hipp_spikes, bg_spikes, cognitive_spikes)

        perceptual_counts = {
            'cochlea': len(cochlea_spikes),
            'cochlear_nucleus': len(cn_spikes),
            'inferior_colliculus': len(ic_spikes),
            'auditory_cortex': len(cortex_spikes),
            'hippocampus': len(hipp_spikes),
            'basal_ganglia': len(bg_spikes)
        }

        self.state.perceptual_spikes = perceptual_counts
        self.state.cognitive_spikes = len(cognitive_spikes)
        self.state.total_spikes = sum(perceptual_counts.values()) + len(cognitive_spikes)
        self.state.active_connections = ff_count + fb_count
        self.state.information_flow = {
            'feedforward': ff_count,
            'feedback': fb_count,
            'fusion_neurons_active': int(np.sum(fusion_output > self.fusion_threshold))
        }
        self.state.fusion_activity = float(np.mean(fusion_output))
        self.state.long_term_memory_count = len(self.hippocampus.memory_traces)

        for spike in cortex_spikes:
            self.spike_history.append(('cortex', spike.channel, current_t))
        for nid in cognitive_spikes:
            self.spike_history.append(('cognitive', nid, self.t))

        self.t += self.dt
        self.total_processed += 1
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time

        return {
            'status': 'success',
            'processing_time_ms': processing_time * 1000,
            'perceptual_path': {
                'layers': perceptual_counts,
                'total_spikes': sum(perceptual_counts.values())
            },
            'cognitive_path': {
                'snn_spikes': len(cognitive_spikes),
                'reservoir_nodes': 50,
                'total_spikes': len(cognitive_spikes)
            },
            'cross_connections': {
                'total': len(self.cross_connections),
                'active': self.state.active_connections,
                'feedforward_count': ff_count,
                'feedback_count': fb_count
            },
            'fusion': {
                'fusion_neurons': self.fusion_neurons,
                'active_neurons': int(np.sum(fusion_output > self.fusion_threshold)),
                'mean_activity': float(np.mean(fusion_output)),
                'max_activity': float(np.max(fusion_output))
            },
            'memory': {
                'working_memory_items': self.state.working_memory_count,
                'long_term_memory_items': self.state.long_term_memory_count,
                'active_ensembles': self.state.ensemble_count
            },
            'stereoscopic_state': self._get_stereoscopic_summary()
        }

    def _generate_cognitive_input(self, cochlea_spikes, cn_spikes, ic_spikes) -> List[int]:
        input_neurons = set()
        input_size = self.cognitive_layer_sizes['input']

        for spike in cochlea_spikes[:30]:
            neuron_id = spike.channel % input_size
            input_neurons.add(neuron_id)

        for spike in cn_spikes[:25]:
            neuron_id = (spike.channel + 10) % input_size
            input_neurons.add(neuron_id)

        for spike in ic_spikes[:20]:
            neuron_id = (spike.channel + 20) % input_size
            input_neurons.add(neuron_id)

        return list(input_neurons)

    def _propagate_feedforward(self, spike, source_layer: str, current_t: float):
        for conn in self.cross_connections:
            if (conn.connection_type == ConnectionType.FEEDFORWARD and
                conn.source_layer == source_layer and
                conn.source_neuron == spike.channel):

                target_time = current_t + conn.delay
                self.delay_buffer.append((
                    target_time,
                    conn.target_neuron,
                    conn.weight,
                    'cognitive'
                ))

    def _propagate_all_feedforward(self, cortex_spikes, hipp_spikes, bg_spikes) -> int:
        ff_count = 0

        for spike in cortex_spikes:
            for conn in self.cross_connections:
                if (conn.connection_type == ConnectionType.FEEDFORWARD and
                    conn.source_layer == 'auditory_cortex'):

                    source_match = (conn.source_neuron == spike.channel % self.perceptual_layer_sizes['auditory_cortex'])
                    if source_match and np.random.random() < 0.4:
                        tgt_neuron = conn.target_neuron + self.cognitive_layer_boundaries[2]
                        if tgt_neuron < len(self.cognitive_snn._neurons):
                            self.cognitive_snn._neurons[tgt_neuron].I_ext += conn.weight * 8.0
                        ff_count += 1

        for spike in hipp_spikes[:100]:
            for conn in self.cross_connections:
                if (conn.connection_type == ConnectionType.FEEDFORWARD and
                    conn.source_layer == 'hippocampus' and
                    conn.source_neuron == spike.channel % self.perceptual_layer_sizes['hippocampus']):
                    if np.random.random() < 0.3:
                        tgt_neuron = conn.target_neuron + self.cognitive_layer_boundaries[3]
                        if tgt_neuron < len(self.cognitive_snn._neurons):
                            self.cognitive_snn._neurons[tgt_neuron].I_ext += conn.weight * 6.0
                        ff_count += 1

        return ff_count

    def _propagate_all_feedback(self, cognitive_spikes) -> int:
        fb_count = 0

        for spike_neuron in cognitive_spikes:
            for conn in self.cross_connections:
                if conn.connection_type == ConnectionType.FEEDBACK:
                    layer_idx = {'output': 3, 'hidden2': 2, 'hidden1': 1}.get(conn.source_layer, 0)
                    layer_start = self.cognitive_layer_boundaries[layer_idx]
                    layer_end = self.cognitive_layer_boundaries[layer_idx + 1]

                    if layer_start <= spike_neuron < layer_end:
                        local_neuron = spike_neuron - layer_start
                        if local_neuron == conn.source_neuron:
                            self._inject_feedback_to_perceptual(conn.target_layer, conn.target_neuron, conn.weight)
                            fb_count += 1

        return fb_count

    def _inject_feedback_to_perceptual(self, target_layer: str, target_neuron: int, weight: float):
        if target_layer == 'auditory_cortex':
            if hasattr(self.auditory_cortex, 'columns'):
                col_idx = target_neuron % len(self.auditory_cortex.columns)
                col = self.auditory_cortex.columns[col_idx]
                if hasattr(col, 'pyramidal_neurons') and col.pyramidal_neurons:
                    n = col.pyramidal_neurons[0]
                    if hasattr(n, 'I_ext'):
                        n.I_ext += weight * 3.0
        elif target_layer == 'hippocampus':
            if hasattr(self.hippocampus, 'neurons') and self.hippocampus.neurons:
                n_idx = target_neuron % len(self.hippocampus.neurons)
                n = self.hippocampus.neurons[n_idx]
                if hasattr(n, 'I_ext'):
                    n.I_ext += weight * 2.0
        elif target_layer == 'inferior_colliculus':
            if hasattr(self.inferior_colliculus, 'neurons') and self.inferior_colliculus.neurons:
                n_idx = target_neuron % len(self.inferior_colliculus.neurons)
                n = self.inferior_colliculus.neurons[n_idx]
                if hasattr(n, 'I_ext'):
                    n.I_ext += weight * 2.0
        elif target_layer == 'cochlear_nucleus':
            if hasattr(self.cochlear_nucleus, 'neurons') and self.cochlear_nucleus.neurons:
                n_idx = target_neuron % len(self.cochlear_nucleus.neurons)
                n = self.cochlear_nucleus.neurons[n_idx]
                if hasattr(n, 'I_ext'):
                    n.I_ext += weight * 2.0

    def _apply_feedback_to_perceptual(self, target_layer: str, current_t: float):
        feedback_strength = 0.0
        for conn in self.cross_connections:
            if (conn.connection_type == ConnectionType.FEEDBACK and
                conn.target_layer == target_layer):
                feedback_strength += conn.weight * 0.1
        return feedback_strength

    def _process_delay_buffer(self, current_t: float):
        still_pending = []

        for target_time, target_neuron, weight, target_path in self.delay_buffer:
            if target_time <= current_t:
                if target_path == 'cognitive' and target_neuron < len(self.cognitive_snn._neurons):
                    self.cognitive_snn._neurons[target_neuron].I_ext += weight * 12.0
            else:
                still_pending.append((target_time, target_neuron, weight, target_path))

        self.delay_buffer = still_pending

    def _fuse_outputs(self, hipp_spikes, bg_spikes, cognitive_spikes) -> np.ndarray:
        percep_output = np.zeros(
            self.perceptual_layer_sizes['hippocampus'] + self.perceptual_layer_sizes['basal_ganglia']
        )

        for spike in hipp_spikes:
            idx = spike.channel % self.perceptual_layer_sizes['hippocampus']
            percep_output[idx] = min(1.0, percep_output[idx] + 0.2)

        for i, spike in enumerate(bg_spikes):
            idx = self.perceptual_layer_sizes['hippocampus'] + (i % self.perceptual_layer_sizes['basal_ganglia'])
            if idx < len(percep_output):
                percep_output[idx] = min(1.0, percep_output[idx] + 0.2)

        cog_output = np.zeros(
            self.cognitive_layer_sizes['output'] + self.cognitive_layer_sizes['reservoir']
        )

        for neuron_id in cognitive_spikes:
            if neuron_id < self.cognitive_layer_sizes['output']:
                cog_output[neuron_id] = min(1.0, cog_output[neuron_id] + 0.3)

        if hasattr(self.reservoir, 'node_states'):
            reservoir_states = self.reservoir.node_states[:self.cognitive_layer_sizes['reservoir']]
            cog_output[self.cognitive_layer_sizes['output']:] = reservoir_states

        percep_contribution = percep_output @ self.percep_to_fusion
        cog_contribution = cog_output @ self.cog_to_fusion

        fusion_total = percep_contribution + cog_contribution + self.fusion_bias
        fusion_output = 1.0 / (1.0 + np.exp(-fusion_total * 5.0))

        return fusion_output

    def _get_stereoscopic_summary(self) -> Dict:
        ff_connections = sum(1 for c in self.cross_connections
                           if c.connection_type == ConnectionType.FEEDFORWARD)
        fb_connections = sum(1 for c in self.cross_connections
                           if c.connection_type == ConnectionType.FEEDBACK)

        return {
            'architecture': 'stereoscopic_dual_path_v3',
            'description': '双通道立体脉冲神经网络 v3.0 - 基于人脑知识存储机制',
            'total_neurons': self._get_total_neurons(),
            'perceptual_path': {
                'name': '生物启发式听觉通路',
                'layers': 6,
                'total_neurons': sum(self.perceptual_layer_sizes.values()),
                'layer_sizes': self.perceptual_layer_sizes
            },
            'cognitive_path': {
                'name': '增强版SNN + 混沌储备池',
                'snn_layers': 4,
                'snn_neurons': sum(self.cognitive_layer_sizes.values()) - self.cognitive_layer_sizes['reservoir'],
                'reservoir_nodes': self.cognitive_layer_sizes['reservoir'],
                'layer_sizes': self.cognitive_layer_sizes
            },
            'cross_connections': {
                'total': len(self.cross_connections),
                'feedforward': ff_connections,
                'feedback': fb_connections,
                'sparsity': self.sparsity,
                'description': '稀疏双向交叉连接（含全网络STDP可塑性）'
            },
            'hippocampal_attractor': {
                'recurrent_connections': len(self.hippocampus_recurrent_synapses),
                'description': 'CA3循环连接实现模式补全与联想记忆'
            },
            'memory_system': {
                'working_memory_capacity': self.working_memory_capacity,
                'consolidation_threshold': self.consolidation_threshold,
                'description': '双记忆系统：短时动态维持 + 长时结构固化'
            },
            'fusion_layer': {
                'neurons': self.fusion_neurons,
                'description': '多模态融合输出层'
            },
            'design_principles': [
                '双通道立体架构',
                '全网络STDP可塑性（毫秒级LTP/LTD）',
                '神经元集群时空编码',
                '双记忆系统（短时/长时）',
                '海马CA3吸引子动力学',
                '层级抽象特征加工',
                '双向信息流（前馈+反馈）',
                '多尺度特征融合'
            ],
            'biological_mapping': {
                'level_1': '底层存储单元：突触可塑性（STDP）',
                'level_2': '知识编码形式：神经元集群时空脉冲模式',
                'level_3': '记忆巩固：短时回响活动 vs 长时权重固化',
                'level_4': '知识提取：吸引子动力学与模式补全',
                'level_5': '层级抽象：多层网络对应皮层层级加工'
            }
        }

    def get_network_state(self) -> Dict:
        return {
            'stereoscopic': self._get_stereoscopic_summary(),
            'state': {
                'total_spikes': self.state.total_spikes,
                'perceptual_spikes': sum(self.state.perceptual_spikes.values()),
                'cognitive_spikes': self.state.cognitive_spikes,
                'cross_connection_active': self.state.active_connections,
                'information_flow': self.state.information_flow,
                'fusion_activity': self.state.fusion_activity,
                'working_memory_count': self.state.working_memory_count,
                'long_term_memory_count': self.state.long_term_memory_count,
                'ensemble_count': self.state.ensemble_count
            },
            'current_time_ms': self.t,
            'learning_enabled': self.learning_enabled,
            'memory_count': len(self.hippocampus.memory_traces) if hasattr(self.hippocampus, 'memory_traces') else 0,
            'rhythm_patterns': len(self.rhythm_patterns),
            'total_processed': self.total_processed,
            'avg_processing_time_ms': (self.total_processing_time / max(1, self.total_processed)) * 1000,
            'hippocampal_recurrent_connections': len(self.hippocampus_recurrent_synapses)
        }

    def learn_melody(self, melody_id: str, audio_signal: np.ndarray):
        self.process_audio(audio_signal)
        self.hippocampus.store_sequence(melody_id)
        seq = self.hippocampus.retrieve_sequence(melody_id)
        if seq:
            rhythm_pattern = [s.timestamp for s in seq]
            self.basal_ganglia.learn_rhythm(rhythm_pattern)
            self.rhythm_patterns[melody_id] = rhythm_pattern

    def recall_melody(self, partial_audio: np.ndarray) -> List:
        self.process_audio(partial_audio)
        recalled = self.hippocampus.retrieve_sequence('last')

        if not recalled and self.rhythm_patterns:
            for pattern_id, pattern in self.rhythm_patterns.items():
                recalled = self.hippocampus.retrieve_sequence(pattern_id)
                if recalled:
                    break

        if not recalled:
            cortex_spikes = []
            try:
                results = self.process_audio(partial_audio)
                cortex_spikes = [s for s in results.get('perceptual_path', {}).get('layers', {}).keys()]
            except:
                pass

            if cortex_spikes:
                partial_pattern = [s % 500 for s in cortex_spikes[:10]]
                completed_pattern = self._attractor_dynamics(partial_pattern)
                if completed_pattern:
                    recalled = []

        return recalled if recalled else []

    def predict_rhythm(self, current_sequence: List[float]) -> Optional[float]:
        if len(current_sequence) < 2:
            return None

        intervals = [current_sequence[i+1] - current_sequence[i]
                     for i in range(len(current_sequence)-1)]

        if not intervals:
            return None

        next_time = self.basal_ganglia.predict_next(current_sequence)

        if next_time is None:
            avg_interval = np.mean(intervals)
            next_time = current_sequence[-1] + avg_interval

        return next_time

    def clear_memory(self):
        self.hippocampus.memory_traces = {}
        self.hippocampus.current_sequence = []
        self.basal_ganglia.rhythm_patterns = {}
        self.basal_ganglia.prediction_buffer = []
        self.rhythm_patterns = {}
        self.spike_history.clear()
        self.delay_buffer = []
        self.fusion_state = np.zeros(self.fusion_neurons)
        self.working_memory = {}
        self.state.working_memory_count = 0
        self.state.long_term_memory_count = 0

        for synapse in self.hippocampus_recurrent_synapses:
            synapse.weight = np.random.uniform(0.1, 0.35)

        print("[OK] 立体SNN记忆已清除")

    def get_ensemble_activity(self) -> Dict:
        result = {}
        for ensemble_id, ensemble in self.neural_ensembles.items():
            result[ensemble_id] = {
                'neuron_count': len(ensemble.neurons),
                'coherence_score': ensemble.coherence_score,
                'last_activation': ensemble.last_activation_time,
                'stability': ensemble.stability,
                'pattern_length': len(ensemble.activation_pattern)
            }
        return result

    def get_synaptic_statistics(self) -> Dict:
        stats = {
            'perceptual_synapses': {},
            'cognitive_synapses': {},
            'hippocampal_recurrent': {},
            'cross_connections': {}
        }

        for layer_name, synapses in self.perceptual_synapses.items():
            if synapses:
                weights = [s.weight for s in synapses]
                ltp_counts = sum(s.ltp_count for s in synapses)
                ltd_counts = sum(s.ltd_count for s in synapses)
                stats['perceptual_synapses'][layer_name] = {
                    'count': len(synapses),
                    'avg_weight': np.mean(weights),
                    'min_weight': np.min(weights),
                    'max_weight': np.max(weights),
                    'ltp_count': ltp_counts,
                    'ltd_count': ltd_counts,
                    'ltp_ratio': ltp_counts / max(1, ltp_counts + ltd_counts)
                }

        if self.cognitive_synapses:
            weights = [s.weight for s in self.cognitive_synapses]
            ltp_counts = sum(s.ltp_count for s in self.cognitive_synapses)
            ltd_counts = sum(s.ltd_count for s in self.cognitive_synapses)
            stats['cognitive_synapses'] = {
                'count': len(self.cognitive_synapses),
                'avg_weight': np.mean(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights),
                'ltp_count': ltp_counts,
                'ltd_count': ltd_counts,
                'ltp_ratio': ltp_counts / max(1, ltp_counts + ltd_counts)
            }

        if self.hippocampus_recurrent_synapses:
            weights = [s.weight for s in self.hippocampus_recurrent_synapses]
            ltp_counts = sum(s.ltp_count for s in self.hippocampus_recurrent_synapses)
            ltd_counts = sum(s.ltd_count for s in self.hippocampus_recurrent_synapses)
            stats['hippocampal_recurrent'] = {
                'count': len(self.hippocampus_recurrent_synapses),
                'avg_weight': np.mean(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights),
                'ltp_count': ltp_counts,
                'ltd_count': ltd_counts,
                'ltp_ratio': ltp_counts / max(1, ltp_counts + ltd_counts)
            }

        if self.cross_connections:
            weights = [c.weight for c in self.cross_connections]
            stats['cross_connections'] = {
                'count': len(self.cross_connections),
                'avg_weight': np.mean(weights),
                'min_weight': np.min(weights),
                'max_weight': np.max(weights)
            }

        return stats


if __name__ == "__main__":
    print("=" * 60)
    print("立体脉冲神经网络（Stereoscopic SNN）v3.0 测试")
    print("=" * 60)

    print("\n初始化立体SNN v3.0...")
    stereo_snn = StereoscopicSNN({
        'num_channels': 128,
        'sample_rate': 44100,
        'sparsity': 0.15
    })

    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 440 * t) * 0.5

    print(f"\n测试音频: {duration}秒, 440Hz正弦波")

    print("\n开始立体SNN处理...")
    results = stereo_snn.process_audio(test_signal)

    print(f"\n处理完成！耗时: {results['processing_time_ms']:.2f}ms")

    print("\n=== 感知通路输出 ===")
    for layer, count in results['perceptual_path']['layers'].items():
        print(f"  {layer}: {count} 脉冲")
    print(f"  总计: {results['perceptual_path']['total_spikes']} 脉冲")

    print("\n=== 认知通路输出 ===")
    print(f"  SNN脉冲: {results['cognitive_path']['snn_spikes']}")
    print(f"  总计: {results['cognitive_path']['total_spikes']} 脉冲")

    print("\n=== 交叉连接 ===")
    cc = results['cross_connections']
    print(f"  总连接数: {cc['total']}")
    print(f"  活跃连接: {cc['active']}")
    print(f"  前馈传递: {cc['feedforward_count']} 次")
    print(f"  反馈传递: {cc['feedback_count']} 次")

    print("\n=== 融合层输出 ===")
    fusion = results['fusion']
    print(f"  融合神经元: {fusion['fusion_neurons']}")
    print(f"  活跃神经元: {fusion['active_neurons']}")
    print(f"  平均活动度: {fusion['mean_activity']:.4f}")
    print(f"  最大活动度: {fusion['max_activity']:.4f}")

    print("\n=== 记忆系统 ===")
    memory = results['memory']
    print(f"  短时工作记忆: {memory['working_memory_items']} 项")
    print(f"  长时记忆: {memory['long_term_memory_items']} 项")
    print(f"  活跃神经元集群: {memory['active_ensembles']}")

    print("\n=== 立体网络架构 ===")
    arch = results['stereoscopic_state']
    print(f"  架构类型: {arch['architecture']}")
    print(f"  总神经元数: {arch['total_neurons']}")
    print(f"  感知通路: {arch['perceptual_path']['name']} ({arch['perceptual_path']['total_neurons']}神经元)")
    print(f"  认知通路: {arch['cognitive_path']['name']} ({arch['cognitive_path']['snn_neurons']}+{arch['cognitive_path']['reservoir_nodes']}节点)")
    print(f"  海马循环连接: {arch['hippocampal_attractor']['recurrent_connections']}")

    print("\n=== 测试旋律学习 ===")
    stereo_snn.learn_melody("test_melody", test_signal)
    state = stereo_snn.get_network_state()
    print(f"  记忆数: {state['memory_count']}")
    print(f"  节奏模式: {state['rhythm_patterns']}")
    print(f"  长时记忆项数: {state['state']['long_term_memory_count']}")

    print("\n=== 测试旋律回忆 ===")
    recalled = stereo_snn.recall_melody(test_signal[:len(test_signal)//3])
    print(f"  回忆脉冲数: {len(recalled)}")

    print("\n=== 测试节奏预测 ===")
    test_seq = [0.1, 0.3, 0.5, 0.7, 0.9]
    next_t = stereo_snn.predict_rhythm(test_seq)
    print(f"  输入序列: {test_seq}")
    print(f"  预测下一个事件: {next_t:.3f}" if next_t else "  无法预测")

    print("\n=== 测试吸引子动力学 ===")
    partial = [10, 20, 30]
    completed = stereo_snn._attractor_dynamics(partial)
    print(f"  部分模式: {partial}")
    print(f"  补全模式: {len(completed)} 个神经元激活")

    print("\n=== 测试突触统计 ===")
    syn_stats = stereo_snn.get_synaptic_statistics()
    print(f"  感知通路突触: {sum(len(v) for v in syn_stats['perceptual_synapses'].values())}")
    print(f"  认知通路突触: {syn_stats['cognitive_synapses'].get('count', 0)}")
    print(f"  海马循环突触: {syn_stats['hippocampal_recurrent'].get('count', 0)}")

    print("\n=== 测试神经元集群 ===")
    ensemble_stats = stereo_snn.get_ensemble_activity()
    print(f"  集群数量: {len(ensemble_stats)}")
    active_ensembles = sum(1 for e in ensemble_stats.values() if e['coherence_score'] > 0)
    print(f"  活跃集群: {active_ensembles}")

    print("\n=== 测试清除记忆 ===")
    stereo_snn.clear_memory()
    state = stereo_snn.get_network_state()
    print(f"  清除后记忆数: {state['memory_count']}")
    print(f"  清除后工作记忆: {state['state']['working_memory_count']}")

    print("\n" + "=" * 60)
    print("立体SNN v3.0 测试完成！")
    print("=" * 60)