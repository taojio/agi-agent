#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版脉冲神经网络(SNN)模块

包含：
1. 高性能SNN实现 - 优化学习效率、计算精度和实时处理能力
2. Dendristor树突脉冲计算网络 - 基于树突计算特性的神经元模型
3. 自组织生长混沌储备池 - 动态演化的储备池计算架构
4. 音乐音频处理专用优化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import time
import math


class NeuronType(Enum):
    """神经元类型"""
    LIF = 'lif'
    IZH = 'izhikevich'
    SRM = 'srM'
    DENDRISTOR = 'dendristor'


class SynapseType(Enum):
    """突触类型"""
    STDP = 'stdp'
    STP = 'stp'
    STDP_STP = 'stdp_stp'
    FIXED = 'fixed'


@dataclass
class Synapse:
    """突触模型"""
    pre_neuron: int
    post_neuron: int
    weight: float
    type: SynapseType = SynapseType.STDP
    delay: float = 1.0
    last_update: float = 0.0
    
    # STDP参数
    A_plus: float = 0.01
    A_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    
    # STP参数
    U: float = 0.5
    tau_f: float = 100.0
    tau_d: float = 500.0
    x: float = 1.0
    u: float = 0.0


@dataclass
class DendriteSegment:
    """树突段模型"""
    id: str
    parent_neuron: int
    position: float
    synapses: List[int] = field(default_factory=list)
    voltage: float = 0.0
    threshold: float = -55.0
    active: bool = False
    spike_time: float = 0.0
    
    # Dendristor特性参数
    memristance: float = 1e6
    conductance: float = 1e-6
    spike_amplitude: float = 0.5
    refractory_period: float = 2.0


@dataclass
class ChaosReservoirNode:
    """混沌储备池节点"""
    id: str
    neurons: List[int] = field(default_factory=list)
    connections: List[Tuple[int, float]] = field(default_factory=list)
    activation: float = 0.0
    chaos_parameter: float = 0.5
    growth_rate: float = 0.1
    pruned: bool = False


@dataclass
class SNNStats:
    """SNN统计信息"""
    total_spikes: int = 0
    active_neurons: int = 0
    synapse_updates: int = 0
    computation_time: float = 0.0
    learning_rate: float = 0.001


class OptimizedLIFNeuron:
    """优化的LIF神经元"""
    
    def __init__(self, neuron_id: int, params: Dict = None):
        self.id = neuron_id
        self.params = params or {}
        
        # 膜电位参数
        self.V_m = self.params.get('V_rest', -70.0)
        self.V_rest = self.params.get('V_rest', -70.0)
        self.V_th = self.params.get('V_th', -55.0)
        self.V_reset = self.params.get('V_reset', -70.0)
        
        # 时间常数
        self.tau_m = self.params.get('tau_m', 10.0)
        self.tau_ref = self.params.get('tau_ref', 2.0)
        
        # 电导
        self.g_L = self.params.get('g_L', 1.0)
        
        # 状态
        self.spiked = False
        self.last_spike_time = -float('inf')
        self.refractory = False
        
        # 输入电流
        self.I_syn = 0.0
        self.I_ext = 0.0
        
        # 优化参数
        self.dt_factor = 1.0
        self.spike_counter = 0
    
    def update(self, dt: float, t: float):
        """更新神经元状态（优化版本）"""
        if t - self.last_spike_time < self.tau_ref:
            self.refractory = True
            self.V_m = self.V_reset
            return
        
        self.refractory = False
        
        dV_dt = (self.g_L * (self.V_rest - self.V_m) + self.I_syn + self.I_ext) / self.tau_m
        self.V_m += dV_dt * dt
        
        self.spiked = False
        if self.V_m >= self.V_th:
            self.spiked = True
            self.V_m = self.V_reset
            self.last_spike_time = t
            self.spike_counter += 1
        
        self.I_syn = 0.0
    
    def add_synaptic_input(self, weight: float, delay: float = 0.0):
        """添加突触输入"""
        self.I_syn += weight


class DendristorNeuron:
    """Dendristor树突计算神经元"""
    
    def __init__(self, neuron_id: int, params: Dict = None):
        self.id = neuron_id
        self.params = params or {}
        
        # 体细胞参数
        self.V_soma = self.params.get('V_rest', -70.0)
        self.V_rest = self.params.get('V_rest', -70.0)
        self.V_th = self.params.get('V_th', -55.0)
        self.V_reset = self.params.get('V_reset', -70.0)
        
        # 时间常数
        self.tau_m = self.params.get('tau_m', 10.0)
        self.tau_ref = self.params.get('tau_ref', 2.0)
        
        # 树突段
        self.dendrite_segments: Dict[str, DendriteSegment] = {}
        self._create_dendrite_segments()
        
        # 状态
        self.spiked = False
        self.last_spike_time = -float('inf')
        self.refractory = False
        
        # 输入电流
        self.I_syn = 0.0
        self.I_ext = 0.0
        
        # Dendristor特性
        self.spike_counter = 0
        self.dendritic_spikes = 0
    
    def _create_dendrite_segments(self):
        """创建树突段"""
        positions = [0.2, 0.4, 0.6, 0.8]
        for i, pos in enumerate(positions):
            segment = DendriteSegment(
                id=f"seg_{i}",
                parent_neuron=self.id,
                position=pos,
                threshold=self.V_th + (i * 5),
                memristance=1e6 - i * 1e5
            )
            self.dendrite_segments[segment.id] = segment
    
    def update(self, dt: float, t: float):
        """更新神经元状态（包含树突计算）"""
        self._update_dendrite_segments(dt, t)
        
        if t - self.last_spike_time < self.tau_ref:
            self.refractory = True
            self.V_soma = self.V_reset
            return
        
        self.refractory = False
        
        dendritic_input = self._integrate_dendritic_input()
        self.I_syn += dendritic_input
        
        dV_dt = (self.V_rest - self.V_soma + self.I_syn + self.I_ext) / self.tau_m
        self.V_soma += dV_dt * dt
        
        self.spiked = False
        if self.V_soma >= self.V_th:
            self.spiked = True
            self.V_soma = self.V_reset
            self.last_spike_time = t
            self.spike_counter += 1
            self._propagate_backpropagating_spike(t)
        
        self.I_syn = 0.0
    
    def _update_dendrite_segments(self, dt: float, t: float):
        """更新树突段状态"""
        for segment in self.dendrite_segments.values():
            if t - segment.spike_time < segment.refractory_period:
                continue
            
            dV_dt = (-segment.voltage + segment.spike_amplitude * self.I_syn * segment.conductance)
            segment.voltage += dV_dt * dt
            
            if segment.voltage >= segment.threshold:
                segment.active = True
                segment.spike_time = t
                segment.voltage = 0.0
                self.dendritic_spikes += 1
                
                segment.memristance *= 0.99
                segment.conductance = 1.0 / segment.memristance
            else:
                segment.active = False
    
    def _integrate_dendritic_input(self) -> float:
        """整合树突段输入"""
        input_sum = 0.0
        for segment in self.dendrite_segments.values():
            if segment.active:
                input_sum += segment.spike_amplitude * segment.position
        
        return input_sum
    
    def _propagate_backpropagating_spike(self, t: float):
        """反向传播尖峰（影响树突段）"""
        for segment in self.dendrite_segments.values():
            attenuation = 1.0 - segment.position
            segment.voltage += 20.0 * attenuation
            segment.spike_time = t


class EnhancedSNN:
    """增强版脉冲神经网络"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 网络参数
        self.num_neurons = self.config.get('num_neurons', 100)
        self.num_layers = self.config.get('num_layers', 4)
        self.neurons_per_layer = self.config.get('neurons_per_layer', [25, 25, 25, 25])
        
        # 神经元类型配置
        self.neuron_type = self.config.get('neuron_type', NeuronType.LIF)
        
        # 神经元集合
        self.neurons: List[OptimizedLIFNeuron] = []
        self.dendristor_neurons: List[DendristorNeuron] = []
        
        # 突触集合
        self.synapses: List[Synapse] = []
        
        # 层结构
        self.layer_boundaries = []
        
        # 统计信息
        self.stats = SNNStats()
        
        # 学习率调度
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.lr_decay = self.config.get('lr_decay', 0.99)
        
        # 初始化网络
        self._initialize_network()
    
    def _initialize_network(self):
        """初始化网络结构"""
        # 创建层边界
        self.layer_boundaries = [0]
        current = 0
        for count in self.neurons_per_layer:
            current += count
            self.layer_boundaries.append(current)
        
        # 创建神经元并缓存神经元引用（避免运行时判断）
        self._neurons = []
        for i in range(self.num_neurons):
            layer_idx = self._get_layer_index(i)
            
            if self.neuron_type == NeuronType.DENDRISTOR:
                neuron = DendristorNeuron(i, {
                    'V_th': -55.0 + layer_idx * 2,
                    'tau_m': 10.0 - layer_idx * 1.5
                })
                self._neurons.append(neuron)
            else:
                neuron = OptimizedLIFNeuron(i, {
                    'V_th': -55.0 + layer_idx * 2,
                    'tau_m': 10.0 - layer_idx * 1.5
                })
                self._neurons.append(neuron)
        
        # 创建突触连接
        self._create_synaptic_connections()
    
    def _get_layer_index(self, neuron_id: int) -> int:
        """获取神经元所在层索引"""
        for i, boundary in enumerate(self.layer_boundaries[1:]):
            if neuron_id < boundary:
                return i
        return len(self.neurons_per_layer) - 1
    
    def _create_synaptic_connections(self):
        """创建突触连接（优化的稀疏连接）"""
        for layer_idx in range(self.num_layers - 1):
            start_pre = self.layer_boundaries[layer_idx]
            end_pre = self.layer_boundaries[layer_idx + 1]
            start_post = self.layer_boundaries[layer_idx + 1]
            end_post = self.layer_boundaries[layer_idx + 2] if layer_idx + 2 < len(self.layer_boundaries) else self.num_neurons
            
            # 稀疏连接：每层连接率50%
            for pre in range(start_pre, end_pre):
                num_post = int((end_post - start_post) * 0.5)
                post_ids = np.random.choice(range(start_post, end_post), num_post, replace=False)
                
                for post in post_ids:
                    synapse = Synapse(
                        pre_neuron=pre,
                        post_neuron=post,
                        weight=np.random.uniform(-0.5, 0.5),
                        type=SynapseType.STDP_STP
                    )
                    self.synapses.append(synapse)
    
    def update(self, dt: float, t: float, input_spikes: List[int] = None):
        """更新网络状态（优化的批量更新）"""
        start_time = time.time()
        
        # 重置尖峰状态
        for neuron in self._neurons:
            neuron.spiked = False
        
        # 处理输入尖峰
        if input_spikes:
            for neuron_id in input_spikes:
                if neuron_id < len(self._neurons):
                    self._neurons[neuron_id].I_ext += 10.0
        
        # 更新所有神经元（批量处理）
        for neuron in self._neurons:
            neuron.update(dt, t)
        
        # STDP学习（向量化优化）
        self._stdp_learning(t)
        
        # 更新统计
        self.stats.total_spikes += sum(1 for n in self._neurons if n.spiked)
        self.stats.active_neurons = sum(1 for n in self._neurons if not n.refractory)
        self.stats.computation_time += time.time() - start_time
    
    def _stdp_learning(self, t: float):
        """STDP学习规则（向量化优化版本）"""
        neurons = self._neurons
        num_neurons = len(neurons)
        
        # 批量获取尖峰状态和上次发放时间（向量化）
        spiked_status = np.array([n.spiked for n in neurons], dtype=bool)
        last_spike_times = np.array([n.last_spike_time for n in neurons])
        
        for synapse in self.synapses:
            pre_idx = synapse.pre_neuron
            post_idx = synapse.post_neuron
            
            # 跳过超出范围的突触
            if pre_idx >= num_neurons or post_idx >= num_neurons:
                continue
            
            pre_spiked = spiked_status[pre_idx]
            post_spiked = spiked_status[post_idx]
            
            if pre_spiked and post_spiked:
                delta_t = last_spike_times[post_idx] - last_spike_times[pre_idx]
                
                if delta_t > 0:
                    synapse.weight += synapse.A_plus * math.exp(-delta_t / synapse.tau_plus)
                else:
                    synapse.weight -= synapse.A_minus * math.exp(delta_t / synapse.tau_minus)
                
                synapse.weight = max(-1.0, min(1.0, synapse.weight))
                self.stats.synapse_updates += 1
    
    def get_spikes(self) -> List[int]:
        """获取当前时刻发放尖峰的神经元ID"""
        return [i for i, neuron in enumerate(self._neurons) if neuron.spiked]
    
    def get_network_state(self) -> Dict:
        """获取网络状态"""
        return {
            'neuron_count': len(self._neurons),
            'synapse_count': len(self.synapses),
            'total_spikes': self.stats.total_spikes,
            'active_neurons': self.stats.active_neurons,
            'mean_weight': np.mean([s.weight for s in self.synapses]) if self.synapses else 0.0,
            'learning_rate': self.learning_rate
        }


class SelfOrganizingChaosReservoir:
    """自组织生长混沌储备池"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 储备池参数
        self.num_nodes = self.config.get('num_nodes', 20)
        self.initial_neurons_per_node = self.config.get('initial_neurons_per_node', 5)
        
        # 混沌参数
        self.chaos_factor = self.config.get('chaos_factor', 0.5)
        self.growth_threshold = self.config.get('growth_threshold', 0.8)
        self.prune_threshold = self.config.get('prune_threshold', 0.1)
        
        # 生长参数
        self.max_nodes = self.config.get('max_nodes', 50)
        self.max_neurons_per_node = self.config.get('max_neurons_per_node', 10)
        
        # 储备池节点
        self.nodes: List[ChaosReservoirNode] = []
        
        # 全局状态
        self.time_step = 0
        self.total_neurons = 0
        self.stats = {'nodes_created': 0, 'nodes_pruned': 0, 'neurons_added': 0}
        
        # 初始化储备池
        self._initialize_reservoir()
    
    def _initialize_reservoir(self):
        """初始化储备池"""
        for i in range(self.num_nodes):
            node = ChaosReservoirNode(
                id=f"node_{i}",
                chaos_parameter=self.chaos_factor + np.random.uniform(-0.1, 0.1),
                growth_rate=np.random.uniform(0.05, 0.15)
            )
            
            # 为每个节点创建初始神经元
            for _ in range(self.initial_neurons_per_node):
                node.neurons.append(self.total_neurons)
                self.total_neurons += 1
            
            self.nodes.append(node)
            self.stats['nodes_created'] += 1
        
        # 创建初始连接
        self._create_initial_connections()
    
    def _create_initial_connections(self):
        """创建初始连接"""
        for i, node in enumerate(self.nodes):
            # 连接到后续节点
            for j in range(i + 1, min(i + 3, len(self.nodes))):
                weight = np.random.uniform(0.1, 0.5)
                node.connections.append((j, weight))
    
    def update(self, inputs: np.ndarray, dt: float):
        """更新储备池状态（包含自组织生长）"""
        self.time_step += 1
        
        # 更新每个节点的激活
        for node in self.nodes:
            if node.pruned:
                continue
            
            # 计算节点激活（混沌动力学）
            node.activation = self._compute_chaos_activation(node, inputs)
            
            # 自组织生长规则
            self._apply_growth_rules(node)
        
        # 修剪无效节点
        self._prune_inactive_nodes()
        
        # 创建新节点
        self._create_new_nodes()
    
    def _compute_chaos_activation(self, node: ChaosReservoirNode, inputs: np.ndarray) -> float:
        """计算混沌激活"""
        # 基础激活来自输入
        input_activation = np.mean(inputs) if len(inputs) > 0 else 0.0
        
        # 混沌动力学（简化的Lorenz吸引子模型）
        prev_activation = node.activation
        chaos_term = node.chaos_parameter * (1 - prev_activation) * prev_activation
        
        # 来自其他节点的输入
        connection_input = 0.0
        for target_idx, weight in node.connections:
            if target_idx < len(self.nodes) and not self.nodes[target_idx].pruned:
                connection_input += self.nodes[target_idx].activation * weight
        
        # 综合激活
        activation = 0.3 * input_activation + 0.5 * chaos_term + 0.2 * connection_input
        activation = max(0.0, min(1.0, activation))
        
        return activation
    
    def _apply_growth_rules(self, node: ChaosReservoirNode):
        """应用生长规则"""
        # 如果激活度高，增加神经元数量
        if node.activation > self.growth_threshold:
            if len(node.neurons) < self.max_neurons_per_node:
                node.neurons.append(self.total_neurons)
                self.total_neurons += 1
                self.stats['neurons_added'] += 1
                
                # 增加连接强度
                for i, (target_idx, weight) in enumerate(node.connections):
                    node.connections[i] = (target_idx, min(1.0, weight + 0.05))
        
        # 如果激活度低，减弱连接
        elif node.activation < self.prune_threshold:
            for i, (target_idx, weight) in enumerate(node.connections):
                node.connections[i] = (target_idx, max(0.01, weight - 0.05))
    
    def _prune_inactive_nodes(self):
        """修剪不活跃节点"""
        for node in self.nodes:
            if not node.pruned and node.activation < self.prune_threshold / 2:
                node.pruned = True
                self.stats['nodes_pruned'] += 1
    
    def _create_new_nodes(self):
        """创建新节点（如果需要）"""
        active_nodes = [n for n in self.nodes if not n.pruned]
        
        # 如果活跃节点数量不足，创建新节点
        if len(active_nodes) < self.num_nodes and len(self.nodes) < self.max_nodes:
            new_node = ChaosReservoirNode(
                id=f"node_{len(self.nodes)}",
                chaos_parameter=self.chaos_factor + np.random.uniform(-0.15, 0.15),
                growth_rate=np.random.uniform(0.05, 0.15)
            )
            
            # 为新节点创建初始神经元
            for _ in range(self.initial_neurons_per_node):
                new_node.neurons.append(self.total_neurons)
                self.total_neurons += 1
            
            # 连接到现有活跃节点
            if active_nodes:
                connections = np.random.choice(len(active_nodes), min(3, len(active_nodes)), replace=False)
                for idx in connections:
                    target_id = int(active_nodes[idx].id.split('_')[1])
                    new_node.connections.append((target_id, np.random.uniform(0.1, 0.4)))
            
            self.nodes.append(new_node)
            self.stats['nodes_created'] += 1
    
    def get_reservoir_state(self) -> Dict:
        """获取储备池状态"""
        active_nodes = [n for n in self.nodes if not n.pruned]
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'total_neurons': self.total_neurons,
            'mean_activation': np.mean([n.activation for n in active_nodes]) if active_nodes else 0.0,
            'stats': self.stats
        }
    
    def get_output(self) -> np.ndarray:
        """获取储备池输出"""
        active_nodes = [n for n in self.nodes if not n.pruned]
        if not active_nodes:
            return np.array([])
        
        return np.array([n.activation for n in active_nodes])


class MusicSNNProcessor:
    """音乐音频专用SNN处理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # SNN核心
        self.snn = EnhancedSNN(self.config.get('snn_config', {}))
        
        # 混沌储备池
        self.reservoir = SelfOrganizingChaosReservoir(self.config.get('reservoir_config', {}))
        
        # 处理状态
        self.t = 0.0
        self.dt = 1.0
        self.processed_frames = 0
        
        # 特征缓冲区
        self.feature_buffer = deque(maxlen=100)
        
        # 输出指标
        self.loss_history = []
        self.accuracy_history = []
    
    def process_audio_features(self, features: np.ndarray):
        """处理音频特征"""
        # 添加到缓冲区
        self.feature_buffer.append(features)
        
        # 计算输入尖峰（基于特征强度）
        input_spikes = self._generate_spikes_from_features(features)
        
        # 更新SNN
        self.snn.update(self.dt, self.t, input_spikes)
        
        # 更新混沌储备池
        reservoir_input = self._prepare_reservoir_input(features)
        self.reservoir.update(reservoir_input, self.dt)
        
        # 获取输出
        spikes = self.snn.get_spikes()
        
        # 更新时间
        self.t += self.dt
        self.processed_frames += 1
        
        return spikes
    
    def _generate_spikes_from_features(self, features: np.ndarray) -> List[int]:
        """从特征生成尖峰"""
        spikes = []
        
        # 假设特征已归一化到[0, 1]
        for i, value in enumerate(features.flat[:min(len(self.snn.neurons), len(features.flat))]):
            if value > 0.5 + np.random.uniform(-0.1, 0.1):
                spikes.append(i)
        
        return spikes
    
    def _prepare_reservoir_input(self, features: np.ndarray) -> np.ndarray:
        """准备储备池输入"""
        # 提取关键特征维度
        if len(features) >= 3:
            return features[:3]
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """训练SNN"""
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            
            for features, target in zip(X, y):
                spikes = self.process_audio_features(features)
                
                # 简单的监督学习：比较尖峰数量
                target_spike_count = int(target * 10)
                actual_spike_count = len(spikes)
                
                epoch_loss += abs(target_spike_count - actual_spike_count)
                if abs(target_spike_count - actual_spike_count) <= 1:
                    correct += 1
            
            # 更新统计
            self.loss_history.append(epoch_loss / len(X))
            accuracy = correct / len(X)
            self.accuracy_history.append(accuracy)
            
            # 学习率衰减
            self.snn.learning_rate *= self.snn.lr_decay
    
    def get_processor_state(self) -> Dict:
        """获取处理器状态"""
        return {
            'snn_state': self.snn.get_network_state(),
            'reservoir_state': self.reservoir.get_reservoir_state(),
            'time': self.t,
            'processed_frames': self.processed_frames,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }


if __name__ == '__main__':
    config = {
        'num_neurons': 100,
        'num_layers': 4,
        'neurons_per_layer': [30, 25, 25, 20],
        'neuron_type': NeuronType.LIF,
        'learning_rate': 0.001
    }
    
    snn = EnhancedSNN(config)
    print("增强版SNN初始化完成")
    print(f"神经元数量: {snn.num_neurons}")
    print(f"突触数量: {len(snn.synapses)}")
    
    reservoir = SelfOrganizingChaosReservoir({
        'num_nodes': 15,
        'chaos_factor': 0.6
    })
    print(f"\n混沌储备池初始化完成")
    print(f"节点数量: {len(reservoir.nodes)}")
    
    processor = MusicSNNProcessor({
        'snn_config': config,
        'reservoir_config': {'num_nodes': 20}
    })
    print(f"\n音乐SNN处理器初始化完成")