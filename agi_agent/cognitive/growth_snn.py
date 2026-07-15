"""
growth_snn.py - 生长型脉冲神经网络

实现具有结构可塑性的脉冲神经网络，其中神经元和突触都具有生长能力，
但网络总大小保持恒定。基于随机概率分布实现动态生长机制。
"""
import os
import math
import time
import random
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceLevel(Enum):
    """资源级别"""
    MINIMAL = 'minimal'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    MAXIMAL = 'maximal'


@dataclass
class NetworkDimensions:
    """网络维度配置"""
    num_neurons: int = 100
    num_synapses: int = 500
    input_size: int = 10
    output_size: int = 10
    num_layers: int = 3
    resource_level: ResourceLevel = ResourceLevel.MEDIUM


class ResourceAwareNetworkSizer:
    """资源感知的网络尺寸确定器"""
    
    def __init__(self):
        self._cpu_count = os.cpu_count() or 4
        self._max_memory_mb = self._estimate_max_memory()
    
    def _estimate_max_memory(self) -> int:
        """估算可用最大内存（MB）"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return int(mem.total / (1024 * 1024))
        except ImportError:
            return 8192
    
    def get_resource_level(self) -> ResourceLevel:
        """根据系统资源确定资源级别"""
        memory = self._max_memory_mb
        
        if memory < 2048:
            return ResourceLevel.MINIMAL
        elif memory < 4096:
            return ResourceLevel.SMALL
        elif memory < 8192:
            return ResourceLevel.MEDIUM
        elif memory < 16384:
            return ResourceLevel.LARGE
        else:
            return ResourceLevel.MAXIMAL
    
    def calculate_dimensions(self, input_size: int = 10, output_size: int = 10) -> NetworkDimensions:
        """根据资源级别计算网络维度"""
        level = self.get_resource_level()
        
        configs = {
            ResourceLevel.MINIMAL: {
                'num_neurons': 50,
                'num_synapses': 200,
                'num_layers': 2,
            },
            ResourceLevel.SMALL: {
                'num_neurons': 100,
                'num_synapses': 500,
                'num_layers': 2,
            },
            ResourceLevel.MEDIUM: {
                'num_neurons': 250,
                'num_synapses': 1500,
                'num_layers': 3,
            },
            ResourceLevel.LARGE: {
                'num_neurons': 500,
                'num_synapses': 3000,
                'num_layers': 3,
            },
            ResourceLevel.MAXIMAL: {
                'num_neurons': 1000,
                'num_synapses': 10000,
                'num_layers': 4,
            },
        }
        
        cfg = configs[level]
        
        logger.info(f"资源级别: {level.value}, 网络配置: {cfg}")
        
        return NetworkDimensions(
            num_neurons=cfg['num_neurons'],
            num_synapses=cfg['num_synapses'],
            input_size=input_size,
            output_size=output_size,
            num_layers=cfg['num_layers'],
            resource_level=level,
        )


class GrowthEvent(Enum):
    """生长事件类型"""
    DENDRITIC_BRANCH = 'dendritic_branch'
    AXONAL_ARBORIZATION = 'axonal_arborization'
    SYNAPSE_REWIRING = 'synapse_rewiring'
    SYNAPSE_STRENGTH_REDISTRIBUTION = 'synapse_strength_redistribution'
    NEURON_MIGRATION = 'neuron_migration'
    DENDRITIC_PRUNING = 'dendritic_pruning'


@dataclass
class GrowthProbabilities:
    """生长概率配置"""
    dendritic_branch_prob: float = 0.01
    axonal_arborization_prob: float = 0.008
    synapse_rewiring_prob: float = 0.02
    synapse_redistribution_prob: float = 0.015
    neuron_migration_prob: float = 0.005
    dendritic_pruning_prob: float = 0.005


@dataclass
class DendriteBranch:
    """树突分支"""
    branch_id: str
    parent_neuron: int
    length: float = 1.0
    thickness: float = 0.5
    synapse_count: int = 0
    last_activity: float = 0.0
    active: bool = True


@dataclass
class AxonTerminal:
    """轴突末梢"""
    terminal_id: str
    parent_neuron: int
    reach: float = 1.0
    target_neurons: List[int] = field(default_factory=list)
    active: bool = True


class GrowthCapableNeuron:
    """具有生长能力的神经元"""
    
    def __init__(self, neuron_id: int, layer: int = 0, params: Dict = None):
        self.id = neuron_id
        self.layer = layer
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
        
        # 树突分支
        self.dendrite_branches: Dict[str, DendriteBranch] = {}
        self._initialize_dendrites()
        
        # 轴突末梢
        self.axon_terminals: Dict[str, AxonTerminal] = {}
        self._initialize_axons()
        
        # 生长状态
        self.growth_events: List[Dict] = []
        self.pruning_candidates: List[str] = []
        
        # 统计
        self.spike_counter = 0
        self.firing_rate = 0.0
        self.recent_firing = []
    
    def _initialize_dendrites(self):
        """初始化树突分支"""
        num_branches = self.params.get('initial_dendrites', 3)
        for i in range(num_branches):
            branch = DendriteBranch(
                branch_id=f"dend_{self.id}_{i}",
                parent_neuron=self.id,
                length=1.0 + random.uniform(-0.2, 0.2),
                thickness=0.5 + random.uniform(-0.1, 0.1),
            )
            self.dendrite_branches[branch.branch_id] = branch
    
    def _initialize_axons(self):
        """初始化轴突末梢"""
        num_terminals = self.params.get('initial_axons', 2)
        for i in range(num_terminals):
            terminal = AxonTerminal(
                terminal_id=f"axon_{self.id}_{i}",
                parent_neuron=self.id,
                reach=1.0 + random.uniform(-0.3, 0.3),
            )
            self.axon_terminals[terminal.terminal_id] = terminal
    
    def update(self, dt: float, t: float):
        """更新神经元状态"""
        if t - self.last_spike_time < self.tau_ref:
            self.refractory = True
            self.V_m = float(self.V_reset)
            return
        
        self.refractory = False
        
        I_syn_val = float(self.I_syn) if hasattr(self.I_syn, '__len__') else self.I_syn
        I_ext_val = float(self.I_ext) if hasattr(self.I_ext, '__len__') else self.I_ext
        
        dV_dt = (self.g_L * (self.V_rest - self.V_m) + I_syn_val + I_ext_val) / self.tau_m
        self.V_m = float(self.V_m + dV_dt * dt)
        
        self.spiked = False
        if self.V_m >= self.V_th:
            self.spiked = True
            self.V_m = float(self.V_reset)
            self.last_spike_time = t
            self.spike_counter += 1
            self.recent_firing.append(t)
            if len(self.recent_firing) > 100:
                self.recent_firing.pop(0)
        
        self.I_syn = 0.0
        
        if self.recent_firing:
            intervals = [self.recent_firing[i] - self.recent_firing[i-1] for i in range(1, len(self.recent_firing))]
            self.firing_rate = len(intervals) / (self.recent_firing[-1] - self.recent_firing[0]) if intervals else 0.0
    
    def add_synaptic_input(self, weight: float):
        """添加突触输入"""
        self.I_syn += weight
    
    def trigger_dendritic_growth(self, prob_distribution: str = 'poisson', rate: float = 0.1):
        """触发树突生长事件"""
        if prob_distribution == 'poisson':
            num_events = np.random.poisson(rate)
        elif prob_distribution == 'bernoulli':
            num_events = 1 if random.random() < rate else 0
        else:
            num_events = 1 if random.random() < rate else 0
        
        for _ in range(num_events):
            if len(self.dendrite_branches) < 10:
                new_id = f"dend_{self.id}_{len(self.dendrite_branches)}"
                new_branch = DendriteBranch(
                    branch_id=new_id,
                    parent_neuron=self.id,
                    length=random.gauss(1.0, 0.2),
                    thickness=random.gauss(0.5, 0.1),
                )
                self.dendrite_branches[new_id] = new_branch
                self.growth_events.append({
                    'event': GrowthEvent.DENDRITIC_BRANCH,
                    'branch_id': new_id,
                    'time': time.time(),
                })
    
    def trigger_axonal_growth(self, prob_distribution: str = 'poisson', rate: float = 0.08):
        """触发轴突生长事件"""
        if prob_distribution == 'poisson':
            num_events = np.random.poisson(rate)
        else:
            num_events = 1 if random.random() < rate else 0
        
        for _ in range(num_events):
            if len(self.axon_terminals) < 8:
                new_id = f"axon_{self.id}_{len(self.axon_terminals)}"
                new_terminal = AxonTerminal(
                    terminal_id=new_id,
                    parent_neuron=self.id,
                    reach=random.gauss(1.0, 0.3),
                )
                self.axon_terminals[new_id] = new_terminal
                self.growth_events.append({
                    'event': GrowthEvent.AXONAL_ARBORIZATION,
                    'terminal_id': new_id,
                    'time': time.time(),
                })
    
    def prune_dendrite(self, prob_distribution: str = 'bernoulli', prob: float = 0.05):
        """修剪树突分支"""
        if prob_distribution == 'bernoulli':
            should_prune = random.random() < prob
        else:
            should_prune = random.random() < prob
        
        if should_prune and len(self.dendrite_branches) > 1:
            inactive_branches = [
                bid for bid, branch in self.dendrite_branches.items()
                if branch.synapse_count == 0
            ]
            
            if inactive_branches:
                to_prune = random.choice(inactive_branches)
                del self.dendrite_branches[to_prune]
                self.growth_events.append({
                    'event': GrowthEvent.DENDRITIC_PRUNING,
                    'branch_id': to_prune,
                    'time': time.time(),
                })
    
    def get_growth_state(self) -> Dict:
        """获取生长状态"""
        return {
            'neuron_id': self.id,
            'num_dendrites': len(self.dendrite_branches),
            'num_axon_terminals': len(self.axon_terminals),
            'spike_counter': self.spike_counter,
            'firing_rate': self.firing_rate,
            'growth_events': len(self.growth_events),
        }


class GrowthCapableSynapse:
    """具有生长能力的突触"""
    
    def __init__(self, synapse_id: str, pre_neuron: int, post_neuron: int, weight: float = 0.5):
        self.id = synapse_id
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        
        # STDP参数
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        
        # 权重范围
        self.w_min = 0.001
        self.w_max = 1.0
        
        # 状态
        self.pre_last_spike = -float('inf')
        self.post_last_spike = -float('inf')
        self.strength_history: List[float] = []
        self.last_update_time = 0.0
        
        # 生长状态
        self.rewiring_candidate = False
        self.weaken_count = 0
        self.strengthen_count = 0
    
    def update_weight(self, pre_spike: bool, post_spike: bool, dt: float = 1.0, t: float = 0.0):
        """更新突触权重（STDP规则）"""
        delta_w = 0.0
        
        if pre_spike:
            delta_w += self.A_plus * math.exp(-(t - self.post_last_spike) / self.tau_plus)
        
        if post_spike:
            delta_w -= self.A_minus * math.exp(-(t - self.pre_last_spike) / self.tau_minus)
        
        self.weight = max(self.w_min, min(self.w_max, self.weight + delta_w))
        
        self.strength_history.append(self.weight)
        if len(self.strength_history) > 50:
            self.strength_history.pop(0)
        
        if delta_w > 0:
            self.strengthen_count += 1
        elif delta_w < 0:
            self.weaken_count += 1
        
        self.last_update_time = t
        
        if pre_spike:
            self.pre_last_spike = t
        if post_spike:
            self.post_last_spike = t
    
    def trigger_rewiring(self, prob_distribution: str = 'bernoulli', prob: float = 0.02) -> bool:
        """触发突触重布线"""
        if prob_distribution == 'bernoulli':
            should_rewire = random.random() < prob
        elif prob_distribution == 'gaussian':
            should_rewire = random.random() < max(0, min(1, random.gauss(prob, prob * 0.3)))
        else:
            should_rewire = random.random() < prob
        
        if should_rewire and self.weight < 0.1:
            self.rewiring_candidate = True
            return True
        
        return False
    
    def redistribute_strength(self, other_synapse, prob_distribution: str = 'gaussian', 
                              mean_ratio: float = 0.3, std_ratio: float = 0.1):
        """重新分配突触强度"""
        if prob_distribution == 'gaussian':
            ratio = max(0.1, min(0.5, random.gauss(mean_ratio, std_ratio)))
        else:
            ratio = mean_ratio
        
        total_weight = self.weight + other_synapse.weight
        self.weight = total_weight * ratio
        other_synapse.weight = total_weight * (1 - ratio)
    
    def get_growth_state(self) -> Dict:
        """获取生长状态"""
        return {
            'synapse_id': self.id,
            'pre_neuron': self.pre_neuron,
            'post_neuron': self.post_neuron,
            'weight': round(self.weight, 4),
            'weaken_count': self.weaken_count,
            'strengthen_count': self.strengthen_count,
            'rewiring_candidate': self.rewiring_candidate,
        }


class GrowthController:
    """生长控制器"""
    
    def __init__(self, probabilities: GrowthProbabilities = None):
        self.probabilities = probabilities or GrowthProbabilities()
        
        # 稳定性监控
        self.firing_rate_history: List[float] = []
        self.average_firing_rate = 0.0
        self.target_firing_rate = 0.1
        
        # 守恒状态
        self.total_neuron_count = 0
        self.total_synapse_count = 0
        
        # 事件计数
        self.event_counts: Dict[str, int] = defaultdict(int)
        
        # 时间步
        self.timestep = 0
    
    def check_stability(self, neurons: List[GrowthCapableNeuron]) -> bool:
        """检查网络稳定性"""
        firing_rates = [n.firing_rate for n in neurons if n.firing_rate > 0]
        
        if not firing_rates:
            return True
        
        avg_rate = sum(firing_rates) / len(firing_rates)
        self.firing_rate_history.append(avg_rate)
        
        if len(self.firing_rate_history) > 100:
            self.firing_rate_history.pop(0)
        
        self.average_firing_rate = avg_rate
        
        variance = sum((r - avg_rate) ** 2 for r in firing_rates) / len(firing_rates)
        
        stable = (0.01 < avg_rate < 0.5) and (variance < 0.01)
        
        if not stable:
            logger.warning(f"网络不稳定: 平均 firing rate={avg_rate:.4f}, 方差={variance:.4f}")
        
        return stable
    
    def adjust_growth_probabilities(self, stable: bool):
        """根据稳定性调整生长概率"""
        factor = 0.8 if not stable else 1.0
        
        self.probabilities.dendritic_branch_prob *= factor
        self.probabilities.axonal_arborization_prob *= factor
        self.probabilities.synapse_rewiring_prob *= factor
    
    def execute_growth_step(self, neurons: List[GrowthCapableNeuron], 
                            synapses: List[GrowthCapableSynapse],
                            t: float = 0.0):
        """执行生长步骤"""
        self.timestep += 1
        
        for neuron in neurons:
            neuron.trigger_dendritic_growth(
                prob_distribution='poisson',
                rate=self.probabilities.dendritic_branch_prob
            )
            
            neuron.trigger_axonal_growth(
                prob_distribution='poisson',
                rate=self.probabilities.axonal_arborization_prob
            )
            
            neuron.prune_dendrite(
                prob_distribution='bernoulli',
                prob=self.probabilities.dendritic_pruning_prob
            )
        
        rewiring_candidates = []
        for synapse in synapses:
            if synapse.trigger_rewiring(
                prob_distribution='bernoulli',
                prob=self.probabilities.synapse_rewiring_prob
            ):
                rewiring_candidates.append(synapse)
        
        if rewiring_candidates:
            self._execute_synapse_rewiring(rewiring_candidates, neurons)
        
        if random.random() < self.probabilities.synapse_redistribution_prob:
            self._execute_strength_redistribution(synapses)
        
        stable = self.check_stability(neurons)
        self.adjust_growth_probabilities(stable)
    
    def _execute_synapse_rewiring(self, candidates: List[GrowthCapableSynapse],
                                   neurons: List[GrowthCapableNeuron]):
        """执行突触重布线"""
        neuron_ids = [n.id for n in neurons]
        
        for synapse in candidates:
            new_pre = random.choice(neuron_ids)
            new_post = random.choice(neuron_ids)
            
            while new_pre == new_post:
                new_post = random.choice(neuron_ids)
            
            old_pre, old_post = synapse.pre_neuron, synapse.post_neuron
            synapse.pre_neuron = new_pre
            synapse.post_neuron = new_post
            synapse.weight = random.gauss(0.3, 0.1)
            synapse.rewiring_candidate = False
            
            self.event_counts['synapse_rewiring'] += 1
            
            logger.debug(f"突触重布线: {synapse.id} ({old_pre}->{old_post}) -> ({new_pre}->{new_post})")
    
    def _execute_strength_redistribution(self, synapses: List[GrowthCapableSynapse]):
        """执行强度重新分配"""
        if len(synapses) < 2:
            return
        
        idx1, idx2 = random.sample(range(len(synapses)), 2)
        synapses[idx1].redistribute_strength(
            synapses[idx2],
            prob_distribution='gaussian',
            mean_ratio=0.3,
            std_ratio=0.1
        )
        
        self.event_counts['strength_redistribution'] += 1
    
    def get_growth_report(self) -> Dict:
        """获取生长报告"""
        return {
            'timestep': self.timestep,
            'average_firing_rate': round(self.average_firing_rate, 4),
            'target_firing_rate': self.target_firing_rate,
            'event_counts': dict(self.event_counts),
            'probabilities': {
                'dendritic_branch': self.probabilities.dendritic_branch_prob,
                'axonal_arborization': self.probabilities.axonal_arborization_prob,
                'synapse_rewiring': self.probabilities.synapse_rewiring_prob,
                'synapse_redistribution': self.probabilities.synapse_redistribution_prob,
                'neuron_migration': self.probabilities.neuron_migration_prob,
                'dendritic_pruning': self.probabilities.dendritic_pruning_prob,
            },
        }


class SpikingGrowthNetwork:
    """具有生长能力的脉冲神经网络"""
    
    def __init__(self, dimensions: NetworkDimensions = None,
                 growth_probabilities: GrowthProbabilities = None):
        if dimensions is None:
            sizer = ResourceAwareNetworkSizer()
            dimensions = sizer.calculate_dimensions()
        
        self.dimensions = dimensions
        self.growth_probabilities = growth_probabilities or GrowthProbabilities()
        
        # 网络组件
        self.neurons: Dict[int, GrowthCapableNeuron] = {}
        self.synapses: Dict[str, GrowthCapableSynapse] = {}
        
        # 生长控制器
        self.growth_controller = GrowthController(self.growth_probabilities)
        
        # 网络结构
        self.neuron_layers: Dict[int, List[int]] = defaultdict(list)
        
        # 初始化网络
        self._initialize_network()
        
        logger.info(f"生长SNN初始化完成: {len(self.neurons)}个神经元, {len(self.synapses)}个突触")
    
    def _initialize_network(self):
        """初始化网络"""
        layers = self.dimensions.num_layers
        neurons_per_layer = self.dimensions.num_neurons // layers
        
        for layer in range(layers):
            start_id = layer * neurons_per_layer
            for i in range(neurons_per_layer):
                neuron_id = start_id + i
                neuron = GrowthCapableNeuron(neuron_id, layer=layer)
                self.neurons[neuron_id] = neuron
                self.neuron_layers[layer].append(neuron_id)
        
        extra_neurons = self.dimensions.num_neurons % layers
        for i in range(extra_neurons):
            neuron_id = self.dimensions.num_neurons - extra_neurons + i
            neuron = GrowthCapableNeuron(neuron_id, layer=layers - 1)
            self.neurons[neuron_id] = neuron
            self.neuron_layers[layers - 1].append(neuron_id)
        
        self._initialize_synapses()
    
    def _initialize_synapses(self):
        """初始化突触"""
        synapse_id = 0
        target_synapses = self.dimensions.num_synapses
        
        for layer in range(self.dimensions.num_layers - 1):
            pre_layer = self.neuron_layers[layer]
            post_layer = self.neuron_layers[layer + 1]
            
            for pre in pre_layer:
                num_connections = min(3, len(post_layer))
                posts = random.sample(post_layer, num_connections)
                
                for post in posts:
                    if synapse_id >= target_synapses:
                        return
                    
                    synapse = GrowthCapableSynapse(
                        synapse_id=f"syn_{synapse_id}",
                        pre_neuron=pre,
                        post_neuron=post,
                        weight=random.uniform(0.1, 0.9)
                    )
                    self.synapses[synapse.id] = synapse
                    synapse_id += 1
        
        while synapse_id < target_synapses:
            pre = random.choice(list(self.neurons.keys()))
            post = random.choice(list(self.neurons.keys()))
            
            if pre != post and not any(
                s.pre_neuron == pre and s.post_neuron == post
                for s in self.synapses.values()
            ):
                synapse = GrowthCapableSynapse(
                    synapse_id=f"syn_{synapse_id}",
                    pre_neuron=pre,
                    post_neuron=post,
                    weight=random.uniform(0.1, 0.9)
                )
                self.synapses[synapse.id] = synapse
                synapse_id += 1
    
    def step(self, input_signal: np.ndarray = None, dt: float = 1.0, t: float = 0.0):
        """执行一个时间步"""
        if input_signal is not None and len(input_signal) > 0:
            input_neurons = self.neuron_layers[0][:len(input_signal)]
            for i, neuron_id in enumerate(input_neurons):
                val = input_signal[i]
                if isinstance(val, (np.ndarray, list)):
                    val = float(np.mean(val))
                else:
                    val = float(val)
                self.neurons[neuron_id].I_ext = val
        
        for neuron in self.neurons.values():
            neuron.update(dt, t)
        
        for synapse in self.synapses.values():
            pre_spike = self.neurons[synapse.pre_neuron].spiked
            post_spike = self.neurons[synapse.post_neuron].spiked
            synapse.update_weight(pre_spike, post_spike, dt, t)
            
            if pre_spike:
                self.neurons[synapse.post_neuron].add_synaptic_input(synapse.weight)
        
        self.growth_controller.execute_growth_step(
            list(self.neurons.values()),
            list(self.synapses.values()),
            t
        )
        
        return self.get_output()
    
    def get_output(self) -> np.ndarray:
        """获取输出"""
        output_layer = self.neuron_layers[self.dimensions.num_layers - 1]
        output = np.zeros(min(self.dimensions.output_size, len(output_layer)), dtype=np.float32)
        
        for i, neuron_id in enumerate(output_layer[:self.dimensions.output_size]):
            output[i] = float(self.neurons[neuron_id].V_m)
        
        return output
    
    def run(self, num_steps: int, input_signals: np.ndarray = None):
        """运行多个时间步"""
        outputs = []
        
        for step in range(num_steps):
            t = step * 1.0
            input_signal = None
            
            if input_signals is not None and step < len(input_signals):
                input_signal = input_signals[step]
            
            self.step(input_signal, t=t)
            outputs.append(self.get_output())
        
        return np.array(outputs)
    
    def get_statistics(self) -> Dict:
        """获取网络统计信息"""
        firing_rates = [n.firing_rate for n in self.neurons.values()]
        weights = [s.weight for s in self.synapses.values()]
        
        return {
            'num_neurons': len(self.neurons),
            'num_synapses': len(self.synapses),
            'num_layers': self.dimensions.num_layers,
            'resource_level': self.dimensions.resource_level.value,
            'avg_firing_rate': sum(firing_rates) / len(firing_rates) if firing_rates else 0.0,
            'min_firing_rate': min(firing_rates) if firing_rates else 0.0,
            'max_firing_rate': max(firing_rates) if firing_rates else 0.0,
            'avg_weight': sum(weights) / len(weights) if weights else 0.0,
            'min_weight': min(weights) if weights else 0.0,
            'max_weight': max(weights) if weights else 0.0,
            'growth_report': self.growth_controller.get_growth_report(),
        }
    
    def get_neuron_growth_states(self) -> List[Dict]:
        """获取所有神经元的生长状态"""
        return [neuron.get_growth_state() for neuron in self.neurons.values()]
    
    def get_synapse_growth_states(self) -> List[Dict]:
        """获取所有突触的生长状态"""
        return [synapse.get_growth_state() for synapse in self.synapses.values()]
    
    def verify_conservation(self) -> bool:
        """验证守恒定律"""
        neuron_count_ok = len(self.neurons) == self.dimensions.num_neurons
        synapse_count_ok = len(self.synapses) == self.dimensions.num_synapses
        
        if not neuron_count_ok or not synapse_count_ok:
            logger.error(f"守恒验证失败: 神经元={len(self.neurons)} (预期={self.dimensions.num_neurons}), "
                        f"突触={len(self.synapses)} (预期={self.dimensions.num_synapses})")
        
        return neuron_count_ok and synapse_count_ok