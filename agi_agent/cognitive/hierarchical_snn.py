#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层级脉冲神经网络（Hierarchical SNN）

实现自底向上的层级化建模思路：
1. 感官编码模块 - 模拟外周感官系统（特征提取+脉冲编码）
2. 缓冲整合模块 - 感官缓冲+特征增强（侧抑制+特征整合）
3. 模式整合模块 - 微柱模型（特征拓扑+高阶模式整合）
4. 高级认知模块 - 海马记忆+基底节预测

核心设计原则：
- 拓扑保真：特征拓扑映射
- 时间编码：TTFS/Rate/Synchrony
- 可塑性学习：STDP
- 事件驱动：低功耗计算

版本: 2.0 (通用版)
日期: 2026-07-05
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


class EncodingScheme(Enum):
    TTFS = "ttfs"
    RATE = "rate"
    SYNCHRONY = "synchrony"
    PHASE = "phase"


@dataclass
class SpikeEvent:
    neuron_id: int
    timestamp: float
    channel: int
    amplitude: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Spike(neuron={self.neuron_id}, t={self.timestamp:.4f}s, ch={self.channel})"


@dataclass
class FeatureChannel:
    channel_id: int
    feature_dimension: int
    center_value: float = 0.0
    bandwidth: float = 1.0
    spikes: List[SpikeEvent] = field(default_factory=list)


class FeatureProjection:
    def __init__(self, num_channels: int = 64, feature_dim: int = 128):
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.channels = []
        self._init_channels()

    def _init_channels(self):
        for i in range(self.num_channels):
            self.channels.append(FeatureChannel(
                channel_id=i,
                feature_dimension=i % self.feature_dim,
                center_value=i / self.num_channels,
                bandwidth=1.0 / self.num_channels
            ))

    def project(self, feature_vector: np.ndarray) -> List[np.ndarray]:
        projected = []
        for channel in self.channels:
            if channel.feature_dimension < len(feature_vector):
                val = feature_vector[channel.feature_dimension]
                similarity = np.exp(-((val - channel.center_value) ** 2) / (2 * channel.bandwidth ** 2))
                projected.append(np.ones(len(feature_vector)) * similarity * val)
            else:
                projected.append(np.zeros(len(feature_vector)))
        return projected


class SignalToSpikeConverter:
    def __init__(self, time_resolution: float = 0.001):
        self.time_resolution = time_resolution
        
        self.attack_time = 0.005
        self.release_time = 0.05
        self.compression_ratio = 0.3
        
        self.ttfs_max_delay = 0.02
        self.ttfs_threshold = 0.01

    def half_wave_rectify(self, signal: np.ndarray) -> np.ndarray:
        return np.maximum(signal, 0)

    def adaptive_gain_control(self, signal: np.ndarray) -> np.ndarray:
        envelope = np.abs(signal)
        
        attack_coeff = 1 - np.exp(-self.time_resolution / self.attack_time)
        release_coeff = 1 - np.exp(-self.time_resolution / self.release_time)
        
        smoothed_env = np.zeros_like(envelope)
        smoothed_env[0] = envelope[0]
        
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_env[i-1]:
                smoothed_env[i] = smoothed_env[i-1] + attack_coeff * (envelope[i] - smoothed_env[i-1])
            else:
                smoothed_env[i] = smoothed_env[i-1] + release_coeff * (envelope[i] - smoothed_env[i-1])
        
        max_env = np.max(smoothed_env) if np.max(smoothed_env) > 0 else 1.0
        gain = (1.0 / (1.0 + smoothed_env / max_env)) ** self.compression_ratio
        
        return signal * gain

    def encode_ttfs(self, signal: np.ndarray, channel_id: int) -> List[SpikeEvent]:
        spikes = []
        signal = self.half_wave_rectify(signal)
        signal = self.adaptive_gain_control(signal)
        
        max_val = np.max(signal) if np.max(signal) > 0 else 1.0
        normalized = signal / max_val
        
        above_thresh = np.where(normalized > self.ttfs_threshold)[0]
        
        if len(above_thresh) > 0:
            first_idx = above_thresh[0]
            base_time = first_idx * self.time_resolution
            
            amplitude = normalized[first_idx]
            delay = (1.0 - amplitude) * self.ttfs_max_delay
            
            spike_time = base_time + delay
            
            spikes.append(SpikeEvent(
                neuron_id=channel_id,
                timestamp=spike_time,
                channel=channel_id,
                amplitude=amplitude,
                features={'encoding': 'ttfs', 'delay': delay}
            ))
        
        return spikes

    def encode_rate(self, signal: np.ndarray, channel_id: int, 
                   window_size: float = 0.05) -> List[SpikeEvent]:
        spikes = []
        signal = self.half_wave_rectify(signal)
        signal = self.adaptive_gain_control(signal)
        
        window_samples = int(window_size / self.time_resolution)
        step_samples = window_samples // 2
        
        for i in range(0, len(signal) - window_samples, step_samples):
            window = signal[i:i+window_samples]
            mean_amplitude = np.mean(window)
            
            firing_rate = min(200, mean_amplitude * 400)
            
            if firing_rate > 0:
                spike_interval = 1.0 / firing_rate
                num_spikes = int(window_size / spike_interval)
                
                for j in range(num_spikes):
                    spike_time = (i + j * spike_interval / self.time_resolution) * self.time_resolution
                    if spike_time < len(signal) * self.time_resolution:
                        spikes.append(SpikeEvent(
                            neuron_id=channel_id,
                            timestamp=spike_time,
                            channel=channel_id,
                            amplitude=mean_amplitude,
                            features={'encoding': 'rate', 'firing_rate': firing_rate}
                        ))
        
        return spikes


class SensoryEncoder:
    def __init__(self, num_channels: int = 128, feature_dim: int = 128, 
                 encoding: EncodingScheme = EncodingScheme.TTFS):
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.encoding = encoding
        self.projection = FeatureProjection(num_channels, feature_dim)
        self.converter = SignalToSpikeConverter()

    def process(self, feature_vector: np.ndarray) -> List[SpikeEvent]:
        if len(feature_vector) == 0:
            return []
            
        if len(feature_vector) < self.feature_dim:
            padded = np.zeros(self.feature_dim)
            padded[:len(feature_vector)] = feature_vector
            feature_vector = padded
        
        projected_signals = self.projection.project(feature_vector)
        
        all_spikes = []
        for i, signal in enumerate(projected_signals):
            if self.encoding == EncodingScheme.TTFS:
                spikes = self.converter.encode_ttfs(signal, i)
            else:
                spikes = self.converter.encode_rate(signal, i)
            all_spikes.extend(spikes)
        
        return sorted(all_spikes, key=lambda x: x.timestamp)


class LIFNeuron:
    def __init__(self, neuron_id: int, v_rest: float = -70.0, v_thresh: float = -50.0, 
                 v_reset: float = -75.0, tau_m: float = 20.0, tau_ref: float = 2.0):
        self.neuron_id = neuron_id
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.tau_ref = tau_ref
        
        self.V = v_rest
        self.I_ext = 0.0
        self.last_spike_time = -float('inf')
        self.refractory = False

    def update(self, dt: float, t: float) -> Optional[SpikeEvent]:
        if self.refractory:
            if t - self.last_spike_time >= self.tau_ref / 1000.0:
                self.refractory = False
            else:
                self.V = self.v_reset
                return None
        
        dV = (-(self.V - self.v_rest) + self.tau_m * self.I_ext) / self.tau_m
        self.V += dV * dt
        
        self.I_ext = 0.0
        
        if self.V >= self.v_thresh:
            self.V = self.v_reset
            self.last_spike_time = t
            self.refractory = True
            return SpikeEvent(
                neuron_id=self.neuron_id,
                timestamp=t,
                channel=self.neuron_id
            )
        
        return None


class STDPSynapse:
    def __init__(self, pre_neuron: int, post_neuron: int, weight: float = 0.5,
                 A_plus: float = 0.01, A_minus: float = 0.012,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 w_min: float = 0.0, w_max: float = 1.0):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        
        self.last_pre_spike = -float('inf')
        self.last_post_spike = -float('inf')

    def apply_stdp(self, pre_spike_time: float, post_spike_time: float):
        delta_t = post_spike_time - pre_spike_time
        
        if delta_t > 0 and delta_t < 0.05:
            dw = self.A_plus * np.exp(-delta_t * 1000 / self.tau_plus)
            self.weight = min(self.w_max, self.weight + dw)
        elif delta_t < 0 and delta_t > -0.05:
            dw = -self.A_minus * np.exp(delta_t * 1000 / self.tau_minus)
            self.weight = max(self.w_min, self.weight + dw)


class SensoryBuffer:
    def __init__(self, num_neurons: int = 128):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(i) for i in range(num_neurons)]
        self.synapses = []
        self._init_synapses()

    def _init_synapses(self):
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < 0.15:
                    self.synapses.append(STDPSynapse(
                        pre_neuron=i, post_neuron=j,
                        weight=np.random.uniform(0.05, 0.3)
                    ))

    def process(self, spikes: List[SpikeEvent], t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        for neuron in self.neurons:
            for spike in spikes:
                for synapse in self.synapses:
                    if synapse.pre_neuron == spike.channel:
                        neuron.I_ext += synapse.weight * spike.amplitude
            
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        for synapse in self.synapses:
            pre_spike = next((s for s in spikes if s.channel == synapse.pre_neuron), None)
            post_spike = next((s for s in output_spikes if s.channel == synapse.post_neuron), None)
            if pre_spike and post_spike:
                synapse.apply_stdp(pre_spike.timestamp, post_spike.timestamp)
        
        return output_spikes


class FeatureExtractor:
    def __init__(self, num_neurons: int = 80, input_size: int = 128):
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.neurons = [LIFNeuron(i) for i in range(num_neurons)]
        self.input_synapses = []
        self.lateral_synapses = []
        self._init_synapses()

    def _init_synapses(self):
        for i in range(self.input_size):
            for j in range(self.num_neurons):
                if np.random.random() < 0.2:
                    self.input_synapses.append(STDPSynapse(
                        pre_neuron=i, post_neuron=j,
                        weight=np.random.uniform(0.1, 0.4)
                    ))
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < 0.1:
                    weight = np.random.uniform(-0.2, -0.05) if i != j else 0
                    self.lateral_synapses.append(STDPSynapse(
                        pre_neuron=i, post_neuron=j,
                        weight=weight
                    ))

    def process(self, spikes: List[SpikeEvent], context_spikes: List[SpikeEvent], 
                t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        for neuron in self.neurons:
            for spike in spikes:
                for synapse in self.input_synapses:
                    if synapse.pre_neuron == spike.channel % self.input_size:
                        neuron.I_ext += synapse.weight * spike.amplitude
            
            for spike in context_spikes:
                for synapse in self.input_synapses:
                    if synapse.pre_neuron == spike.channel % self.input_size:
                        neuron.I_ext += synapse.weight * spike.amplitude * 0.5
            
            for synapse in self.lateral_synapses:
                for spike in spikes:
                    if spike.channel == synapse.pre_neuron:
                        neuron.I_ext += synapse.weight * 0.3
            
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        return output_spikes


class CorticalColumn:
    def __init__(self, column_id: int, num_pyramidal: int = 8, num_inhibitory: int = 2):
        self.column_id = column_id
        self.pyramidal_neurons = [LIFNeuron(column_id * 10 + i) for i in range(num_pyramidal)]
        self.inhibitory_neurons = [LIFNeuron(column_id * 10 + num_pyramidal + i) for i in range(num_inhibitory)]
        self.intra_column_synapses = []
        self._init_synapses()

    def _init_synapses(self):
        for i, pre in enumerate(self.pyramidal_neurons):
            for j, post in enumerate(self.pyramidal_neurons):
                if i != j and np.random.random() < 0.3:
                    self.intra_column_synapses.append(STDPSynapse(
                        pre_neuron=pre.neuron_id, post_neuron=post.neuron_id,
                        weight=np.random.uniform(0.1, 0.4)
                    ))
        
        for inh in self.inhibitory_neurons:
            for pyr in self.pyramidal_neurons:
                self.intra_column_synapses.append(STDPSynapse(
                    pre_neuron=inh.neuron_id, post_neuron=pyr.neuron_id,
                    weight=np.random.uniform(-0.3, -0.1)
                ))

    def process(self, spikes: List[SpikeEvent], t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        all_neurons = self.pyramidal_neurons + self.inhibitory_neurons
        
        for neuron in all_neurons:
            for spike in spikes:
                for synapse in self.intra_column_synapses:
                    if synapse.pre_neuron == spike.channel:
                        neuron.I_ext += synapse.weight * spike.amplitude * 0.5
            
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        return output_spikes


class PatternIntegrator:
    def __init__(self, num_columns: int = 40):
        self.num_columns = num_columns
        self.columns = [CorticalColumn(i) for i in range(num_columns)]
        self.inter_column_synapses = []
        self._init_inter_column_synapses()

    def _init_inter_column_synapses(self):
        for i, col1 in enumerate(self.columns):
            for j, col2 in enumerate(self.columns):
                if i != j and np.random.random() < 0.05:
                    for pyr1 in col1.pyramidal_neurons:
                        for pyr2 in col2.pyramidal_neurons:
                            if np.random.random() < 0.3:
                                self.inter_column_synapses.append(STDPSynapse(
                                    pre_neuron=pyr1.neuron_id,
                                    post_neuron=pyr2.neuron_id,
                                    weight=np.random.uniform(0.05, 0.2)
                                ))

    def process(self, spikes: List[SpikeEvent], t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        for column in self.columns:
            col_spikes = column.process(spikes, t, dt)
            output_spikes.extend(col_spikes)
        
        for synapse in self.inter_column_synapses:
            pre_spike = next((s for s in output_spikes if s.channel == synapse.pre_neuron), None)
            post_spike = next((s for s in output_spikes if s.channel == synapse.post_neuron), None)
            if pre_spike and post_spike:
                synapse.apply_stdp(pre_spike.timestamp, post_spike.timestamp)
        
        return output_spikes


class HippocampalMemory:
    def __init__(self, num_neurons: int = 500):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(i) for i in range(num_neurons)]
        self.recurrent_synapses = []
        self.memory_traces: Dict[str, List[SpikeEvent]] = {}
        self.current_sequence: List[SpikeEvent] = []
        self._init_recurrent_connections()

    def _init_recurrent_connections(self):
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < 0.15:
                    self.recurrent_synapses.append(STDPSynapse(
                        pre_neuron=i, post_neuron=j,
                        weight=np.random.uniform(0.1, 0.4),
                        A_plus=0.015, A_minus=0.018,
                        tau_plus=25.0, tau_minus=25.0
                    ))

    def process(self, spikes: List[SpikeEvent], t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        for neuron in self.neurons:
            for spike in spikes:
                neuron.I_ext += 0.3 * spike.amplitude
            
            for synapse in self.recurrent_synapses:
                for spike in output_spikes:
                    if spike.channel == synapse.pre_neuron:
                        neuron.I_ext += synapse.weight * 0.5
            
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        self.current_sequence.extend(output_spikes[:10])
        if len(self.current_sequence) > 50:
            self.current_sequence = self.current_sequence[-50:]
        
        return output_spikes

    def store_sequence(self, pattern_id: str):
        self.memory_traces[pattern_id] = list(self.current_sequence)

    def retrieve_sequence(self, pattern_id: str) -> Optional[List[SpikeEvent]]:
        return self.memory_traces.get(pattern_id)


class BasalGanglia:
    def __init__(self, num_neurons: int = 300):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(i) for i in range(num_neurons)]
        self.striatum_neurons = self.neurons[:200]
        self.globus_pallidus = self.neurons[200:250]
        self.substantia_nigra = self.neurons[250:]
        
        self.rhythm_patterns: Dict[str, List[float]] = {}
        self.prediction_buffer: List[float] = []

    def process(self, spikes: List[SpikeEvent], t: float, dt: float) -> List[SpikeEvent]:
        output_spikes = []
        
        for neuron in self.striatum_neurons:
            for spike in spikes:
                neuron.I_ext += 0.2 * spike.amplitude
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        for neuron in self.globus_pallidus:
            for spike in output_spikes[:50]:
                neuron.I_ext -= 0.3
            new_spike = neuron.update(dt, t)
            if new_spike:
                output_spikes.append(new_spike)
        
        return output_spikes

    def learn_pattern(self, pattern_id: str, sequence: List[float]):
        self.rhythm_patterns[pattern_id] = sequence

    def predict_next(self, current_sequence: List[float]) -> Optional[float]:
        if not current_sequence or len(current_sequence) < 2:
            return None
        
        intervals = [current_sequence[i+1] - current_sequence[i] 
                     for i in range(len(current_sequence)-1)]
        
        if not intervals:
            return None
        
        avg_interval = np.mean(intervals)
        return current_sequence[-1] + avg_interval