#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物启发式听觉脉冲神经网络（Bio-Auditory SNN）

实现自底向上的层级化建模思路：
1. 耳蜗换能模块 - 模拟外周听觉系统（基底膜+毛细胞）
2. 脑干模拟模块 - 耳蜗核+下丘（侧抑制+声源定位）
3. 听觉皮层模块 - 微柱模型（频率拓扑+特征整合）
4. 高级认知模块 - 海马记忆+基底节预测

核心设计原则：
- 拓扑保真：频率拓扑映射
- 时间编码：TTFS/Rate/Synchrony
- 可塑性学习：STDP
- 事件驱动：低功耗计算

版本: 1.0
日期: 2026-07-03
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
import scipy.signal as sp_signal


class EncodingScheme(Enum):
    """脉冲编码方案"""
    TTFS = "ttfs"
    RATE = "rate"
    SYNCHRONY = "synchrony"
    PHASE = "phase"


@dataclass
class SpikeEvent:
    """脉冲事件"""
    neuron_id: int
    timestamp: float
    channel: int
    amplitude: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Spike(neuron={self.neuron_id}, t={self.timestamp:.4f}s, ch={self.channel})"


@dataclass
class AuditoryChannel:
    """听觉通道（对应基底膜位置）"""
    channel_id: int
    center_frequency: float
    bandwidth: float
    spikes: List[SpikeEvent] = field(default_factory=list)


class GammatoneFilterbank:
    """
    伽马通滤波器组 - 模拟基底膜分频特性
    
    Gammatone滤波器是听觉信号处理中模拟耳蜗基底膜特性的标准工具。
    滤波器组覆盖人耳可听频率范围（20Hz-20kHz），通常使用64-128通道。
    
    滤波器冲激响应：
    h(t) = t^(n-1) * exp(-2π * b * t) * cos(2π * fc * t + φ)
    
    参数：
    - fc: 中心频率
    - b: 带宽参数（与ERB相关）
    - n: 滤波器阶数（通常取4）
    """
    
    def __init__(self, num_channels: int = 64, 
                 min_freq: float = 20.0, max_freq: float = 20000.0):
        self.num_channels = num_channels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.channels = []
        self._init_channels()
    
    def _erb_rate(self, freq: float) -> float:
        """ERB（等效矩形带宽）率"""
        return 21.4 * math.log10(1 + freq / 229.0)
    
    def _erb_to_freq(self, erb: float) -> float:
        """ERB率转频率"""
        return 229.0 * (10 ** (erb / 21.4) - 1)
    
    def _init_channels(self):
        """初始化滤波器通道（基于ERB刻度）"""
        min_erb = self._erb_rate(self.min_freq)
        max_erb = self._erb_rate(self.max_freq)
        erb_step = (max_erb - min_erb) / (self.num_channels - 1)
        
        for i in range(self.num_channels):
            erb = min_erb + i * erb_step
            fc = self._erb_to_freq(erb)
            erb_fc = self._erb_rate(fc)
            bw = 1.019 * erb_fc
            
            self.channels.append(AuditoryChannel(
                channel_id=i,
                center_frequency=fc,
                bandwidth=bw
            ))
    
    def _gammatone_impulse_response(self, fc: float, bw: float, 
                                    duration: float, fs: float) -> np.ndarray:
        """生成单个伽马通滤波器的冲激响应"""
        t = np.linspace(0, duration, int(fs * duration))
        n = 4
        
        b = 1.019 * bw
        envelope = t ** (n - 1) * np.exp(-2 * math.pi * b * t)
        carrier = np.cos(2 * math.pi * fc * t)
        
        return envelope * carrier
    
    def filter(self, signal: np.ndarray, fs: float) -> List[np.ndarray]:
        """
        使用伽马通滤波器组滤波音频信号
        
        Args:
            signal: 输入音频信号
            fs: 采样率
            
        Returns:
            各通道滤波后的信号列表
        """
        filtered_signals = []
        duration = len(signal) / fs
        
        for channel in self.channels:
            ir = self._gammatone_impulse_response(
                channel.center_frequency, 
                channel.bandwidth,
                duration, fs
            )
            
            filtered = sp_signal.fftconvolve(signal, ir, mode='same')
            filtered_signals.append(filtered)
        
        return filtered_signals


class InnerHairCellModel:
    """
    内毛细胞换能模型
    
    模拟毛细胞将机械振动转换为神经脉冲的过程：
    1. 半波整流 - 模拟毛细胞去极化
    2. 自适应增益压缩 - 模拟耳蜗的动态范围压缩
    3. 脉冲编码 - TTFS/Rate编码
    """
    
    def __init__(self, sample_rate: float = 44100):
        self.sample_rate = sample_rate
        self.time_resolution = 1.0 / sample_rate
        
        # 自适应增益参数
        self.attack_time = 0.005
        self.release_time = 0.05
        self.compression_ratio = 0.3
        
        # TTFS参数
        self.ttfs_max_delay = 0.02
        self.ttfs_threshold = 0.01
    
    def half_wave_rectify(self, signal: np.ndarray) -> np.ndarray:
        """半波整流 - 只保留正半周"""
        return np.maximum(signal, 0)
    
    def adaptive_gain_control(self, signal: np.ndarray) -> np.ndarray:
        """
        自适应增益压缩
        
        模拟耳蜗的动态范围压缩特性：
        - 强信号：增益降低（压缩）
        - 弱信号：增益提高（放大）
        """
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
        """
        TTFS（Time-to-First-Spike）编码
        
        信号强度越高，首次脉冲发放越早
        完全复现听神经的编码逻辑
        """
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
        """
        发放率编码
        
        在滑动窗口内根据信号强度生成脉冲
        """
        spikes = []
        signal = self.half_wave_rectify(signal)
        signal = self.adaptive_gain_control(signal)
        
        window_samples = int(window_size * self.sample_rate)
        step_samples = window_samples // 2
        
        for i in range(0, len(signal) - window_samples, step_samples):
            window = signal[i:i+window_samples]
            mean_amplitude = np.mean(window)
            
            firing_rate = min(200, mean_amplitude * 400)
            
            if firing_rate > 0:
                spike_interval = 1.0 / firing_rate
                num_spikes = int(window_size / spike_interval)
                
                for j in range(num_spikes):
                    spike_time = (i + j * spike_interval * self.sample_rate) * self.time_resolution
                    if spike_time < len(signal) * self.time_resolution:
                        spikes.append(SpikeEvent(
                            neuron_id=channel_id,
                            timestamp=spike_time,
                            channel=channel_id,
                            amplitude=mean_amplitude,
                            features={'encoding': 'rate', 'firing_rate': firing_rate}
                        ))
        
        return spikes
    
    def encode_phase_locking(self, signal: np.ndarray, channel_id: int, 
                             center_frequency: float = 0.0) -> List[SpikeEvent]:
        """
        相位锁定编码（Phase Locking）
        
        听神经纤维的放电脉冲与声波周期保持同步，每完成一个周期发放一次脉冲。
        信息承载于脉冲相对于振荡周期的时间位置，而非单纯的发放率。
        
        这是音高感知和协和性的生理基础——实验中观测到的FFR（频率跟随响应）
        就是脑干层面神经元脉冲锁相声波的直接证据。
        
        Args:
            signal: 滤波后的通道信号
            channel_id: 通道ID
            center_frequency: 通道中心频率（用于精确计算相位）
            
        Returns:
            脉冲事件列表，每个脉冲的时间精确对齐声波的特定相位
        """
        spikes = []
        
        signal = self.half_wave_rectify(signal)
        signal = self.adaptive_gain_control(signal)
        
        analytic_signal = sp_signal.hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        zero_crossings = np.where(np.logical_and(
            instantaneous_phase[:-1] < 0,
            instantaneous_phase[1:] >= 0
        ))[0]
        
        if len(zero_crossings) == 0 or zero_crossings[0] > 0:
            if signal[0] > 0:
                first_peak = np.argmax(signal[:min(100, len(signal))])
                if first_peak < 50:
                    spikes.append(SpikeEvent(
                        neuron_id=channel_id,
                        timestamp=first_peak * self.time_resolution,
                        channel=channel_id,
                        amplitude=signal[first_peak],
                        features={'encoding': 'phase_locking', 'phase': 0.0}
                    ))
        
        for idx in zero_crossings:
            spike_time = idx * self.time_resolution
            amplitude = np.max(signal[max(0, idx-5):min(len(signal), idx+5)])
            
            if amplitude > self.ttfs_threshold * 0.5:
                exact_phase = instantaneous_phase[idx] % (2 * math.pi)
                
                spikes.append(SpikeEvent(
                    neuron_id=channel_id,
                    timestamp=spike_time,
                    channel=channel_id,
                    amplitude=amplitude,
                    features={
                        'encoding': 'phase_locking', 
                        'phase': float(exact_phase),
                        'frequency': center_frequency
                    }
                ))
        
        return spikes
    
    def encode(self, signal: np.ndarray, channel_id: int, 
               encoding: EncodingScheme = EncodingScheme.TTFS) -> List[SpikeEvent]:
        """
        将信号编码为脉冲序列
        
        Args:
            signal: 滤波后的通道信号
            channel_id: 通道ID
            encoding: 编码方案
            
        Returns:
            脉冲事件列表
        """
        if encoding == EncodingScheme.TTFS:
            return self.encode_ttfs(signal, channel_id)
        elif encoding == EncodingScheme.RATE:
            return self.encode_rate(signal, channel_id)
        elif encoding == EncodingScheme.SYNCHRONY:
            return self.encode_synchrony(signal, channel_id)
        elif encoding == EncodingScheme.PHASE:
            return self.encode_phase_locking(signal, channel_id)
        else:
            return self.encode_ttfs(signal, channel_id)
    
    def encode_synchrony(self, signal: np.ndarray, channel_id: int) -> List[SpikeEvent]:
        """
        同步编码 - 在信号峰值时刻发放脉冲
        """
        spikes = []
        signal = self.half_wave_rectify(signal)
        
        peaks, _ = signal.find_peaks(signal, height=self.ttfs_threshold, distance=10)
        
        for peak_idx in peaks:
            spike_time = peak_idx * self.time_resolution
            amplitude = signal[peak_idx]
            
            spikes.append(SpikeEvent(
                neuron_id=channel_id,
                timestamp=spike_time,
                channel=channel_id,
                amplitude=amplitude,
                features={'encoding': 'synchrony'}
            ))
        
        return spikes


class CochleaModel:
    """
    完整耳蜗模型
    
    集成伽马通滤波器组和毛细胞换能：
    1. 基底膜模拟 - 分频
    2. 毛细胞换能 - 脉冲编码
    """
    
    def __init__(self, num_channels: int = 64, sample_rate: float = 44100):
        self.filterbank = GammatoneFilterbank(num_channels)
        self.hair_cell = InnerHairCellModel(sample_rate)
        self.sample_rate = sample_rate
    
    def process(self, audio_signal: np.ndarray, 
                encoding: EncodingScheme = EncodingScheme.TTFS) -> List[SpikeEvent]:
        """
        将音频信号转换为多通道脉冲序列
        
        Args:
            audio_signal: 输入音频信号
            encoding: 编码方案
            
        Returns:
            所有通道的脉冲事件列表
        """
        filtered_signals = self.filterbank.filter(audio_signal, self.sample_rate)
        
        all_spikes = []
        for channel_id, filtered_signal in enumerate(filtered_signals):
            spikes = self.hair_cell.encode(filtered_signal, channel_id, encoding)
            all_spikes.extend(spikes)
        
        all_spikes.sort(key=lambda s: s.timestamp)
        
        return all_spikes


class LIFNeuron:
    """
    漏电积分发放神经元（Leaky Integrate-and-Fire）
    
    膜电位更新方程：
    τ_m * dV/dt = -(V - V_rest) + R_m * I_syn
    
    当 V >= V_th 时发放脉冲，V重置为V_reset
    """
    
    def __init__(self, neuron_id: int, params: Dict = None):
        self.id = neuron_id
        self.params = params or {}
        
        self.V_rest = self.params.get('V_rest', -60.0)
        self.V_th = self.params.get('V_th', -40.0)
        self.V_reset = self.params.get('V_reset', -60.0)
        self.tau_m = self.params.get('tau_m', 2.0)
        self.tau_ref = self.params.get('tau_ref', 1.0)
        
        self.V = self.V_rest
        self.last_spike_time = -float('inf')
        self.spike_count = 0
        self.refractory = False
        
        self.I_syn = 0.0
        self.I_ext = 0.0
    
    def update(self, dt: float, t: float) -> Optional[SpikeEvent]:
        """
        更新神经元状态
        
        Args:
            dt: 时间步长(ms)
            t: 当前时间(ms)
            
        Returns:
            SpikeEvent if fired, None otherwise
        """
        if t - self.last_spike_time < self.tau_ref:
            self.refractory = True
            self.V = self.V_reset
            return None
        
        self.refractory = False
        
        dV_dt = (self.V_rest - self.V + self.I_syn + self.I_ext) / self.tau_m
        self.V += dV_dt * dt
        
        fired = False
        if self.V >= self.V_th:
            self.V = self.V_reset
            self.last_spike_time = t
            self.spike_count += 1
            fired = True
        
        self.I_syn = 0.0
        
        if fired:
            return SpikeEvent(
                neuron_id=self.id,
                timestamp=t / 1000.0,
                channel=self.id,
                amplitude=1.0,
                features={'neuron_type': 'lif'}
            )
        
        return None
    
    def receive_input(self, weight: float):
        """接收突触输入"""
        self.I_syn += weight


class IzhikevichNeuron:
    """
    Izhikevich神经元模型（更接近生物神经元）
    
    模型方程：
    dV/dt = 0.04*V^2 + 5*V + 140 - u + I
    du/dt = a*(b*V - u)
    
    发放后：V = c, u = u + d
    
    参数配置：
    - RS (规则发放): a=0.02, b=0.2, c=-65, d=8
    - IB (爆发发放): a=0.02, b=0.2, c=-55, d=4
    - FS (快速发放): a=0.1, b=0.2, c=-65, d=2
    """
    
    def __init__(self, neuron_id: int, params: Dict = None):
        self.id = neuron_id
        self.params = params or {}
        
        # 默认RS神经元参数
        self.a = self.params.get('a', 0.05)
        self.b = self.params.get('b', 0.25)
        self.c = self.params.get('c', -60.0)
        self.d = self.params.get('d', 10.0)
        
        self.V = self.c
        self.u = self.b * self.V
        
        self.last_spike_time = -float('inf')
        self.spike_count = 0
        
        self.I_syn = 0.0
        self.I_ext = 0.0
    
    def update(self, dt: float, t: float) -> Optional[SpikeEvent]:
        """
        更新神经元状态
        
        Args:
            dt: 时间步长(ms)
            t: 当前时间(ms)
            
        Returns:
            SpikeEvent if fired, None otherwise
        """
        dV_dt = 0.04 * self.V ** 2 + 5.0 * self.V + 140.0 - self.u + self.I_syn + self.I_ext
        du_dt = self.a * (self.b * self.V - self.u)
        
        self.V += dV_dt * dt * 0.5
        self.u += du_dt * dt * 0.5
        
        fired = False
        if self.V >= 30.0:
            self.V = self.c
            self.u += self.d
            self.last_spike_time = t
            self.spike_count += 1
            fired = True
        
        self.I_syn = 0.0
        
        if fired:
            return SpikeEvent(
                neuron_id=self.id,
                timestamp=t / 1000.0,
                channel=self.id,
                amplitude=1.0,
                features={'neuron_type': 'izhikevich'}
            )
        
        return None
    
    def receive_input(self, weight: float):
        """接收突触输入"""
        self.I_syn += weight


class CochlearNucleus:
    """
    耳蜗核模型 - 脑干低层处理
    
    功能：
    1. 使用LIF/Izhikevich神经元处理听神经输入
    2. 侧抑制连接，强化频率对比度
    3. 提取声音边界和时域特征
    """
    
    def __init__(self, num_neurons: int = 128):
        self.num_neurons = num_neurons
        self.neurons = [LIFNeuron(i) for i in range(num_neurons)]
        
        # 侧抑制连接矩阵
        self.lateral_inhibition = np.zeros((num_neurons, num_neurons))
        self._init_lateral_inhibition()
        
        self.output_spikes: List[SpikeEvent] = []
    
    def _init_lateral_inhibition(self):
        """
        初始化侧抑制连接
        
        同层神经元之间的抑制连接，距离越近抑制越强
        模拟耳蜗核的频率对比度增强机制
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    distance = abs(i - j)
                    self.lateral_inhibition[i, j] = np.exp(-distance ** 2 / 8.0) * 0.5
    
    def process(self, input_spikes: List[SpikeEvent], t: float, dt: float = 1.0) -> List[SpikeEvent]:
        """
        处理输入脉冲，应用侧抑制
        
        Args:
            input_spikes: 输入脉冲事件
            t: 当前时间(ms)
            dt: 时间步长(ms)
            
        Returns:
            输出脉冲列表
        """
        self.output_spikes = []
        
        for spike in input_spikes:
            if spike.channel < self.num_neurons:
                self.neurons[spike.channel].receive_input(50.0)
            
            for i, neuron in enumerate(self.neurons):
                if i == spike.channel:
                    continue
                distance = abs(i - spike.channel)
                if distance <= 3:
                    neuron.I_syn -= 10.0 / (distance + 1)
        
        for i, neuron in enumerate(self.neurons):
            spike_out = neuron.update(dt, t)
            if spike_out:
                spike_out.channel = i
                self.output_spikes.append(spike_out)
        
        return self.output_spikes


class InferiorColliculus:
    """
    下丘模型 - 声源定位
    
    功能：
    1. 双耳时间差（ITD）神经元实现声源定位
    2. 整合来自耳蜗核的输入
    3. 提取空间听觉特征
    """
    
    def __init__(self, num_neurons: int = 64):
        self.num_neurons = num_neurons
        self.neurons = [IzhikevichNeuron(i, {'a': 0.02, 'b': 0.2, 'c': -55.0, 'd': 4}) 
                       for i in range(num_neurons)]
        
        # ITD（双耳时间差）神经元配置
        self.itd_neurons = []
        self._init_itd_neurons()
        
        self.output_spikes: List[SpikeEvent] = []
    
    def _init_itd_neurons(self):
        """初始化ITD神经元"""
        # 创建覆盖不同时间差的ITD神经元
        itd_ranges = np.linspace(-1.0, 1.0, 16)
        for i, itd in enumerate(itd_ranges):
            neuron = IzhikevichNeuron(self.num_neurons + i, {'a': 0.05, 'b': 0.25, 'c': -60.0, 'd': 6})
            neuron.itd_preference = itd
            self.itd_neurons.append(neuron)
    
    def process(self, left_spikes: List[SpikeEvent], right_spikes: List[SpikeEvent],
                t: float, dt: float = 1.0) -> List[SpikeEvent]:
        """
        处理双耳输入，计算声源定位
        
        Args:
            left_spikes: 左耳输入脉冲
            right_spikes: 右耳输入脉冲
            t: 当前时间(ms)
            dt: 时间步长(ms)
            
        Returns:
            输出脉冲列表
        """
        self.output_spikes = []
        
        for spike in left_spikes + right_spikes:
            idx = spike.channel % self.num_neurons
            self.neurons[idx].receive_input(30.0)
        
        for neuron in self.neurons:
            spike = neuron.update(dt, t)
            if spike:
                self.output_spikes.append(spike)
        
        for itd_neuron in self.itd_neurons:
            itd_neuron.receive_input(15.0)
            spike = itd_neuron.update(dt, t)
            if spike:
                spike.features['itd_preference'] = itd_neuron.itd_preference
                self.output_spikes.append(spike)
        
        return self.output_spikes


class CorticalMicrocolumn:
    """
    皮层微柱模型 - 听觉皮层中层处理
    
    包含兴奋性锥体神经元和抑制性中间神经元，
    模拟初级听觉皮层的频率拓扑与特征整合。
    """
    
    def __init__(self, column_id: int, num_excitatory: int = 8, num_inhibitory: int = 2):
        self.column_id = column_id
        
        # 兴奋性锥体神经元
        self.excitatory_neurons = [IzhikevichNeuron(
            column_id * 10 + i,
            {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8}
        ) for i in range(num_excitatory)]
        
        # 抑制性中间神经元
        self.inhibitory_neurons = [IzhikevichNeuron(
            column_id * 10 + num_excitatory + i,
            {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2}
        ) for i in range(num_inhibitory)]
        
        # 柱内连接权重
        self.ee_weight = 0.5
        self.ei_weight = 1.0
        self.ie_weight = 1.5
    
    def update(self, t: float, dt: float = 1.0, external_input: float = 0.0) -> List[SpikeEvent]:
        """
        更新微柱状态
        
        Args:
            t: 当前时间(ms)
            dt: 时间步长(ms)
            external_input: 外部输入电流
            
        Returns:
            输出脉冲列表
        """
        spikes = []
        
        # 首先更新兴奋性神经元
        for exc_neuron in self.excitatory_neurons:
            exc_neuron.I_ext += external_input
            spike = exc_neuron.update(dt, t)
            if spike:
                spikes.append(spike)
        
        # 兴奋性神经元输出反馈到微柱
        exc_output = len(spikes)
        
        # 更新抑制性神经元（接收兴奋性输入）
        for inh_neuron in self.inhibitory_neurons:
            inh_neuron.I_ext += exc_output * self.ei_weight
            spike = inh_neuron.update(dt, t)
            if spike:
                spikes.append(spike)
        
        # 抑制性神经元反馈抑制兴奋性神经元
        inh_output = sum(1 for n in self.inhibitory_neurons if n.V > -55.0)
        
        for exc_neuron in self.excitatory_neurons:
            exc_neuron.I_syn -= inh_output * self.ie_weight
        
        return spikes


class AuditoryCortex:
    """
    听觉皮层模型 - 中级处理
    
    构建具有拓扑结构的皮层微柱阵列，
    通过同层抑制、邻层兴奋的连接模式，
    自动提取音高变化、节奏间隔等基础模式。
    """
    
    def __init__(self, num_columns: int = 32):
        self.num_columns = num_columns
        self.columns = [CorticalMicrocolumn(i) for i in range(num_columns)]
        
        # 柱间连接（侧向兴奋+长距离抑制）
        self.column_connections = np.zeros((num_columns, num_columns))
        self._init_column_connections()
        
        self.output_spikes: List[SpikeEvent] = []
    
    def _init_column_connections(self):
        """初始化柱间连接"""
        for i in range(self.num_columns):
            for j in range(self.num_columns):
                distance = abs(i - j)
                if distance == 0:
                    continue
                elif distance <= 2:
                    # 邻近柱：兴奋
                    self.column_connections[i, j] = 0.3
                elif distance <= 5:
                    # 中距离：弱抑制
                    self.column_connections[i, j] = -0.1
                else:
                    # 长距离：强抑制
                    self.column_connections[i, j] = -0.05
    
    def process(self, input_spikes: List[SpikeEvent], t: float, dt: float = 1.0) -> List[SpikeEvent]:
        """
        处理输入脉冲，提取听觉特征
        
        Args:
            input_spikes: 输入脉冲事件
            t: 当前时间(ms)
            dt: 时间步长(ms)
            
        Returns:
            输出脉冲列表
        """
        self.output_spikes = []
        
        column_inputs = np.zeros(self.num_columns)
        for spike in input_spikes:
            col_idx = min(spike.channel // 2, self.num_columns - 1)
            column_inputs[col_idx] += 20.0
        
        all_spikes = []
        for i, column in enumerate(self.columns):
            lateral_input = np.sum(self.column_connections[i, :] * column_inputs)
            spikes = column.update(t, dt, column_inputs[i] + lateral_input)
            all_spikes.extend(spikes)
        
        self.output_spikes = all_spikes
        return all_spikes


class HippocampalMemory:
    """
    海马记忆系统模型
    
    负责音乐时序记忆存储：
    - 通过STDP学习旋律的先后顺序
    - 形成链式脉冲序列记忆
    - 实现"听到片段回忆全曲"的联想功能
    """
    
    def __init__(self, num_neurons: int = 200):
        self.num_neurons = num_neurons
        self.neurons = [IzhikevichNeuron(i, {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8}) 
                       for i in range(num_neurons)]
        
        # STDP突触连接
        self.synapses = []
        self._init_stdp_synapses()
        
        # 时序记忆缓冲区
        self.memory_traces: Dict[str, List[SpikeEvent]] = {}
        self.current_sequence: List[SpikeEvent] = []
    
    def _init_stdp_synapses(self):
        """初始化STDP突触"""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < 0.1:
                    synapse = STDPSynapse(i, j, weight=np.random.uniform(0.1, 0.5))
                    self.synapses.append(synapse)
    
    def process(self, input_spikes: List[SpikeEvent], t: float, dt: float = 1.0) -> List[SpikeEvent]:
        """
        处理输入脉冲，更新记忆
        
        Args:
            input_spikes: 输入脉冲事件
            t: 当前时间(ms)
            dt: 时间步长(ms)
            
        Returns:
            输出脉冲列表
        """
        output_spikes = []
        
        # 更新时序记忆
        self.current_sequence.extend(input_spikes)
        if len(self.current_sequence) > 100:
            self.current_sequence = self.current_sequence[-100:]
        
        # 激活神经元
        for spike in input_spikes:
            idx = spike.channel % self.num_neurons
            self.neurons[idx].receive_input(10.0)
        
        # 更新神经元和STDP
        for neuron in self.neurons:
            spike = neuron.update(dt, t)
            if spike:
                output_spikes.append(spike)
        
        # STDP学习
        self._apply_stdp(t)
        
        return output_spikes
    
    def _apply_stdp(self, t: float):
        """应用STDP学习规则"""
        for synapse in self.synapses:
            pre_time = self.neurons[synapse.pre_neuron].last_spike_time
            post_time = self.neurons[synapse.post_neuron].last_spike_time
            
            if pre_time > 0 and post_time > 0:
                delta_t = post_time - pre_time
                synapse.update_weight(delta_t)
    
    def store_sequence(self, sequence_id: str):
        """存储当前序列到长时记忆"""
        if self.current_sequence:
            self.memory_traces[sequence_id] = list(self.current_sequence)
    
    def retrieve_sequence(self, sequence_id: str) -> Optional[List[SpikeEvent]]:
        """从长时记忆检索序列"""
        return self.memory_traces.get(sequence_id)
    
    def associate(self, partial_sequence: List[SpikeEvent]) -> Optional[List[SpikeEvent]]:
        """
        联想检索 - 根据片段回忆完整序列
        
        Args:
            partial_sequence: 部分脉冲序列
            
        Returns:
            完整序列（如果找到匹配）
        """
        if not partial_sequence:
            return None
        
        # 简单的匹配算法：比较前几个脉冲的时间模式
        partial_times = [s.timestamp for s in partial_sequence[:5]]
        
        for seq_id, full_sequence in self.memory_traces.items():
            full_times = [s.timestamp for s in full_sequence[:5]]
            
            # 比较时间差模式
            if len(full_times) >= len(partial_times):
                matches = True
                for i in range(len(partial_times) - 1):
                    partial_diff = partial_times[i+1] - partial_times[i]
                    full_diff = full_times[i+1] - full_times[i]
                    if abs(partial_diff - full_diff) > 0.01:
                        matches = False
                        break
                
                if matches:
                    return full_sequence
        
        return None


class BasalGanglia:
    """
    基底节-丘脑环路模型
    
    负责节奏预测和运动规划：
    - 学习和预测音乐节奏模式
    - 选择适当的运动输出（演奏动作）
    """
    
    def __init__(self, num_neurons: int = 100):
        self.num_neurons = num_neurons
        
        # 纹状体神经元（抑制性）
        self.striatum = [IzhikevichNeuron(i, {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2}) 
                        for i in range(num_neurons // 2)]
        
        # 苍白球神经元（抑制性）
        self.globus_pallidus = [IzhikevichNeuron(num_neurons // 2 + i, 
                                                 {'a': 0.05, 'b': 0.25, 'c': -60.0, 'd': 3}) 
                               for i in range(num_neurons // 4)]
        
        # 丘脑神经元（兴奋性）
        self.thalamus = [IzhikevichNeuron(num_neurons * 3 // 4 + i, 
                                          {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8}) 
                         for i in range(num_neurons // 4)]
        
        # 预测缓冲区
        self.prediction_buffer: List[float] = []
        self.rhythm_patterns: Dict[str, List[float]] = {}
    
    def process(self, input_spikes: List[SpikeEvent], t: float, dt: float = 1.0) -> List[SpikeEvent]:
        """
        处理输入脉冲，进行节奏预测
        
        Args:
            input_spikes: 输入脉冲事件
            t: 当前时间(ms)
            dt: 时间步长(ms)
            
        Returns:
            输出脉冲列表
        """
        output_spikes = []
        
        # 更新纹状体
        for spike in input_spikes:
            idx = spike.channel % len(self.striatum)
            self.striatum[idx].receive_input(8.0)
        
        striatum_spikes = []
        for neuron in self.striatum:
            spike = neuron.update(dt, t)
            if spike:
                striatum_spikes.append(spike)
        
        # 更新苍白球（接收纹状体抑制输入）
        for neuron in self.globus_pallidus:
            neuron.I_syn -= len(striatum_spikes) * 0.5
            spike = neuron.update(dt, t)
            if spike:
                output_spikes.append(spike)
        
        # 更新丘脑（接收苍白球抑制输入）
        for neuron in self.thalamus:
            neuron.I_syn -= len(self.globus_pallidus) * 0.3
            # 添加预测输入
            if self.prediction_buffer and t / 1000.0 > self.prediction_buffer[0]:
                neuron.I_syn += 5.0
                self.prediction_buffer = self.prediction_buffer[1:]
            
            spike = neuron.update(dt, t)
            if spike:
                output_spikes.append(spike)
        
        return output_spikes
    
    def learn_rhythm(self, rhythm_pattern: List[float]):
        """学习节奏模式"""
        pattern_id = str(hash(tuple(rhythm_pattern)))
        self.rhythm_patterns[pattern_id] = rhythm_pattern
    
    def predict_next(self, current_sequence: List[float]) -> Optional[float]:
        """
        预测下一个节奏事件的时间
        
        Args:
            current_sequence: 当前节奏序列的时间点
            
        Returns:
            预测的下一个时间点
        """
        if not current_sequence or len(current_sequence) < 2:
            return None
        
        # 查找匹配的节奏模式
        for pattern in self.rhythm_patterns.values():
            if len(pattern) <= len(current_sequence):
                continue
            
            # 比较最后几个时间差
            recent_diffs = np.diff(current_sequence[-3:])
            pattern_diffs = np.diff(pattern[:len(recent_diffs) + 1])
            
            if np.allclose(recent_diffs, pattern_diffs, atol=0.05):
                # 预测下一个时间点
                next_diff = pattern[len(recent_diffs) + 1] - pattern[len(recent_diffs)]
                return current_sequence[-1] + next_diff
        
        return None


class STDPSynapse:
    """
    STDP（Spike-Timing-Dependent Plasticity）突触
    
    规则逻辑：
    - 若突触前神经元脉冲先于突触后神经元发放（Δt > 0）→ LTP（权重增强）
    - 若突触后脉冲在前（Δt < 0）→ LTD（权重减弱）
    
    数学公式：
    Δw = A₊·exp(-Δt/τ₊)  当 Δt > 0 (LTP)
    Δw = -A₋·exp(Δt/τ₋)  当 Δt < 0 (LTD)
    """
    
    def __init__(self, pre_neuron: int, post_neuron: int, weight: float = 0.5):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        
        # STDP参数
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        
        # 权重边界
        self.w_min = 0.01
        self.w_max = 1.0
        
        # 学习统计
        self.ltp_count = 0
        self.ltd_count = 0
    
    def update_weight(self, delta_t: float):
        """
        根据脉冲时序差更新权重
        
        Args:
            delta_t: 时序差 (t_post - t_pre)
        """
        if delta_t > 0:
            # LTP: 前突触先发放
            dw = self.A_plus * np.exp(-delta_t / self.tau_plus)
            self.ltp_count += 1
        else:
            # LTD: 后突触先发放
            dw = -self.A_minus * np.exp(delta_t / self.tau_minus)
            self.ltd_count += 1
        
        self.weight += dw
        self.weight = max(self.w_min, min(self.w_max, self.weight))
    
    def get_stats(self) -> Dict:
        """获取学习统计"""
        return {
            'weight': self.weight,
            'ltp_count': self.ltp_count,
            'ltd_count': self.ltd_count,
            'ltp_ratio': self.ltp_count / max(1, self.ltp_count + self.ltd_count)
        }


class BioAuditorySNN:
    """
    完整的生物启发式听觉脉冲神经网络
    
    层级结构：
    1. 耳蜗换能模块（CochleaModel）
    2. 耳蜗核（CochlearNucleus）
    3. 下丘（InferiorColliculus）
    4. 听觉皮层（AuditoryCortex）
    5. 高级认知区（HippocampalMemory + BasalGanglia）
    
    核心设计原则：
    - 拓扑保真：频率拓扑映射
    - 时间编码：TTFS/Rate/Synchrony
    - 可塑性学习：STDP
    - 事件驱动：低功耗计算
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 配置参数
        self.num_channels = self.config.get('num_channels', 64)
        self.sample_rate = self.config.get('sample_rate', 44100)
        self.encoding = self.config.get('encoding', EncodingScheme.TTFS)
        
        # 层级模块
        self.cochlea = CochleaModel(self.num_channels, self.sample_rate)
        self.cochlear_nucleus = CochlearNucleus(self.num_channels)
        self.inferior_colliculus = InferiorColliculus(self.num_channels // 2)
        self.auditory_cortex = AuditoryCortex(self.num_channels // 4)
        self.hippocampus = HippocampalMemory(200)
        self.basal_ganglia = BasalGanglia(100)
        
        # 处理状态
        self.t = 0.0
        self.dt = 1.0
        
        # 脉冲历史
        self.spike_history: List[SpikeEvent] = []
        
        # 学习标志
        self.learning_enabled = True
    
    def process_audio(self, audio_signal: np.ndarray) -> Dict[str, List[SpikeEvent]]:
        """
        处理音频信号，经过完整的听觉通路
        
        Args:
            audio_signal: 输入音频信号
            
        Returns:
            各层级的脉冲输出
        """
        results = {}
        
        # 第一步：耳蜗换能
        cochlea_spikes = self.cochlea.process(audio_signal, self.encoding)
        results['cochlea'] = cochlea_spikes
        
        # 第二步：耳蜗核处理（侧抑制）
        cn_spikes = []
        current_t = 0.0
        for spike in cochlea_spikes:
            t_ms = spike.timestamp * 1000
            while current_t < t_ms:
                current_t += self.dt
            
            cn_spikes.extend(self.cochlear_nucleus.process([spike], current_t, self.dt))
        results['cochlear_nucleus'] = cn_spikes
        
        # 第三步：下丘处理（声源定位）
        ic_spikes = []
        current_t = 0.0
        for spike in cn_spikes:
            t_ms = spike.timestamp * 1000 if hasattr(spike, 'timestamp') else current_t
            while current_t < t_ms:
                current_t += self.dt
            
            ic_spikes.extend(self.inferior_colliculus.process([spike], [spike], current_t, self.dt))
        results['inferior_colliculus'] = ic_spikes
        
        # 第四步：听觉皮层处理（特征提取）
        cortex_spikes = []
        current_t = 0.0
        for spike in ic_spikes:
            t_ms = getattr(spike, 'timestamp', 0) * 1000 if getattr(spike, 'timestamp', 0) > 0 else current_t
            while current_t < t_ms:
                current_t += self.dt
            
            cortex_spikes.extend(self.auditory_cortex.process([spike], current_t, self.dt))
        results['auditory_cortex'] = cortex_spikes
        
        # 第五步：高级认知处理（记忆+预测）
        hippocampus_spikes = []
        bg_spikes = []
        current_t = 0.0
        for spike in cortex_spikes:
            t_ms = getattr(spike, 'timestamp', 0) * 1000 if getattr(spike, 'timestamp', 0) > 0 else current_t
            while current_t < t_ms:
                current_t += self.dt
            
            hippocampus_spikes.extend(self.hippocampus.process([spike], current_t, self.dt))
            bg_spikes.extend(self.basal_ganglia.process([spike], current_t, self.dt))
        results['hippocampus'] = hippocampus_spikes
        results['basal_ganglia'] = bg_spikes
        
        self.spike_history.extend(cochlea_spikes)
        
        return results
    
    def learn_melody(self, melody_id: str, audio_signal: np.ndarray):
        """
        学习一段旋律并存储到海马记忆
        
        Args:
            melody_id: 旋律ID
            audio_signal: 旋律音频信号
        """
        results = self.process_audio(audio_signal)
        self.hippocampus.store_sequence(melody_id)
        
        # 提取节奏模式用于基底节学习
        cortex_spikes = results.get('auditory_cortex', [])
        if cortex_spikes:
            rhythm_pattern = [s.timestamp for s in cortex_spikes]
            self.basal_ganglia.learn_rhythm(rhythm_pattern)
    
    def recall_melody(self, partial_audio: np.ndarray) -> Optional[List[SpikeEvent]]:
        """
        根据片段回忆完整旋律
        
        Args:
            partial_audio: 片段音频
            
        Returns:
            完整序列的脉冲事件（如果找到匹配）
        """
        results = self.process_audio(partial_audio)
        cortex_spikes = results.get('auditory_cortex', [])
        
        return self.hippocampus.associate(cortex_spikes)
    
    def predict_rhythm(self, current_sequence: List[float]) -> Optional[float]:
        """
        预测下一个节奏事件
        
        Args:
            current_sequence: 当前节奏序列的时间点
            
        Returns:
            预测的下一个时间点
        """
        return self.basal_ganglia.predict_next(current_sequence)
    
    def get_network_state(self) -> Dict:
        """获取网络状态"""
        return {
            'cochlea_channels': self.num_channels,
            'sample_rate': self.sample_rate,
            'encoding': self.encoding.value,
            'total_spikes': len(self.spike_history),
            'learning_enabled': self.learning_enabled,
            'hippocampus_memory_count': len(self.hippocampus.memory_traces),
            'basal_ganglia_patterns': len(self.basal_ganglia.rhythm_patterns)
        }


if __name__ == "__main__":
    print("生物启发式听觉SNN初始化...")
    
    snn = BioAuditorySNN({
        'num_channels': 64,
        'sample_rate': 44100,
        'encoding': EncodingScheme.TTFS
    })
    
    print(f"网络状态: {snn.get_network_state()}")
    
    # 生成测试音频信号（1秒440Hz正弦波）
    duration = 1.0
    t = np.linspace(0, duration, int(snn.sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # 测试处理流程
    print("\n测试音频处理流程...")
    results = snn.process_audio(test_signal)
    
    for layer, spikes in results.items():
        print(f"  {layer}: {len(spikes)} 个脉冲")
    
    # 测试学习功能
    print("\n测试旋律学习功能...")
    snn.learn_melody("test_melody", test_signal)
    print(f"学习后网络状态: {snn.get_network_state()}")
    
    # 测试回忆功能
    print("\n测试旋律回忆功能...")
    partial_signal = test_signal[:int(snn.sample_rate * 0.3)]
    recalled = snn.recall_melody(partial_signal)
    if recalled:
        print(f"  成功回忆到 {len(recalled)} 个脉冲")
    else:
        print("  未找到匹配的记忆")
    
    print("\n生物启发式听觉SNN测试完成！")