"""
reflex/spiking_core.py - SNN脉冲神经网络核心

实现类脑脉冲神经网络，支持STDP无监督学习、时序预测、事件驱动休眠
"""
import numpy as np
from collections import deque
from typing import Dict


class LIFNeuron:
    def __init__(self, tau_mem=20.0, tau_syn=5.0, threshold=1.0, reset=0.0, refractory_period=5.0):
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset = reset
        self.refractory_period = refractory_period
        self.membrane_potential = 0.0
        self.synaptic_current = 0.0
        self.last_spike_time = -np.inf
        self.spike_history = deque(maxlen=100)
        self.refractory_timer = 0.0

    def forward(self, input_current, dt=1.0):
        self.refractory_timer = max(0.0, self.refractory_timer - dt)
        
        self.synaptic_current = self.synaptic_current * np.exp(-dt / self.tau_syn) + input_current
        
        dV = (-self.membrane_potential + self.synaptic_current) / self.tau_mem * dt
        
        if self.refractory_timer > 0:
            dV = 0.0
        
        self.membrane_potential += dV
        
        spike = 0.0
        if self.membrane_potential >= self.threshold:
            spike = 1.0
            self.membrane_potential = self.reset
            self.last_spike_time = 0
            self.refractory_timer = self.refractory_period
            self.spike_history.append(1)
        else:
            self.last_spike_time += dt
            self.spike_history.append(0)
        
        return spike

    def get_spike_rate(self, window=20):
        recent = list(self.spike_history)[-window:]
        if not recent:
            return 0.0
        return sum(recent) / window


class STDPSynapse:
    def __init__(self, weight=0.5, tau_plus=20.0, tau_minus=20.0, a_plus=0.01, a_minus=0.01, w_min=0.0, w_max=1.0):
        self.weight = weight
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.pre_last_spike = -np.inf
        self.post_last_spike = -np.inf
        self.weight_history = deque(maxlen=50)

    def update_weight(self, pre_spike, post_spike, dt=1.0):
        pre_spike = float(pre_spike)
        post_spike = float(post_spike)
        
        if pre_spike > 0.5:
            self.pre_last_spike = 0
        else:
            self.pre_last_spike += dt
        
        if post_spike > 0.5:
            self.post_last_spike = 0
        else:
            self.post_last_spike += dt
        
        delta_w = 0.0
        
        if pre_spike > 0.5 and post_spike <= 0.5:
            delta_w -= self.a_minus * np.exp(-self.post_last_spike / self.tau_minus)
        elif post_spike > 0.5 and pre_spike <= 0.5:
            delta_w += self.a_plus * np.exp(-self.pre_last_spike / self.tau_plus)
        
        self.weight = max(self.w_min, min(self.w_max, self.weight + delta_w))
        self.weight_history.append(self.weight)
        
        return delta_w


class SpikingLayer:
    def __init__(self, input_dim=16, output_dim=32, tau_mem=20.0, tau_syn=5.0, threshold=1.0, learning_enabled=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_enabled = learning_enabled
        
        self.neurons = [LIFNeuron(tau_mem, tau_syn, threshold) for _ in range(output_dim)]
        self.synapses = [[STDPSynapse(np.random.uniform(0.1, 0.5)) 
                          for _ in range(output_dim)] 
                         for _ in range(input_dim)]
        
        self.output_spikes = np.zeros(output_dim)
        self.activation_history = deque(maxlen=10)
    
    def calibrate_for_foundation(self, input_patterns):
        """
        冷启动筑基期初始化校准
        
        预设基础权重，确保标准输入能稳定触发输出脉冲
        """
        for pattern_idx, pattern in enumerate(input_patterns):
            pattern = np.array(pattern).flatten()
            if len(pattern) != self.input_dim:
                continue
            
            target_neuron = pattern_idx % self.output_dim
            for j in range(self.input_dim):
                if abs(pattern[j]) > 0.3:
                    self.synapses[j][target_neuron].weight = min(1.0, self.synapses[j][target_neuron].weight + 0.3)
        
        for i in range(self.output_dim):
            self.neurons[i].threshold = 0.8

    def forward(self, input_signal, dt=1.0):
        input_signal = np.array(input_signal).flatten()
        self.output_spikes = np.zeros(self.output_dim)
        
        for i in range(self.output_dim):
            current = 0.0
            for j in range(min(self.input_dim, len(input_signal))):
                current += input_signal[j] * self.synapses[j][i].weight
            
            spike = self.neurons[i].forward(current, dt)
            self.output_spikes[i] = spike
        
        self.activation_history.append(self.output_spikes.copy())
        
        return self.output_spikes

    def learn(self, pre_spikes, post_spikes, dt=1.0):
        if not self.learning_enabled:
            return 0.0
        
        total_delta = 0.0
        for i in range(self.output_dim):
            for j in range(min(self.input_dim, len(pre_spikes))):
                delta = self.synapses[j][i].update_weight(pre_spikes[j], post_spikes[i], dt)
                total_delta += abs(delta)
        return total_delta / (self.input_dim * self.output_dim)

    def get_weights(self):
        return [[self.synapses[i][j].weight for j in range(self.output_dim)] 
                for i in range(self.input_dim)]

    def set_weights(self, weights):
        for i in range(min(self.input_dim, len(weights))):
            for j in range(min(self.output_dim, len(weights[i]))):
                self.synapses[i][j].weight = weights[i][j]

    def get_layer_activity(self):
        return float(np.mean([n.get_spike_rate() for n in self.neurons]))


class SpikingCore:
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16, learning_enabled=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_enabled = learning_enabled
        
        self.layer1 = SpikingLayer(input_dim, hidden_dim, learning_enabled=learning_enabled)
        self.layer2 = SpikingLayer(hidden_dim, output_dim, learning_enabled=learning_enabled)
        
        self.spike_history = deque(maxlen=50)
        self.timesteps = 5
        
        self._is_active = True
        self._sleep_mode = False
        self._alert_threshold = 0.3

    def forward(self, input_signal, dt=1.0):
        if self._sleep_mode:
            alert_signal = np.mean(np.abs(input_signal))
            if alert_signal > self._alert_threshold:
                self.wake_up()
            else:
                return np.zeros(self.output_dim)
        
        hidden_spikes = np.zeros(self.hidden_dim)
        output_spikes = np.zeros(self.output_dim)
        
        for t in range(self.timesteps):
            hidden_spikes += self.layer1.forward(input_signal if t == 0 else hidden_spikes, dt)
            output_spikes += self.layer2.forward(hidden_spikes, dt)
        
        hidden_spikes = hidden_spikes / self.timesteps
        output_spikes = output_spikes / self.timesteps
        
        self.spike_history.append({
            "input": input_signal,
            "hidden": hidden_spikes,
            "output": output_spikes,
            "timestamp": time.time()
        })
        
        return output_spikes

    def learn(self, dt=1.0):
        if not self.learning_enabled or len(self.spike_history) < 2:
            return 0.0
        
        recent = list(self.spike_history)[-2:]
        pre_hidden = recent[0]["hidden"]
        post_hidden = recent[1]["hidden"]
        pre_output = recent[0]["output"]
        post_output = recent[1]["output"]
        
        delta1 = self.layer1.learn(recent[0]["input"], post_hidden, dt)
        delta2 = self.layer2.learn(pre_hidden, post_output, dt)
        
        return (delta1 + delta2) / 2

    def predict_next(self, input_signal, steps=3):
        predictions = []
        current = np.array(input_signal).flatten()
        
        for _ in range(steps):
            pred = self.forward(current)
            predictions.append(pred)
            current = pred
        
        return predictions

    def detect_anomaly(self, input_signal, threshold=3.0):
        output = self.forward(input_signal)
        
        if not self.spike_history:
            return 0.0, output
        
        recent_outputs = [h["output"] for h in list(self.spike_history)[-10:]]
        if not recent_outputs:
            return 0.0, output
        
        mean_output = np.mean(recent_outputs, axis=0)
        std_output = np.std(recent_outputs, axis=0) + 1e-8
        
        z_scores = np.abs(output - mean_output) / std_output
        anomaly_score = np.mean(z_scores)
        
        return anomaly_score, output

    def enter_sleep_mode(self):
        self._sleep_mode = True
        self._is_active = False

    def wake_up(self):
        self._sleep_mode = False
        self._is_active = True

    def is_active(self):
        return self._is_active

    def get_spike_rate(self):
        if not self.spike_history:
            return 0.0
        
        total_spikes = float(sum(np.sum(h["output"]) for h in self.spike_history))
        return total_spikes / (len(self.spike_history) * self.output_dim)

    def get_activity_level(self):
        return {
            "layer1_activity": self.layer1.get_layer_activity(),
            "layer2_activity": self.layer2.get_layer_activity(),
            "overall_spike_rate": self.get_spike_rate(),
            "is_active": self.is_active()
        }

    def save_weights(self, path):
        weights = {
            "layer1": self.layer1.get_weights(),
            "layer2": self.layer2.get_weights()
        }
        np.savez(path, **weights)

    def load_weights(self, path):
        data = np.load(path)
        if "layer1" in data:
            self.layer1.set_weights(data["layer1"])
        if "layer2" in data:
            self.layer2.set_weights(data["layer2"])

    def update_weights(self, updates: Dict[str, float]):
        """
        更新突触权重

        Args:
            updates: 连接ID到权重增量的映射，例如 {"synapse_0": 0.01, "synapse_1": -0.015}
                     连接ID格式为 "synapse_{index}"，index 为突触在层内的扁平索引
        """
        layers = [self.layer1, self.layer2]
        for connection_id, weight_delta in updates.items():
            try:
                idx = int(str(connection_id).split("_")[-1])
            except (ValueError, IndexError):
                continue

            for layer in layers:
                input_dim = layer.input_dim
                output_dim = layer.output_dim
                total = input_dim * output_dim
                if 0 <= idx < total:
                    i = idx % input_dim
                    j = idx // input_dim
                    if i < input_dim and j < output_dim:
                        synapse = layer.synapses[i][j]
                        synapse.weight = max(synapse.w_min, min(synapse.w_max, synapse.weight + weight_delta))


import time
