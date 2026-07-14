import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ..config.settings import DEVICE


class LIFNeuron:
    def __init__(self, tau_mem=20.0, tau_syn=5.0, threshold=1.0, reset=0.0):
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.reset = reset
        self.membrane_potential = 0.0
        self.synaptic_current = 0.0
        self.last_spike_time = -np.inf
        self.spike_history = deque(maxlen=100)

    def forward(self, input_current, dt=1.0):
        self.synaptic_current = self.synaptic_current * np.exp(-dt / self.tau_syn) + input_current
        
        dV = (-self.membrane_potential + self.synaptic_current) / self.tau_mem * dt
        self.membrane_potential += dV
        
        spike = 0.0
        if self.membrane_potential >= self.threshold:
            spike = 1.0
            self.membrane_potential = self.reset
            self.last_spike_time = 0
            self.spike_history.append(1)
        else:
            self.last_spike_time += dt
            self.spike_history.append(0)
        
        return spike


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

    def update_weight(self, pre_spike, post_spike, dt=1.0):
        if isinstance(pre_spike, np.ndarray):
            pre_spike = float(pre_spike.flat[0]) if pre_spike.size > 0 else 0.0
        else:
            pre_spike = float(pre_spike)
        
        if isinstance(post_spike, np.ndarray):
            post_spike = float(post_spike.flat[0]) if post_spike.size > 0 else 0.0
        else:
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
        
        return delta_w


class SpikingLayer:
    def __init__(self, input_dim=16, output_dim=32, tau_mem=20.0, tau_syn=5.0, threshold=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.neurons = [LIFNeuron(tau_mem, tau_syn, threshold) for _ in range(output_dim)]
        self.synapses = [[STDPSynapse(np.random.uniform(0.1, 0.5)) 
                          for _ in range(output_dim)] 
                         for _ in range(input_dim)]
        
        self.output_spikes = np.zeros(output_dim)

    def forward(self, input_signal, dt=1.0):
        input_signal = np.array(input_signal).flatten()
        self.output_spikes = np.zeros(self.output_dim)
        
        for i in range(self.output_dim):
            current = 0.0
            for j in range(min(self.input_dim, len(input_signal))):
                current += input_signal[j] * self.synapses[j][i].weight
            
            spike = self.neurons[i].forward(current, dt)
            self.output_spikes[i] = spike
        
        return self.output_spikes

    def learn(self, pre_spikes, post_spikes, dt=1.0):
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


class SpikingNeuralNetwork:
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer1 = SpikingLayer(input_dim, hidden_dim)
        self.layer2 = SpikingLayer(hidden_dim, output_dim)
        
        self.spike_history = deque(maxlen=50)
        self.learning_enabled = True

    def forward(self, input_signal, dt=1.0):
        hidden_spikes = self.layer1.forward(input_signal, dt)
        output_spikes = self.layer2.forward(hidden_spikes, dt)
        
        self.spike_history.append({
            "input": input_signal,
            "hidden": hidden_spikes,
            "output": output_spikes
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

    def convert_to_dense(self):
        w1 = np.array(self.layer1.get_weights())
        w2 = np.array(self.layer2.get_weights())
        
        dense1 = nn.Linear(self.input_dim, self.hidden_dim).to(DEVICE)
        dense2 = nn.Linear(self.hidden_dim, self.output_dim).to(DEVICE)
        
        with torch.no_grad():
            dense1.weight.data = torch.tensor(w1.T, dtype=torch.float32).to(DEVICE)
            dense2.weight.data = torch.tensor(w2.T, dtype=torch.float32).to(DEVICE)
        
        return nn.Sequential(dense1, nn.ReLU(), dense2).to(DEVICE)

    def get_spike_rate(self):
        if not self.spike_history:
            return 0.0
        
        total_spikes = sum(np.sum(h["output"]) for h in self.spike_history)
        return total_spikes / (len(self.spike_history) * self.output_dim)


class SNNEnhancer:
    def __init__(self, feature_dim=16):
        self.feature_dim = feature_dim
        self.snn = SpikingNeuralNetwork(input_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)
        self.integration_weight = 0.3
        self.timesteps = 5
        self.learning_rate = 1e-3

    def enhance(self, feature, learn=True):
        feature_np = feature.detach().cpu().numpy() if hasattr(feature, 'detach') else np.array(feature)
        
        snn_output = np.zeros(self.feature_dim)
        for t in range(self.timesteps):
            snn_output += self.snn.forward(feature_np, dt=1.0)
        
        snn_output = snn_output / self.timesteps
        
        if learn:
            self.snn.learn()
        
        enhanced = (1 - self.integration_weight) * feature_np + self.integration_weight * snn_output
        
        return torch.tensor(enhanced, dtype=torch.float32).to(DEVICE)

    def set_integration_weight(self, weight):
        self.integration_weight = max(0.0, min(1.0, weight))

    def set_learning_rate(self, lr):
        self.learning_rate = max(1e-5, min(1e-2, lr))

    def get_snn_stats(self):
        return {
            "spike_rate": self.snn.get_spike_rate(),
            "integration_weight": self.integration_weight,
            "timesteps": self.timesteps
        }

    def resize(self, new_feature_dim):
        self.feature_dim = new_feature_dim
        self.snn = SpikingNeuralNetwork(input_dim=new_feature_dim, hidden_dim=32, output_dim=new_feature_dim)