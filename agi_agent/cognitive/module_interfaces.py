#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各功能模块的神经接口适配器

将每个模块的状态编码为脉冲模式，并能从接收到的脉冲模式解码出控制信号
"""

import numpy as np
from typing import List, Dict
from .module_synaptic_bus import NeuralInterface, Spike, SignalType


class MemoryNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('memory', num_neurons=64)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        memory_count = state.get('total_entries', 0)
        active_tier = state.get('active_tier', 'L1')
        
        tier_map = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4}
        tier_idx = tier_map.get(active_tier, 0)
        
        neuron_idx = tier_idx * 12 + (memory_count % 12)
        strength = min(1.0, memory_count / 100.0)
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'total_entries': memory_count, 'tier': active_tier}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'parameters': {}}
        
        for spike in spikes:
            if spike.signal_type == SignalType.REQUEST:
                result['action'] = 'retrieve'
                result['parameters'] = spike.payload
            elif spike.signal_type == SignalType.CONTROL:
                result['action'] = 'store'
                result['parameters'] = spike.payload
        
        return result


class KnowledgeGraphNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('knowledge_graph', num_neurons=64)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        node_count = state.get('nodes', 0)
        edge_count = state.get('edges', 0)
        
        neuron_idx = (node_count + edge_count) % self.num_neurons
        strength = min(1.0, (node_count + edge_count) / 500.0)
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'nodes': node_count, 'edges': edge_count}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'query': {}}
        
        for spike in spikes:
            if spike.signal_type == SignalType.REQUEST:
                result['action'] = 'query'
                result['query'] = spike.payload
            elif spike.signal_type == SignalType.CONTROL:
                result['action'] = 'update'
                result['query'] = spike.payload
        
        return result


class DecisionNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('decision', num_neurons=48)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        confidence = state.get('confidence', 0.5)
        action_count = state.get('action_count', 0)
        
        neuron_idx = action_count % self.num_neurons
        strength = confidence
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.CONTROL,
            strength=strength,
            payload={'confidence': confidence, 'action_count': action_count}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': 'decide', 'inputs': [], 'confidence': 0.5}
        
        total_strength = 0.0
        count = 0
        
        for spike in spikes:
            total_strength += spike.strength
            count += 1
            if 'payload' in spike.payload:
                result['inputs'].append(spike.payload)
        
        if count > 0:
            result['confidence'] = min(1.0, total_strength / count)
        
        return result


class ExecutionNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('execution', num_neurons=48)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        status = state.get('status', 'idle')
        progress = state.get('progress', 0.0)
        
        status_map = {'idle': 0, 'running': 1, 'completed': 2, 'failed': 3}
        status_idx = status_map.get(status, 0)
        
        neuron_idx = status_idx * 12 + int(progress * 12)
        strength = progress if status == 'running' else 0.5
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'status': status, 'progress': progress}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'task': {}}
        
        for spike in spikes:
            if spike.signal_type == SignalType.CONTROL:
                result['action'] = 'execute'
                result['task'] = spike.payload
            elif spike.signal_type == SignalType.ACKNOWLEDGE:
                result['action'] = 'complete'
        
        return result


class PerceptionNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('perception', num_neurons=64)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        feature_dim = state.get('feature_dim', 16)
        novelty = state.get('novelty', 0.0)
        confidence = state.get('confidence', 0.5)
        
        neuron_idx = int(novelty * self.num_neurons)
        strength = confidence
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'feature_dim': feature_dim, 'novelty': novelty, 'confidence': confidence}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': 'perceive', 'sensitivity': 1.0}
        
        total_strength = sum(s.spike.strength for s in spikes) if spikes else 0.0
        result['sensitivity'] = min(2.0, total_strength / len(spikes)) if spikes else 1.0
        
        return result


class SecurityNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('security', num_neurons=32)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        risk_level = state.get('risk_level', 'low')
        threat_count = state.get('threat_count', 0)
        
        risk_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        risk_idx = risk_map.get(risk_level, 0)
        
        neuron_idx = risk_idx * 8 + (threat_count % 8)
        strength = min(1.0, threat_count / 10.0) if threat_count > 0 else 0.2
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.CONTROL if risk_level in ['high', 'critical'] else SignalType.DATA,
            strength=strength,
            payload={'risk_level': risk_level, 'threat_count': threat_count}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'risk_assessment': 'low'}
        
        for spike in spikes:
            if spike.signal_type == SignalType.CONTROL:
                result['action'] = 'block'
                result['risk_assessment'] = 'high'
            elif spike.signal_type == SignalType.REQUEST:
                result['action'] = 'verify'
                result['risk_assessment'] = 'medium'
        
        return result


class SoulNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('soul', num_neurons=48)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        identity = state.get('identity', {})
        goal_count = state.get('goal_count', 0)
        personality = state.get('personality', {})
        
        goal_idx = goal_count % 12
        neuron_idx = goal_idx
        
        if personality:
            openness = personality.get('curiosity', 50) / 100.0
            neuron_idx += int(openness * 12)
        
        strength = 0.7
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'identity': identity, 'goal_count': goal_count, 'personality': personality}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'identity_update': {}, 'goal_update': {}}
        
        for spike in spikes:
            if spike.signal_type == SignalType.CONTROL:
                result['action'] = 'update_identity'
                result['identity_update'] = spike.payload
            elif spike.signal_type == SignalType.REQUEST:
                result['action'] = 'evaluate_goal'
                result['goal_update'] = spike.payload
        
        return result


class SkillsNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('skills', num_neurons=48)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        skill_count = state.get('skill_count', 0)
        active_skills = state.get('active_skills', [])
        
        neuron_idx = skill_count % self.num_neurons
        strength = min(1.0, len(active_skills) / 5.0)
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'skill_count': skill_count, 'active_skills': active_skills}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'skill': None, 'parameters': {}}
        
        for spike in spikes:
            if spike.signal_type == SignalType.REQUEST:
                result['action'] = 'execute'
                result['skill'] = spike.payload.get('skill')
                result['parameters'] = spike.payload.get('parameters', {})
            elif spike.signal_type == SignalType.CONTROL:
                result['action'] = 'learn'
                result['skill'] = spike.payload.get('skill')
        
        return result


class EvolutionNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('evolution', num_neurons=32)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        evolution_count = state.get('evolution_count', 0)
        fitness = state.get('fitness', 0.5)
        level = state.get('level', 'individual')
        
        level_map = {'individual': 0, 'population': 1, 'species': 2, 'ecosystem': 3}
        level_idx = level_map.get(level, 0)
        
        neuron_idx = level_idx * 8 + (evolution_count % 8)
        strength = fitness
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.CONTROL,
            strength=strength,
            payload={'evolution_count': evolution_count, 'fitness': fitness, 'level': level}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'direction': 'exploration'}
        
        total_strength = sum(s.strength for s in spikes) if spikes else 0.0
        
        if total_strength > 0.8:
            result['action'] = 'mutate'
            result['direction'] = 'exploitation'
        elif total_strength > 0.4:
            result['action'] = 'select'
            result['direction'] = 'exploration'
        
        return result


class SelfImprovementNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('self_improvement', num_neurons=32)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        performance_score = state.get('performance_score', 85)
        issue_count = state.get('issue_count', 0)
        improvement_count = state.get('improvement_count', 0)
        
        score_idx = int(performance_score / 20)
        neuron_idx = score_idx * 8 + (improvement_count % 8)
        strength = min(1.0, performance_score / 100.0)
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.CONTROL,
            strength=strength,
            payload={'performance_score': performance_score, 'issue_count': issue_count, 'improvement_count': improvement_count}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'priority': 'medium', 'target': None}
        
        for spike in spikes:
            if spike.signal_type == SignalType.CONTROL:
                result['action'] = 'diagnose'
                result['target'] = spike.payload.get('target')
                result['priority'] = spike.payload.get('priority', 'medium')
            elif spike.signal_type == SignalType.REQUEST:
                result['action'] = 'verify'
        
        return result


class MetaCognitionNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('metacognition', num_neurons=48)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        awareness_level = state.get('awareness_level', 0.5)
        monitoring_count = state.get('monitoring_count', 0)
        strategy_effectiveness = state.get('strategy_effectiveness', 0.7)
        
        neuron_idx = int(awareness_level * self.num_neurons)
        strength = strategy_effectiveness
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.CONTROL,
            strength=strength,
            payload={'awareness_level': awareness_level, 'monitoring_count': monitoring_count, 'strategy_effectiveness': strategy_effectiveness}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'strategy': None, 'adjustment': 0.0}
        
        total_strength = sum(s.strength for s in spikes) if spikes else 0.0
        
        if total_strength > 0.8:
            result['action'] = 'adjust_strategy'
            result['adjustment'] = 0.2
        elif total_strength > 0.4:
            result['action'] = 'monitor'
            result['adjustment'] = 0.1
        
        return result


class HomeostasisNeuralInterface(NeuralInterface):
    def __init__(self):
        super().__init__('homeostasis', num_neurons=32)
        
    def encode_state(self, state: Dict) -> List[Spike]:
        spikes = []
        
        energy_level = state.get('energy_level', 0.8)
        resource_usage = state.get('resource_usage', {})
        stability = state.get('stability', 0.9)
        
        energy_idx = int((1.0 - energy_level) * 16)
        neuron_idx = energy_idx
        
        if resource_usage:
            cpu_usage = resource_usage.get('cpu', 0) / 100.0
            neuron_idx += int(cpu_usage * 16)
        
        strength = stability
        
        spikes.append(Spike(
            timestamp=self.last_encode_time,
            module_id=self.module_id,
            neuron_index=neuron_idx,
            signal_type=SignalType.DATA,
            strength=strength,
            payload={'energy_level': energy_level, 'resource_usage': resource_usage, 'stability': stability}
        ))
        
        return spikes
    
    def decode_spikes(self, spikes: List[Spike]) -> Dict:
        result = {'action': None, 'regulation': 'normal'}
        
        total_strength = sum(s.strength for s in spikes) if spikes else 0.0
        
        if total_strength < 0.3:
            result['action'] = 'conserve'
            result['regulation'] = 'low'
        elif total_strength > 0.9:
            result['action'] = 'allocate'
            result['regulation'] = 'high'
        
        return result


INTERFACE_MAP = {
    'memory': MemoryNeuralInterface,
    'knowledge_graph': KnowledgeGraphNeuralInterface,
    'decision': DecisionNeuralInterface,
    'execution': ExecutionNeuralInterface,
    'perception': PerceptionNeuralInterface,
    'security': SecurityNeuralInterface,
    'soul': SoulNeuralInterface,
    'skills': SkillsNeuralInterface,
    'evolution': EvolutionNeuralInterface,
    'self_improvement': SelfImprovementNeuralInterface,
    'metacognition': MetaCognitionNeuralInterface,
    'homeostasis': HomeostasisNeuralInterface,
}


def create_interface(module_id: str) -> NeuralInterface:
    interface_class = INTERFACE_MAP.get(module_id)
    if interface_class:
        return interface_class()
    return NeuralInterface(module_id)