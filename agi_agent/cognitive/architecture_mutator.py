import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ..config.settings import DEVICE


class ArchitectureMutator:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.mutation_history = deque(maxlen=100)
        self.mutation_rate = 0.1
        self.min_dim = 8
        self.max_dim = 128
        self.enabled = True
    
    def mutate(self, trigger_type='impasse', context=None):
        if not self.enabled:
            return False
        
        if np.random.random() > self.mutation_rate:
            return False
        
        mutations = []
        
        if trigger_type == 'impasse':
            mutations = self._generate_impasse_mutations(context)
        elif trigger_type == 'stagnation':
            mutations = self._generate_stagnation_mutations(context)
        elif trigger_type == 'opportunity':
            mutations = self._generate_opportunity_mutations(context)
        
        for mutation in mutations:
            success = self._apply_mutation(mutation)
            if success:
                self.mutation_history.append({
                    'type': mutation['type'],
                    'trigger': trigger_type,
                    'timestamp': np.random.randint(1000000),
                    'context': context
                })
        
        return len(mutations) > 0
    
    def _generate_impasse_mutations(self, context):
        mutations = []
        
        if context and context.get('confidence', 0.5) < 0.3:
            mutations.append({
                'type': 'increase_cognitive_capacity',
                'params': {'target_module': 'dual_cognition', 'scale_factor': 1.5}
            })
        
        if context and context.get('causal_effect', 0.0) < 0.2:
            mutations.append({
                'type': 'enhance_causal_reasoning',
                'params': {'add_layers': 1, 'increase_nodes': 20}
            })
        
        mutations.append({
            'type': 'adjust_snn_connectivity',
            'params': {'connection_density': 0.3}
        })
        
        return mutations
    
    def _generate_stagnation_mutations(self, context):
        mutations = []
        
        mutations.append({
            'type': 'restructure_knowledge_graph',
            'params': {'prune_threshold': 0.1, 'merge_similar': True}
        })
        
        mutations.append({
            'type': 'reset_suboptimal_modules',
            'params': {'modules': ['cognitive']}
        })
        
        return mutations
    
    def _generate_opportunity_mutations(self, context):
        mutations = []
        
        mutations.append({
            'type': 'expand_perception_dimension',
            'params': {'new_dim': min(self.orchestrator.perception.get_feature_dim() * 2, self.max_dim)}
        })
        
        mutations.append({
            'type': 'increase_rl_depth',
            'params': {'num_options': 5}
        })
        
        return mutations
    
    def _apply_mutation(self, mutation):
        mutation_type = mutation['type']
        params = mutation.get('params', {})
        
        try:
            if mutation_type == 'increase_cognitive_capacity':
                self._mutate_cognitive_capacity(params)
            elif mutation_type == 'enhance_causal_reasoning':
                self._mutate_causal_reasoning(params)
            elif mutation_type == 'adjust_snn_connectivity':
                self._mutate_snn_connectivity(params)
            elif mutation_type == 'restructure_knowledge_graph':
                self._mutate_knowledge_graph(params)
            elif mutation_type == 'reset_suboptimal_modules':
                self._mutate_reset_modules(params)
            elif mutation_type == 'expand_perception_dimension':
                self._mutate_expand_perception(params)
            elif mutation_type == 'increase_rl_depth':
                self._mutate_rl_depth(params)
            else:
                return False
            
            return True
        except Exception as e:
            return False
    
    def _mutate_cognitive_capacity(self, params):
        target_module = params.get('target_module', 'dual_cognition')
        scale_factor = params.get('scale_factor', 1.5)
        
        if hasattr(self.orchestrator, target_module):
            module = getattr(self.orchestrator, target_module)
            if hasattr(module, 'resize'):
                current_dim = self.orchestrator.perception.get_feature_dim()
                new_dim = min(int(current_dim * scale_factor), self.max_dim)
                module.resize(new_dim)
    
    def _mutate_causal_reasoning(self, params):
        add_layers = params.get('add_layers', 1)
        increase_nodes = params.get('increase_nodes', 20)
        
        new_dim = min(self.orchestrator.causal_reasoner.feature_dim + increase_nodes, self.max_dim)
        self.orchestrator.causal_reasoner.resize(new_dim)
    
    def _mutate_snn_connectivity(self, params):
        connection_density = params.get('connection_density', 0.3)
        
        if hasattr(self.orchestrator.snn_enhancer, 'snn'):
            snn = self.orchestrator.snn_enhancer.snn
            if hasattr(snn, 'layers'):
                for layer in snn.layers:
                    if hasattr(layer, 'synapses'):
                        num_synapses = len(layer.synapses)
                        target_synapses = int(num_synapses * connection_density)
                        if target_synapses < len(layer.synapses):
                            layer.synapses = layer.synapses[:target_synapses]
    
    def _mutate_knowledge_graph(self, params):
        prune_threshold = params.get('prune_threshold', 0.1)
        merge_similar = params.get('merge_similar', True)
        
        kg = self.orchestrator.knowledge_graph
        if hasattr(kg, 'nodes'):
            nodes_to_remove = []
            for node_id, node_data in kg.nodes.items():
                if isinstance(node_data, dict) and node_data.get('importance', 1.0) < prune_threshold:
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove[:5]:
                if hasattr(kg, 'remove_node'):
                    kg.remove_node(node_id)
    
    def _mutate_reset_modules(self, params):
        modules = params.get('modules', [])
        
        for module_name in modules:
            if hasattr(self.orchestrator, module_name):
                module = getattr(self.orchestrator, module_name)
                if hasattr(module, 'reset'):
                    module.reset()
    
    def _mutate_expand_perception(self, params):
        new_dim = params.get('new_dim', 32)
        
        if new_dim > self.orchestrator.perception.get_feature_dim():
            self.orchestrator.perception.resize(new_dim)
            self.orchestrator._handle_structural_change(new_dim)
    
    def _mutate_rl_depth(self, params):
        num_options = params.get('num_options', 5)
        
        if hasattr(self.orchestrator.execution, 'high_level_policy'):
            policy = self.orchestrator.execution.high_level_policy
            if hasattr(policy, 'num_options'):
                policy.num_options = max(policy.num_options, num_options)
    
    def should_mutate(self, cognitive_state):
        confidence = cognitive_state.get('confidence', 0.5)
        impasse_count = cognitive_state.get('impasse_count', 0)
        stagnation_score = cognitive_state.get('stagnation_score', 0.0)
        
        if confidence < 0.3 and impasse_count > 3:
            return 'impasse'
        elif stagnation_score > 0.8:
            return 'stagnation'
        elif confidence > 0.8:
            return 'opportunity'
        
        return None
    
    def get_mutation_summary(self):
        recent = list(self.mutation_history)[-10:] if len(self.mutation_history) > 10 else list(self.mutation_history)
        
        mutation_counts = {}
        for m in recent:
            mutation_counts[m['type']] = mutation_counts.get(m['type'], 0) + 1
        
        return {
            'total_mutations': len(self.mutation_history),
            'recent_mutations': len(recent),
            'mutation_types': mutation_counts,
            'mutation_rate': self.mutation_rate
        }