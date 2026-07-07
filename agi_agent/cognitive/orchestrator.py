import numpy as np
import torch
from collections import deque
from ..config.settings import DEVICE


class UnifiedCognitiveOrchestrator:
    def __init__(self, perception, cognition, dual_cognition, snn_enhancer, causal_reasoner, 
                 meta_cog, homeostasis, execution, knowledge_graph, self_model=None):
        self.perception = perception
        self.cognition = cognition
        self.dual_cognition = dual_cognition
        self.snn_enhancer = snn_enhancer
        self.causal_reasoner = causal_reasoner
        self.meta_cog = meta_cog
        self.homeostasis = homeostasis
        self.execution = execution
        self.knowledge_graph = knowledge_graph
        self.self_model = self_model
        
        if self.self_model is not None and hasattr(self.homeostasis, 'set_self_model'):
            self.homeostasis.set_self_model(self.self_model)
        
        self.module_states = {}
        self.integration_weights = {
            'causal_to_rl': 0.3,
            'homeostasis_to_snn': 0.4,
            'metacog_to_architecture': 0.5,
            'knowledge_to_cognition': 0.3,
            'self_model_to_goal': 0.4
        }
        
        self.history = deque(maxlen=50)
        self.enabled = True
    
    def orchestrate(self, obs_tensor):
        if not self.enabled:
            return None
        
        feat, fe, structure_changed = self.perception.update(obs_tensor)
        
        if structure_changed:
            self._handle_structural_change(feat.shape[-1])
        
        fused_feat = self._integrate_multimodal(feat)
        
        self.snn_enhancer.set_learning_rate(self._homeostasis_driven_learning_rate())
        
        enhanced_feat = self.snn_enhancer.enhance(fused_feat)
        
        self_reflection = None
        internal_state_prediction = None
        if self.self_model is not None:
            needs_status = self.homeostasis.get_needs_status() if hasattr(self.homeostasis, 'get_needs_status') else {}
            internal_state_vec = np.array([
                needs_status.get('energy', {}).get('value', 0.7),
                needs_status.get('attention', {}).get('value', 0.6),
                needs_status.get('security', {}).get('value', 0.8),
                needs_status.get('curiosity', {}).get('value', 0.5),
                needs_status.get('competence', {}).get('value', 0.5)
            ])
            
            trajectory = self.self_model.predict_self_trajectory(internal_state_vec)
            self_reflection = self.self_model.generate_self_reflection(internal_state_vec, trajectory)
            internal_state_prediction = trajectory[-1]
        
        causal_result = self.causal_reasoner.reason(enhanced_feat)
        
        self.execution.set_causal_bias(causal_result.get('causal_effect', 0.0))
        
        if self_reflection and self_reflection.get('detected_problems'):
            self.execution.set_exploration_noise(min(0.5, self.execution.exploration_noise + 0.1))
        
        dual_result = self.dual_cognition.think(enhanced_feat)
        final_pred = dual_result["prediction"]
        confidence = dual_result.get("confidence", 0.5)
        system_used = dual_result.get("system", "system2")
        
        is_impasse, impasse_record = self.meta_cog.check_cognitive_impasse()
        mutation_proposal = None
        if is_impasse:
            subgoal = self.meta_cog.handle_impasse("general_task", {
                "feature_dim": fused_feat.shape,
                "system_used": system_used,
                "confidence": confidence,
                "causal_effect": causal_result.get('causal_effect', 0.0),
                "self_reflection": self_reflection
            })
            if subgoal:
                mutation_proposal = self._generate_mutation_proposal(subgoal)
        
        if final_pred is None:
            context_type = "novel" if is_impasse else "structured"
            strategy = self.meta_cog.select_learning_strategy(context_type, 1.0 - confidence)
            pred_seq = self.cognitive.autonomous_thinking(fused_feat)
            final_pred = pred_seq[-1]
        else:
            self.dual_cognition.learn_pattern(fused_feat, final_pred)
        
        self._update_knowledge_with_causal(fused_feat, final_pred, causal_result)
        
        homeo_goal = self._get_homeostatic_goal()
        
        action = self._cognitively_informed_action(enhanced_feat, final_pred, causal_result, homeo_goal)
        
        entropy_val = self._calc_entropy(fused_feat)
        self.meta_cog.monitor(fe, entropy_val, 0.0, 0.0)
        
        if self.self_model is not None:
            self.self_model.boundary_detector.record_self_state(fused_feat)
        
        self.history.append({
            'confidence': confidence,
            'causal_effect': causal_result.get('causal_effect', 0.0),
            'system_used': system_used,
            'is_impasse': is_impasse,
            'self_awareness': self_reflection.get('self_awareness', 0.5) if self_reflection else 0.5
        })
        
        return {
            'prediction': final_pred,
            'action': action,
            'confidence': confidence,
            'causal_result': causal_result,
            'free_energy': fe,
            'system_used': system_used,
            'is_impasse': is_impasse,
            'fused_feat': fused_feat,
            'entropy': entropy_val,
            'mutation_proposal': mutation_proposal,
            'homeostatic_goal': homeo_goal,
            'self_reflection': self_reflection,
            'internal_state_prediction': internal_state_prediction
        }
    
    def _integrate_multimodal(self, feat):
        feat_np = feat.detach().cpu().numpy() if hasattr(feat, 'detach') else np.array(feat)
        return feat_np
    
    def _homeostasis_driven_learning_rate(self):
        if hasattr(self.homeostasis, 'get_needs_status'):
            needs = self.homeostasis.get_needs_status()
        elif hasattr(self.homeostasis, 'goal_generator'):
            needs = self.homeostasis.goal_generator.internal_energy.get_need_status()
        else:
            return 1e-3
        
        curiosity = needs.get('curiosity', {}).get('value', 0.5)
        competence = needs.get('competence', {}).get('value', 0.5)
        
        base_lr = 1e-3
        modulated_lr = base_lr * (1 + curiosity * 0.5 - competence * 0.3)
        
        return min(max(modulated_lr, 1e-5), 1e-2)
    
    def _update_knowledge_with_causal(self, features, prediction, causal_result):
        if 'analogous_actions' in causal_result and causal_result['analogous_actions']:
            for analog in causal_result['analogous_actions']:
                if analog['similarity'] > 0.7:
                    self.knowledge_graph.add_node(f"analog_rule_{id(analog)}", {
                        'action': analog['action'],
                        'similarity': analog['similarity']
                    })
    
    def _cognitively_informed_action(self, features, prediction, causal_result, homeo_goal=None):
        causal_effect = causal_result.get('causal_effect', 0.0)
        
        if causal_effect > 0.5:
            self.execution.set_exploration_noise(0.1)
        else:
            self.execution.set_exploration_noise(0.3)
        
        if homeo_goal:
            goal_type = homeo_goal.get('type', '')
            if goal_type == 'explore_unknown':
                self.execution.set_exploration_noise(min(0.5, self.execution.exploration_noise + 0.1))
            elif goal_type == 'focus_on_task':
                self.execution.set_exploration_noise(max(0.05, self.execution.exploration_noise - 0.05))
        
        analogous_actions = causal_result.get('analogous_actions', [])
        if analogous_actions and analogous_actions[0].get('similarity', 0) > 0.8:
            self.execution.set_causal_bias(0.8)
        
        return self.execution.autonomous_action(features, prediction)
    
    def _generate_mutation_proposal(self, subgoal):
        subgoal_type = subgoal.get('type', '')
        if subgoal_type == 'knowledge_gap':
            return {
                'type': 'increase_cognitive_capacity',
                'target': 'dual_cognition',
                'scale_factor': 1.5,
                'reason': 'knowledge_gap_detected'
            }
        elif subgoal_type == 'complex_reasoning':
            return {
                'type': 'enhance_causal_reasoning',
                'increase_nodes': 20,
                'reason': 'complex_reasoning_required'
            }
        elif subgoal_type == 'exploration_needed':
            return {
                'type': 'expand_perception_dimension',
                'new_dim': min(self.perception.get_feature_dim() * 2, 128),
                'reason': 'exploration_required'
            }
        return None
    
    def _get_homeostatic_goal(self):
        if hasattr(self.homeostasis, 'goal_generator'):
            return self.homeostasis.goal_generator.get_highest_priority_goal()
        return None
    
    def _trigger_architecture_adjustment(self, subgoal):
        if subgoal.get('type') == 'knowledge_gap':
            self.meta_cog.schedule_resource('learning', priority=1.0)
        elif subgoal.get('type') == 'complex_reasoning':
            self.meta_cog.schedule_resource('cognitive', priority=1.0)
    
    def _handle_structural_change(self, new_dim):
        self.cognition.resize(new_dim)
        self.dual_cognition.resize(new_dim)
        self.snn_enhancer.resize(new_dim)
        self.causal_reasoner.resize(new_dim)
        self.execution.hardware_adapt(new_dim)
        if hasattr(self.homeostasis, 'resize'):
            self.homeostasis.resize(new_dim)
        else:
            self.homeostasis = type(self.homeostasis)(feature_dim=new_dim)
        if self.self_model is not None:
            self.self_model.resize(new_dim)
    
    def _calc_entropy(self, features):
        feat_np = np.array(features) if not isinstance(features, np.ndarray) else features
        feat_np = np.maximum(feat_np, 0.0)
        total = np.sum(feat_np) + 1e-10
        prob = feat_np / total
        prob = np.maximum(prob, 1e-10)
        return -np.sum(prob * np.log(prob))
    
    def get_orchestration_summary(self):
        recent = list(self.history)[-10:] if len(self.history) > 10 else list(self.history)
        avg_confidence = np.mean([h['confidence'] for h in recent])
        avg_causal_effect = np.mean([h['causal_effect'] for h in recent])
        impasse_rate = sum(1 for h in recent if h['is_impasse']) / len(recent) if recent else 0
        
        return {
            'avg_confidence': avg_confidence,
            'avg_causal_effect': avg_causal_effect,
            'impasse_rate': impasse_rate,
            'history_length': len(self.history),
            'integration_weights': self.integration_weights
        }