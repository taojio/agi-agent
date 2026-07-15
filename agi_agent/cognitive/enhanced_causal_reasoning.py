import numpy as np
from collections import deque
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from scipy.stats import pearsonr
from scipy.special import expit

from .causal_reasoning import CausalGraph, CausalInferenceEngine
from ..config.settings import DEVICE


class CausalStrengthMetric(Enum):
    GRANGER = "granger"
    MUTUAL_INFORMATION = "mutual_information"
    CONDITIONAL_PREDICTION_GAIN = "conditional_prediction_gain"


class TimeSeriesCausalDiscovery:
    def __init__(self, max_lag: int = 5, top_k_edges: int = 10,
                 strength_metric: str = "conditional_prediction_gain"):
        self.max_lag = max_lag
        self.top_k_edges = top_k_edges
        self.strength_metric = CausalStrengthMetric(strength_metric)
        self.causal_graph = None
        self.scalers = {}
        self.discovery_results = {}

    def discover(self, time_series: np.ndarray, variable_names: Optional[List[str]] = None):
        n_vars = time_series.shape[1]
        
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(n_vars)]
        
        self.causal_graph = CausalGraph(num_nodes=n_vars)
        self.causal_graph.node_names = variable_names
        
        self._fit_scalers(time_series)
        
        causal_scores = []
        
        for cause_idx in range(n_vars):
            for effect_idx in range(n_vars):
                if cause_idx == effect_idx:
                    continue
                
                for lag in range(1, self.max_lag + 1):
                    score = self._compute_causal_strength(
                        time_series[:, cause_idx],
                        time_series[:, effect_idx],
                        lag
                    )
                    
                    if score > 0.01:
                        causal_scores.append({
                            "cause": cause_idx,
                            "effect": effect_idx,
                            "lag": lag,
                            "strength": score
                        })
        
        causal_scores.sort(key=lambda x: -x["strength"])
        
        used_pairs = set()
        for result in causal_scores[:self.top_k_edges]:
            pair_key = (result["cause"], result["effect"])
            if pair_key not in used_pairs:
                self.causal_graph.add_edge(
                    result["cause"],
                    result["effect"],
                    result["strength"]
                )
                used_pairs.add(pair_key)
        
        self.discovery_results = {
            "total_tests": n_vars * (n_vars - 1) * self.max_lag,
            "edges_found": len(used_pairs),
            "top_edges": causal_scores[:self.top_k_edges],
            "variable_names": variable_names
        }
        
        return self.causal_graph

    def _fit_scalers(self, time_series: np.ndarray):
        for i in range(time_series.shape[1]):
            scaler = StandardScaler()
            scaler.fit(time_series[:, i:i+1])
            self.scalers[i] = scaler

    def _compute_causal_strength(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        cause_scaled = self.scalers[cause][0].transform(cause.reshape(-1, 1)).flatten()
        effect_scaled = self.scalers[effect][0].transform(effect.reshape(-1, 1)).flatten()
        
        cause_lagged = np.roll(cause_scaled, lag)
        cause_lagged[:lag] = cause_scaled[:lag]
        
        if self.strength_metric == CausalStrengthMetric.GRANGER:
            return self._granger_causality(cause_lagged, effect_scaled, lag)
        elif self.strength_metric == CausalStrengthMetric.MUTUAL_INFORMATION:
            return self._mutual_information(cause_lagged, effect_scaled)
        else:
            return self._conditional_prediction_gain(cause_lagged, effect_scaled, lag)

    def _granger_causality(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        n = len(effect) - lag
        Y = effect[lag:]
        
        X_self = np.column_stack([np.roll(effect, i)[lag:] for i in range(1, lag + 1)])
        X_cause = np.column_stack([X_self, cause[lag:]])
        
        beta_self = np.linalg.lstsq(X_self, Y, rcond=None)[0]
        beta_full = np.linalg.lstsq(X_cause, Y, rcond=None)[0]
        
        res_self = Y - X_self @ beta_self
        res_full = Y - X_cause @ beta_full
        
        ssr_self = np.sum(res_self ** 2)
        ssr_full = np.sum(res_full ** 2)
        
        if ssr_self < 1e-10:
            return 0.0
        
        f_stat = ((ssr_self - ssr_full) / lag) / (ssr_full / (n - 2 * lag))
        return float(min(f_stat / 100, 1.0))

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        min_len = min(len(x), len(y))
        x_norm = (x[:min_len] - np.mean(x[:min_len])) / (np.std(x[:min_len]) + 1e-8)
        y_norm = (y[:min_len] - np.mean(y[:min_len])) / (np.std(y[:min_len]) + 1e-8)
        
        corr, _ = pearsonr(x_norm, y_norm)
        return float(0.5 * np.log(1 / (1 - corr ** 2 + 1e-8)))

    def _conditional_prediction_gain(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        n = len(effect) - lag
        Y = effect[lag:]
        
        X_self = np.column_stack([np.roll(effect, i)[lag:] for i in range(1, lag + 1)])
        X_cause = np.column_stack([X_self, cause[lag:]])
        
        X_self_train, X_self_test, Y_train, Y_test = train_test_split(X_self, Y, test_size=0.3, shuffle=False)
        X_cause_train, X_cause_test = train_test_split(X_cause, test_size=0.3, shuffle=False)
        
        beta_self = np.linalg.lstsq(X_self_train, Y_train, rcond=None)[0]
        beta_full = np.linalg.lstsq(X_cause_train, Y_train, rcond=None)[0]
        
        pred_self = X_self_test @ beta_self
        pred_full = X_cause_test @ beta_full
        
        mse_self = mean_squared_error(Y_test, pred_self)
        mse_full = mean_squared_error(Y_test, pred_full)
        
        if mse_self < 1e-10:
            return 0.0
        
        gain = (mse_self - mse_full) / mse_self
        return float(max(0, min(gain, 1.0)))

    def get_visualization_data(self) -> Dict[str, Any]:
        if self.causal_graph is None:
            return {}
        
        edges = []
        for i in range(self.causal_graph.num_nodes):
            for j in range(self.causal_graph.num_nodes):
                if self.causal_graph.adjacency_matrix[i, j] > 0:
                    edges.append({
                        "source": self.causal_graph.node_names[i],
                        "target": self.causal_graph.node_names[j],
                        "strength": float(self.causal_graph.causal_strength[i, j])
                    })
        
        return {
            "nodes": self.causal_graph.node_names,
            "edges": edges,
            "discovery_summary": self.discovery_results
        }


class DoubleMLCausalEffect:
    def __init__(self, base_model: str = "mlp", n_folds: int = 5):
        self.base_model = base_model
        self.n_folds = n_folds
        self.models = {}
        self.scaler = StandardScaler()
        self.effect_estimates = []

    def estimate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        X_scaled = self.scaler.fit_transform(X)
        
        T = T.flatten() if T.ndim > 1 else T
        Y = Y.flatten() if Y.ndim > 1 else Y
        
        estimates = []
        
        for fold in range(self.n_folds):
            idx = np.arange(len(Y))
            train_idx = idx[idx % self.n_folds != fold]
            test_idx = idx[idx % self.n_folds == fold]
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            m_model = self._create_model(X_train.shape[1], 1)
            y_model = self._create_model(X_train.shape[1] + 1, 1)
            
            m_model.fit(torch.tensor(X_train, dtype=torch.float32), 
                       torch.tensor(T_train, dtype=torch.float32).unsqueeze(1))
            y_model.fit(torch.tensor(np.column_stack([X_train, T_train])), 
                       torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1))
            
            m_pred = m_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
            y_pred_0 = y_model(torch.tensor(np.column_stack([X_test, np.zeros(len(X_test))])), 
                             dtype=torch.float32).detach().numpy().flatten()
            y_pred_1 = y_model(torch.tensor(np.column_stack([X_test, np.ones(len(X_test))])), 
                             dtype=torch.float32).detach().numpy().flatten()
            
            residuals_y = Y_test - y_pred_0
            residuals_t = T_test - m_pred
            
            weight = residuals_t ** 2
            weight = weight / np.sum(weight)
            
            theta_hat = np.sum(residuals_y * residuals_t * weight)
            estimates.append(theta_hat)
        
        self.effect_estimates = estimates
        
        return {
            "ate": float(np.mean(estimates)),
            "std_error": float(np.std(estimates) / np.sqrt(self.n_folds)),
            "confidence_interval": self._compute_ci(estimates),
            "individual_estimates": [float(e) for e in estimates]
        }

    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        class TrainedModel(nn.Module):
            def __init__(self, model, optimizer, criterion):
                super().__init__()
                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion
            
            def fit(self, X, y, epochs: int = 50):
                for _ in range(epochs):
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()
            
            def forward(self, X):
                return self.model(X)
        
        return TrainedModel(model, optimizer, criterion)

    def _compute_ci(self, estimates: List[float], alpha: float = 0.05) -> Tuple[float, float]:
        mean_val = np.mean(estimates)
        std_val = np.std(estimates)
        n = len(estimates)
        
        margin = 1.96 * std_val / np.sqrt(n)
        
        return float(mean_val - margin), float(mean_val + margin)


class CausalChainAnalyzer:
    def __init__(self, causal_graph: Optional[CausalGraph] = None):
        self.causal_graph = causal_graph
        self.chain_results = {}

    def set_graph(self, causal_graph: CausalGraph):
        self.causal_graph = causal_graph

    def find_key_chains(self, source_node: int, target_node: int, 
                       max_length: int = 5) -> List[Dict[str, Any]]:
        if self.causal_graph is None:
            return []
        
        all_paths = self._find_all_paths(source_node, target_node, max_length)
        
        chains = []
        for path in all_paths:
            chain_strength = self._compute_chain_strength(path)
            bottleneck = self._find_bottleneck(path)
            
            chains.append({
                "path": [self.causal_graph.node_names[i] for i in path],
                "path_indices": path,
                "strength": chain_strength,
                "length": len(path) - 1,
                "bottleneck": bottleneck,
                "bottleneck_strength": self.causal_graph.causal_strength[
                    bottleneck[0], bottleneck[1]] if bottleneck else None
            })
        
        chains.sort(key=lambda x: -x["strength"])
        
        return chains

    def _find_all_paths(self, start: int, end: int, max_length: int, 
                       path: Optional[List[int]] = None) -> List[List[int]]:
        if path is None:
            path = []
        
        path = path + [start]
        paths = []
        
        if start == end and len(path) > 1:
            paths.append(path)
        elif len(path) <= max_length:
            for neighbor in range(self.causal_graph.num_nodes):
                if self.causal_graph.adjacency_matrix[start, neighbor] > 0 and neighbor not in path:
                    paths.extend(self._find_all_paths(neighbor, end, max_length, path))
        
        return paths

    def _compute_chain_strength(self, path: List[int]) -> float:
        strength = 1.0
        for i in range(len(path) - 1):
            strength *= self.causal_graph.causal_strength[path[i], path[i + 1]]
        
        return float(strength)

    def _find_bottleneck(self, path: List[int]) -> Optional[Tuple[int, int]]:
        min_strength = float('inf')
        bottleneck = None
        
        for i in range(len(path) - 1):
            strength = self.causal_graph.causal_strength[path[i], path[i + 1]]
            if strength < min_strength:
                min_strength = strength
                bottleneck = (path[i], path[i + 1])
        
        return bottleneck

    def generate_path_report(self, source_node: int, target_node: int) -> Dict[str, Any]:
        chains = self.find_key_chains(source_node, target_node)
        
        if not chains:
            return {
                "source": self.causal_graph.node_names[source_node],
                "target": self.causal_graph.node_names[target_node],
                "message": "No causal paths found between source and target"
            }
        
        top_chain = chains[0]
        
        return {
            "source": self.causal_graph.node_names[source_node],
            "target": self.causal_graph.node_names[target_node],
            "total_paths_found": len(chains),
            "top_chain": top_chain,
            "all_chains": chains,
            "recommendations": self._generate_recommendations(chains)
        }

    def _generate_recommendations(self, chains: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        
        for chain in chains[:3]:
            if chain["bottleneck"]:
                bottleneck_name = f"{self.causal_graph.node_names[chain['bottleneck'][0]]} -> " \
                               f"{self.causal_graph.node_names[chain['bottleneck'][1]]}"
                recommendations.append(
                    f"路径 '{chain['path']}' 的瓶颈为 {bottleneck_name}，强度 {chain['bottleneck_strength']:.4f}"
                )
        
        return recommendations

    def analyze_entire_graph(self) -> Dict[str, Any]:
        if self.causal_graph is None:
            return {}
        
        all_chains = []
        for i in range(self.causal_graph.num_nodes):
            for j in range(self.causal_graph.num_nodes):
                if i != j:
                    chains = self.find_key_chains(i, j, max_length=4)
                    all_chains.extend(chains)
        
        all_chains.sort(key=lambda x: -x["strength"])
        
        bottleneck_counts = {}
        for chain in all_chains:
            if chain["bottleneck"]:
                key = (chain["bottleneck"][0], chain["bottleneck"][1])
                bottleneck_counts[key] = bottleneck_counts.get(key, 0) + 1
        
        critical_bottlenecks = sorted(bottleneck_counts.items(), key=lambda x: -x[1])[:5]
        
        return {
            "total_chains_analyzed": len(all_chains),
            "strongest_chains": all_chains[:10],
            "critical_bottlenecks": [
                {
                    "edge": (self.causal_graph.node_names[b[0]], 
                            self.causal_graph.node_names[b[1]]),
                    "occurrences": count
                }
                for b, count in critical_bottlenecks
            ]
        }


class CausalTransferLearner:
    def __init__(self, method: str = "cdan"):
        self.method = method.lower()
        self.source_graph = None
        self.target_graph = None
        self.transfer_results = {}

    def transfer(self, source_graph: CausalGraph, target_data: np.ndarray,
                 target_variable_names: Optional[List[str]] = None) -> CausalGraph:
        self.source_graph = source_graph
        
        n_target_vars = target_data.shape[1]
        
        if target_variable_names is None:
            target_variable_names = [f"target_var_{i}" for i in range(n_target_vars)]
        
        self.target_graph = CausalGraph(num_nodes=n_target_vars)
        self.target_graph.node_names = target_variable_names
        
        if self.method == "direct":
            self._direct_transfer(source_graph)
        elif self.method == "dan":
            self._dan_transfer(source_graph, target_data)
        elif self.method == "cdan":
            self._cdan_transfer(source_graph, target_data)
        else:
            self._direct_transfer(source_graph)
        
        return self.target_graph

    def _direct_transfer(self, source_graph: CausalGraph):
        n_nodes = min(source_graph.num_nodes, self.target_graph.num_nodes)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if source_graph.adjacency_matrix[i, j] > 0:
                    self.target_graph.add_edge(i, j, source_graph.causal_strength[i, j])

    def _dan_transfer(self, source_graph: CausalGraph, target_data: np.ndarray):
        n_target_vars = target_data.shape[1]
        n_source_vars = source_graph.num_nodes
        
        alignment_scores = self._compute_alignment(source_graph, target_data)
        
        for i in range(n_source_vars):
            for j in range(n_source_vars):
                if source_graph.adjacency_matrix[i, j] > 0:
                    if i < n_target_vars and j < n_target_vars:
                        source_strength = source_graph.causal_strength[i, j]
                        alignment = alignment_scores.get((i, j), 0.5)
                        transferred_strength = source_strength * alignment
                        
                        if transferred_strength > 0.1:
                            self.target_graph.add_edge(i, j, transferred_strength)

    def _cdan_transfer(self, source_graph: CausalGraph, target_data: np.ndarray):
        n_target_vars = target_data.shape[1]
        n_source_vars = source_graph.num_nodes
        
        alignment_scores = self._compute_alignment(source_graph, target_data)
        conditional_scores = self._compute_conditional_alignment(source_graph, target_data)
        
        for i in range(n_source_vars):
            for j in range(n_source_vars):
                if source_graph.adjacency_matrix[i, j] > 0:
                    if i < n_target_vars and j < n_target_vars:
                        source_strength = source_graph.causal_strength[i, j]
                        alignment = alignment_scores.get((i, j), 0.5)
                        conditional = conditional_scores.get((i, j), 0.8)
                        
                        transferred_strength = source_strength * alignment * conditional
                        
                        if transferred_strength > 0.1:
                            self.target_graph.add_edge(i, j, transferred_strength)

    def _compute_alignment(self, source_graph: CausalGraph, target_data: np.ndarray) -> Dict[Tuple[int, int], float]:
        n_target_vars = target_data.shape[1]
        n_source_vars = source_graph.num_nodes
        
        scores = {}
        
        for i in range(min(n_source_vars, n_target_vars)):
            for j in range(min(n_source_vars, n_target_vars)):
                if i == j:
                    continue
                
                corr = np.corrcoef(target_data[:, i], target_data[:, j])[0, 1]
                scores[(i, j)] = (corr + 1) / 2
        
        return scores

    def _compute_conditional_alignment(self, source_graph: CausalGraph, 
                                       target_data: np.ndarray) -> Dict[Tuple[int, int], float]:
        n_target_vars = target_data.shape[1]
        scores = {}
        
        for i in range(n_target_vars):
            for j in range(n_target_vars):
                if i == j:
                    continue
                
                other_vars = [k for k in range(n_target_vars) if k != i and k != j]
                if len(other_vars) >= 1:
                    cond_corr = self._partial_correlation(target_data[:, i], 
                                                          target_data[:, j],
                                                          target_data[:, other_vars])
                    scores[(i, j)] = (cond_corr + 1) / 2
                else:
                    scores[(i, j)] = 0.5
        
        return scores

    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, 
                            covariates: np.ndarray) -> float:
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        
        X = np.column_stack([x, covariates])
        Y = y
        
        beta_x = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta_y = np.linalg.lstsq(np.column_stack([y, covariates]), x, rcond=None)[0]
        
        res_x = Y - X @ beta_x
        res_y = x - np.column_stack([y, covariates]) @ beta_y
        
        corr, _ = pearsonr(res_x, res_y)
        return float(corr)


class EnhancedCounterfactualReasoner:
    def __init__(self, causal_graph: Optional[CausalGraph] = None):
        self.causal_graph = causal_graph
        self.reasoning_history = deque(maxlen=100)

    def set_graph(self, causal_graph: CausalGraph):
        self.causal_graph = causal_graph

    def infer_counterfactual(self, current_state: np.ndarray,
                            intervention: Dict[int, float],
                            hypothetical_action: Dict[int, float],
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.causal_graph is None:
            return {
                "error": "No causal graph set",
                "reliability": 0.0
            }
        
        reliability = self._assess_reliability(current_state, intervention, context)
        
        effect_estimate = self._estimate_effect(current_state, intervention, hypothetical_action)
        
        result = {
            "counterfactual_state": effect_estimate,
            "intervention": intervention,
            "hypothetical_action": hypothetical_action,
            "reliability": reliability,
            "reliability_explanation": self._explain_reliability(reliability),
            "context": context
        }
        
        self.reasoning_history.append(result)
        
        return result

    def _estimate_effect(self, state: np.ndarray, intervention: Dict[int, float],
                        hypothetical: Dict[int, float]) -> np.ndarray:
        new_state = state.copy()
        
        for node, value in hypothetical.items():
            effect = self._propagate_effect(node, value)
            new_state = new_state + effect
        
        return new_state

    def _propagate_effect(self, start_node: int, value: float) -> np.ndarray:
        effect = np.zeros(self.causal_graph.num_nodes)
        visited = set()
        
        queue = [(start_node, value)]
        
        while queue:
            node, current_value = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            effect[node] += current_value
            
            for target in range(self.causal_graph.num_nodes):
                if self.causal_graph.adjacency_matrix[node, target] > 0:
                    strength = self.causal_graph.causal_strength[node, target]
                    propagated_value = current_value * strength
                    queue.append((target, propagated_value))
        
        return effect

    def _assess_reliability(self, state: np.ndarray, intervention: Dict[int, float],
                           context: Optional[Dict[str, Any]] = None) -> float:
        factors = []
        
        known_nodes = sum(1 for val in state if not np.isnan(val))
        factors.append(min(known_nodes / self.causal_graph.num_nodes, 1.0))
        
        edge_count = 0
        for node in intervention.keys():
            for target in range(self.causal_graph.num_nodes):
                if self.causal_graph.adjacency_matrix[node, target] > 0:
                    edge_count += 1
        factors.append(min(edge_count / self.causal_graph.num_nodes, 1.0))
        
        avg_strength = np.mean(self.causal_graph.causal_strength[self.causal_graph.causal_strength > 0])
        factors.append(avg_strength)
        
        return float(np.mean(factors))

    def _explain_reliability(self, reliability: float) -> str:
        if reliability > 0.8:
            return "推理结果高度可靠，因果图结构完整且干预路径清晰"
        elif reliability > 0.5:
            return "推理结果中等可靠，部分因果关系可能不够明确"
        else:
            return "推理结果可靠性较低，建议收集更多数据或验证因果假设"

    def simulate_multiple_scenarios(self, base_state: np.ndarray,
                                   scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        
        for scenario in scenarios:
            intervention = scenario.get("intervention", {})
            hypothetical = scenario.get("hypothetical_action", {})
            
            result = self.infer_counterfactual(base_state, intervention, hypothetical)
            result["scenario_name"] = scenario.get("name", "unnamed")
            
            results.append(result)
        
        return results

    def get_reasoning_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.reasoning_history)[-limit:]


class EnhancedCausalReasoner:
    def __init__(self):
        self.time_series_discovery = TimeSeriesCausalDiscovery()
        self.dml_effect = DoubleMLCausalEffect()
        self.chain_analyzer = CausalChainAnalyzer()
        self.transfer_learner = CausalTransferLearner()
        self.counterfactual_reasoner = EnhancedCounterfactualReasoner()
    
    def discover_time_series_causality(self, time_series: np.ndarray,
                                       variable_names: Optional[List[str]] = None,
                                       **kwargs) -> CausalGraph:
        return self.time_series_discovery.discover(time_series, variable_names, **kwargs)
    
    def estimate_causal_effect(self, treatment: np.ndarray, outcome: np.ndarray,
                               covariates: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.dml_effect.estimate(treatment, outcome, covariates, **kwargs)
    
    def analyze_causal_chains(self, graph: CausalGraph, **kwargs) -> Dict[str, Any]:
        return self.chain_analyzer.analyze(graph, **kwargs)
    
    def transfer_causal_model(self, source_data: np.ndarray, target_data: np.ndarray,
                              source_graph: Optional[CausalGraph] = None, **kwargs) -> Dict[str, Any]:
        return self.transfer_learner.transfer(source_data, target_data, source_graph, **kwargs)
    
    def reason_counterfactual(self, state: np.ndarray, intervention: Dict[int, float],
                              hypothetical: Optional[Dict[int, float]] = None, **kwargs) -> Dict[str, Any]:
        return self.counterfactual_reasoner.infer_counterfactual(state, intervention, hypothetical, **kwargs)


def get_enhanced_causal_reasoner() -> EnhancedCausalReasoner:
    return EnhancedCausalReasoner()
