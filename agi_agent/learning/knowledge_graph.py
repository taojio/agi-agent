import torch
import numpy as np
from collections import defaultdict
from typing import Dict
from ..config.settings import DEVICE


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.node_count = 0
        self.max_nodes = 1000
        self.max_edges_per_node = 20
        self.merge_threshold = 0.85
        self.cluster_interval = 100
        self.last_cluster_step = 0

    def add_node(self, feature: torch.Tensor, label: str = None):
        similar_node, similarity = self.find_similar_node(feature, threshold=self.merge_threshold)
        
        if similar_node is not None and similarity > self.merge_threshold:
            self.nodes[similar_node]["activation_count"] += 1
            self.nodes[similar_node]["feature"] = (
                self.nodes[similar_node]["feature"] * (self.nodes[similar_node]["activation_count"] - 1) +
                feature.detach().cpu()
            ) / self.nodes[similar_node]["activation_count"]
            return similar_node
        
        node_id = f"node_{self.node_count}"
        self.node_count += 1
        
        self.nodes[node_id] = {
            "feature": feature.detach().cpu(),
            "label": label,
            "activation_count": 1,
            "created_at": self.node_count,
            "last_accessed": self.node_count
        }
        
        if self.node_count > self.max_nodes:
            self._prune_low_activation_node()
        
        self.last_cluster_step += 1
        if self.last_cluster_step >= self.cluster_interval:
            self._cluster_nodes()
            self.last_cluster_step = 0
        
        return node_id

    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0):
        if len(self.edges[from_node]) < self.max_edges_per_node:
            self.edges[from_node].append({
                "target": to_node,
                "weight": weight,
                "count": 1
            })

    def update_edge(self, from_node: str, to_node: str, weight_delta: float = 0.1):
        for edge in self.edges[from_node]:
            if edge["target"] == to_node:
                edge["weight"] = min(1.0, max(0.0, edge["weight"] + weight_delta))
                edge["count"] += 1
                return

        self.add_edge(from_node, to_node, weight_delta)

    def update_node(self, node_id: str, attrs: Dict):
        """
        更新节点属性

        Args:
            node_id: 节点ID
            attrs: 待更新的属性字典，将合并到现有节点属性中
        """
        if node_id not in self.nodes:
            return False
        self.nodes[node_id].update(attrs)
        return True

    def query_neighbors(self, node_id: str, k: int = 5):
        if node_id not in self.edges:
            return []
        
        sorted_edges = sorted(self.edges[node_id], key=lambda x: x["weight"], reverse=True)
        return sorted_edges[:k]

    def find_similar_node(self, feature: torch.Tensor, threshold: float = 0.8):
        best_match = None
        best_similarity = 0.0
        
        for node_id, data in self.nodes.items():
            node_feat = data["feature"].to(DEVICE)
            
            if len(feature) != len(node_feat):
                min_len = min(len(feature), len(node_feat))
                feature_trimmed = feature[:min_len]
                node_feat_trimmed = node_feat[:min_len]
            else:
                feature_trimmed = feature
                node_feat_trimmed = node_feat
            
            similarity = torch.nn.functional.cosine_similarity(
                feature_trimmed, node_feat_trimmed, dim=-1
            ).item()
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = node_id
        
        return best_match, best_similarity

    def _prune_low_activation_node(self):
        if not self.nodes:
            return
        
        worst_node = None
        worst_score = float('inf')
        
        for node_id, data in self.nodes.items():
            score = data["created_at"] / (data["activation_count"] + 1)
            if score < worst_score:
                worst_score = score
                worst_node = node_id
        
        if worst_node is not None:
            del self.nodes[worst_node]
            
            if worst_node in self.edges:
                del self.edges[worst_node]
            
            for node_id in list(self.edges.keys()):
                self.edges[node_id] = [e for e in self.edges[node_id] if e["target"] != worst_node]

    def _cluster_nodes(self):
        if len(self.nodes) < 2:
            return
        
        node_ids = list(self.nodes.keys())
        clusters = []
        
        for i in range(len(node_ids)):
            node_id = node_ids[i]
            already_clustered = False
            
            for cluster in clusters:
                representative = cluster[0]
                rep_feat = self.nodes[representative]["feature"].to(DEVICE)
                curr_feat = self.nodes[node_id]["feature"].to(DEVICE)
                
                min_len = min(len(rep_feat), len(curr_feat))
                similarity = torch.nn.functional.cosine_similarity(
                    rep_feat[:min_len], curr_feat[:min_len], dim=-1
                ).item()
                
                if similarity > self.merge_threshold * 0.9:
                    cluster.append(node_id)
                    already_clustered = True
                    break
            
            if not already_clustered:
                clusters.append([node_id])
        
        for cluster in clusters:
            if len(cluster) > 1:
                self._merge_cluster(cluster)

    def _merge_cluster(self, cluster):
        if len(cluster) < 2:
            return
        
        main_node = cluster[0]
        total_activation = self.nodes[main_node]["activation_count"]
        merged_feature = self.nodes[main_node]["feature"].clone()
        
        for node_id in cluster[1:]:
            total_activation += self.nodes[node_id]["activation_count"]
            merged_feature += self.nodes[node_id]["feature"] * self.nodes[node_id]["activation_count"]
            
            for edge in self.edges.get(node_id, []):
                self.update_edge(main_node, edge["target"], edge["weight"])
            
            for from_node, edges in list(self.edges.items()):
                for edge in edges:
                    if edge["target"] == node_id:
                        edge["target"] = main_node
            
            if node_id in self.edges:
                del self.edges[node_id]
            
            del self.nodes[node_id]
        
        merged_feature /= total_activation
        self.nodes[main_node]["feature"] = merged_feature
        self.nodes[main_node]["activation_count"] = total_activation

    def get_summary(self):
        total_edges = sum(len(edges) for edges in self.edges.values())
        avg_activation = np.mean([n["activation_count"] for n in self.nodes.values()]) if self.nodes else 0.0
        return {
            "nodes": len(self.nodes),
            "edges": total_edges,
            "max_nodes": self.max_nodes,
            "avg_activation": avg_activation
        }