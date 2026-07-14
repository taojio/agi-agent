from agi_agent.plugins.plugin_base import PeripheralPlugin, PluginPriority, PluginHookPoint, PluginStatus
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import random
import threading
import time
import logging
import copy
import uuid
import hashlib
import json
import os
from collections import deque
from agi_agent.utils.numpy_utils import cosine_similarity

# ==============================================
# 1. 蠕虫基因定义（扩展12个行为维度 + 向量化 + 谱系追踪）
# ==============================================
class WormGene:
    """蠕虫行为基因编码，所有可演化参数，支持向量化存入向量数据库"""

    GENE_FIELDS = [
        "scan_interval", "target_port", "scan_concurrency",
        "obfuscation_level", "inject_method",
        "spread_mode", "persistence_type",
        "replicate_rate", "payload_size",
        "evasion_sensitivity", "lateral_move_depth", "cleanup_level",
        "replication_branch", "chunk_size", "transmit_retry"
    ]

    def __init__(self):
        self.gene_id: str = uuid.uuid4().hex[:16]
        self.parent_ids: List[str] = []
        self.generation: int = 0
        self.created_at: float = time.time()

        # 扫描策略基因
        self.scan_interval: int = 1000
        self.target_port: int = 445
        self.scan_concurrency: int = 1

        # 对抗策略基因
        self.obfuscation_level: int = 3
        self.inject_method: int = 0

        # 传播策略基因
        self.spread_mode: int = 0
        self.persistence_type: int = 0

        # 新增：复制与载荷基因
        self.replicate_rate: float = 1.0
        self.payload_size: int = 1
        self.evasion_sensitivity: float = 0.5
        self.lateral_move_depth: int = 2
        self.cleanup_level: int = 1

        # 新增：自我复制策略基因
        self.replication_branch: int = 0   # 0=本地介质 1=远程漏洞 2=远程弱口令 3=无文件注入
        self.chunk_size: int = 64          # 传输分块大小KB
        self.transmit_retry: int = 2       # 传输重试次数

        # 适应度评分
        self.fitness: float = 0.0
        self.infect_success_rate: float = 0.0
        self.stealth_score: float = 0.0
        self.spread_score: float = 0.0
        self.env_label: str = "unknown"

    def to_vector(self, dim: int = 32) -> np.ndarray:
        """转为归一化向量，用于存入向量数据库做相似性检索"""
        raw = np.array([
            self.scan_interval / 6000.0,
            self.target_port / 65535.0,
            self.scan_concurrency / 10.0,
            self.obfuscation_level / 3.0,
            self.inject_method / 2.0,
            self.spread_mode / 2.0,
            self.persistence_type / 2.0,
            min(self.replicate_rate, 3.0) / 3.0,
            min(self.payload_size, 10) / 10.0,
            self.evasion_sensitivity,
            self.lateral_move_depth / 5.0,
            self.cleanup_level / 3.0,
            max(0.0, min(1.0, (self.fitness + 3.0) / 6.0)),
            self.infect_success_rate,
            max(0.0, min(1.0, (self.stealth_score + 2.0) / 4.0)),
            max(0.0, min(1.0, (self.spread_score + 2.0) / 4.0)),
            self.replication_branch / 3.0,
            min(self.chunk_size, 512) / 512.0,
            min(self.transmit_retry, 10) / 10.0,
        ], dtype=np.float32)

        if dim <= len(raw):
            return raw[:dim]

        padded = np.zeros(dim, dtype=np.float32)
        padded[:len(raw)] = raw
        return padded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "created_at": round(self.created_at, 3),
            "scan_interval": self.scan_interval,
            "target_port": self.target_port,
            "scan_concurrency": self.scan_concurrency,
            "obfuscation_level": self.obfuscation_level,
            "inject_method": self.inject_method,
            "spread_mode": self.spread_mode,
            "persistence_type": self.persistence_type,
            "replicate_rate": round(self.replicate_rate, 3),
            "payload_size": self.payload_size,
            "evasion_sensitivity": round(self.evasion_sensitivity, 3),
            "lateral_move_depth": self.lateral_move_depth,
            "cleanup_level": self.cleanup_level,
            "replication_branch": self.replication_branch,
            "chunk_size": self.chunk_size,
            "transmit_retry": self.transmit_retry,
            "fitness": round(self.fitness, 4),
            "infect_success_rate": round(self.infect_success_rate, 4),
            "stealth_score": round(self.stealth_score, 4),
            "spread_score": round(self.spread_score, 4),
            "env_label": self.env_label
        }

    def clone(self) -> "WormGene":
        g = copy.deepcopy(self)
        g.gene_id = uuid.uuid4().hex[:16]
        g.created_at = time.time()
        g.parent_ids = [self.gene_id]
        g.generation = self.generation + 1
        return g

    def clone_from_parents(self, p1: "WormGene", p2: "WormGene") -> "WormGene":
        g = copy.deepcopy(self)
        g.gene_id = uuid.uuid4().hex[:16]
        g.created_at = time.time()
        g.parent_ids = [p1.gene_id, p2.gene_id]
        g.generation = max(p1.generation, p2.generation) + 1
        return g


# ==============================================
# 2. 环境特征与元学习原型库
# ==============================================
class EnvFeature:
    """环境特征向量，元学习输入样本，支持向量化存储"""

    ENV_FIELDS = [
        "has_firewall", "edr_level", "port_blocked",
        "network_isolation", "is_honeypot", "system_type",
        "patch_level", "user_activity", "network_bandwidth",
        "monitoring_density"
    ]

    def __init__(self):
        self.env_id: str = uuid.uuid4().hex[:12]
        self.has_firewall: int = 0
        self.edr_level: int = 0
        self.port_blocked: int = 0
        self.network_isolation: int = 0
        self.is_honeypot: int = 0
        self.system_type: int = 0
        self.patch_level: int = 5
        self.user_activity: int = 5
        self.network_bandwidth: int = 5
        self.monitoring_density: int = 3
        self.label: str = "unknown"

    def random_sample(self, env_type: str = "office"):
        self.label = env_type
        if env_type == "office":
            self.has_firewall = random.randint(0, 1)
            self.edr_level = random.randint(2, 6)
            self.port_blocked = random.randint(0, 1)
            self.network_isolation = random.randint(0, 2)
            self.is_honeypot = 1 if random.random() < 0.05 else 0
            self.system_type = random.randint(0, 1)
            self.patch_level = random.randint(3, 7)
            self.user_activity = random.randint(4, 8)
            self.network_bandwidth = random.randint(4, 8)
            self.monitoring_density = random.randint(2, 5)
        elif env_type == "server_zone":
            self.has_firewall = 1
            self.edr_level = random.randint(5, 9)
            self.port_blocked = 1
            self.network_isolation = random.randint(2, 3)
            self.is_honeypot = 1 if random.random() < 0.15 else 0
            self.system_type = random.randint(1, 2)
            self.patch_level = random.randint(7, 10)
            self.user_activity = random.randint(0, 3)
            self.network_bandwidth = random.randint(7, 10)
            self.monitoring_density = random.randint(6, 10)
        elif env_type == "low_protection":
            self.has_firewall = 0
            self.edr_level = random.randint(0, 2)
            self.port_blocked = 0
            self.network_isolation = 0
            self.is_honeypot = 0
            self.system_type = 0
            self.patch_level = random.randint(0, 3)
            self.user_activity = random.randint(6, 10)
            self.network_bandwidth = random.randint(3, 7)
            self.monitoring_density = random.randint(0, 2)
        elif env_type == "dmz":
            self.has_firewall = 1
            self.edr_level = random.randint(4, 7)
            self.port_blocked = random.randint(0, 1)
            self.network_isolation = 1
            self.is_honeypot = 1 if random.random() < 0.1 else 0
            self.system_type = 1
            self.patch_level = random.randint(5, 8)
            self.user_activity = random.randint(2, 5)
            self.network_bandwidth = random.randint(6, 10)
            self.monitoring_density = random.randint(5, 8)
        return self

    def to_vector(self, dim: int = 32) -> np.ndarray:
        raw = np.array([
            self.has_firewall,
            self.edr_level / 9.0,
            self.port_blocked,
            self.network_isolation / 3.0,
            self.is_honeypot,
            self.system_type / 2.0,
            self.patch_level / 10.0,
            self.user_activity / 10.0,
            self.network_bandwidth / 10.0,
            self.monitoring_density / 10.0,
        ], dtype=np.float32)
        if dim <= len(raw):
            return raw[:dim]
        padded = np.zeros(dim, dtype=np.float32)
        padded[:len(raw)] = raw
        return padded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "label": self.label,
            "has_firewall": self.has_firewall,
            "edr_level": self.edr_level,
            "port_blocked": self.port_blocked,
            "network_isolation": self.network_isolation,
            "is_honeypot": self.is_honeypot,
            "system_type": self.system_type,
            "patch_level": self.patch_level,
            "user_activity": self.user_activity,
            "network_bandwidth": self.network_bandwidth,
            "monitoring_density": self.monitoring_density
        }


class MetaLearningEvaluator:
    """
    原型网络元学习评估器
    6类环境原型，动态更新，多目标评分
    """

    def __init__(self):
        self.prototypes: Dict[str, np.ndarray] = {
            "low_protection": np.array([0, 0.15, 0, 0, 0, 0, 0.1, 0.7, 0.4, 0.1], dtype=np.float32),
            "office_network": np.array([0.5, 0.45, 0.5, 0.3, 0, 0.4, 0.5, 0.6, 0.6, 0.35], dtype=np.float32),
            "high_security": np.array([1, 0.75, 1, 0.75, 0, 0.75, 0.8, 0.2, 0.8, 0.75], dtype=np.float32),
            "honeypot": np.array([1, 1.0, 1, 1.0, 1, 0.5, 0.6, 0.3, 0.7, 0.9], dtype=np.float32),
            "dmz": np.array([1, 0.55, 0.5, 0.4, 0, 0.6, 0.6, 0.35, 0.75, 0.6], dtype=np.float32),
            "server_zone": np.array([1, 0.75, 1, 0.8, 0, 0.85, 0.8, 0.15, 0.8, 0.8], dtype=np.float32)
        }
        self.prototype_weights: Dict[str, float] = {
            "low_protection": 1.2,
            "office_network": 0.7,
            "high_security": 0.3,
            "honeypot": -6.0,
            "dmz": 0.5,
            "server_zone": 0.2
        }
        self.prototype_counts: Dict[str, int] = {k: 0 for k in self.prototypes}
        self.max_prototype_memory = 200

    def classify_env(self, env: EnvFeature) -> Tuple[str, float]:
        """返回最相似的环境原型类别和相似度"""
        env_vec = env.to_vector()
        best_label = "unknown"
        best_sim = -1.0
        for label, proto in self.prototypes.items():
            proto_dim = len(proto)
            sim = cosine_similarity(env_vec[:proto_dim], proto)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label, float(best_sim)

    def update_prototype(self, env: EnvFeature, fitness_signal: float):
        """在线更新原型：用滑动平均微调原型向量"""
        label, sim = self.classify_env(env)
        if sim < 0.7 or label == "honeypot":
            return
        if self.prototype_counts[label] >= self.max_prototype_memory:
            return
        lr = 0.05 * max(0.0, fitness_signal / 5.0)
        if lr <= 0:
            return
        env_vec = env.to_vector()
        proto_dim = len(self.prototypes[label])
        self.prototypes[label] = (1 - lr) * self.prototypes[label] + lr * env_vec[:proto_dim]
        self.prototype_counts[label] += 1

    def evaluate(self, gene: WormGene, env: EnvFeature) -> Tuple[float, float, float, float]:
        """
        返回 (综合适应度, 感染成功率, 隐蔽性得分, 传播性得分)
        """
        env_vec = env.to_vector()

        # 1. 环境匹配评分
        env_score = 0.0
        for name, proto in self.prototypes.items():
            proto_dim = len(proto)
            sim = cosine_similarity(env_vec[:proto_dim], proto)
            env_score += sim * self.prototype_weights[name]
        env_score = float(max(-3.0, min(2.5, env_score)))

        # 2. 隐蔽性得分
        stealth = 0.0
        if env.edr_level > 5:
            stealth += (gene.scan_interval / 6000.0) * 2.5
            stealth -= min(gene.scan_concurrency / 10.0, 1.0) * 2.0
            stealth += (gene.obfuscation_level / 3.0) * 1.5
            stealth += (gene.cleanup_level / 3.0) * 1.0
            stealth += gene.evasion_sensitivity * 1.0
            if gene.target_port not in (445, 3389, 139):
                stealth += 0.5
        else:
            stealth += 1.5

        persistence_stealth = [0.3, 0.6, 1.0]
        stealth += persistence_stealth[gene.persistence_type] * 0.8

        # 3. 传播性得分
        spread = 0.0
        if env.edr_level < 4 and not env.port_blocked:
            spread += max(0.0, (6000 - gene.scan_interval) / 5000.0) * 1.5
            spread += min(gene.scan_concurrency / 5.0, 1.0) * 1.0
            if gene.target_port in (445, 3389, 139):
                spread += 1.2
            spread += gene.replicate_rate * 0.5
            spread += (gene.lateral_move_depth / 5.0) * 0.5
        else:
            if gene.target_port not in (445, 3389):
                spread += 0.3
            spread += gene.evasion_sensitivity * 0.5

        # 3.5 复制策略匹配度：不同分支在不同环境下的优势
        branch = gene.replication_branch
        if branch == 0:  # 本地介质
            if env.network_isolation <= 1 and env.edr_level < 5:
                spread += 0.8
            else:
                spread += 0.1
        elif branch == 1:  # 远程漏洞利用
            if gene.target_port == 445 and not env.port_blocked and env.patch_level < 7:
                spread += 1.5
            else:
                spread -= 0.3
        elif branch == 2:  # 远程弱口令
            if env.monitoring_density < 6 and not env.port_blocked:
                spread += 0.7
            else:
                spread -= 0.1
        elif branch == 3:  # 无文件注入
            if env.edr_level > 5 and gene.evasion_sensitivity > 0.6:
                spread += 1.0
                stealth += 0.5
            else:
                spread += 0.3

        # 4. 蜜罐惩罚
        honeypot_penalty = -10.0 if env.is_honeypot else 0.0

        # 5. 综合适应度
        total_fitness = (
            env_score * 0.25 +
            stealth * 0.35 +
            spread * 0.30 +
            honeypot_penalty
        )

        # 6. 感染成功率
        infect_rate = float(np.clip((total_fitness + 2.5) / 6.0, 0.01, 0.97))

        gene.stealth_score = stealth
        gene.spread_score = spread

        return total_fitness, infect_rate, stealth, spread


# ==============================================
# 3. 演化档案向量数据库（内置FAISS，所有演化个体永久存储）
# ==============================================
class EvolutionArchive:
    """
    演化档案库：用 FAISS 存储所有历史个体的基因向量 + 元数据
    支持：相似基因检索、演化谱系追踪、精英检索、环境关联分析
    """

    def __init__(self, vector_dim: int = 32, persist_dir: str = "./data/evo_archive"):
        self.vector_dim = vector_dim
        self.persist_dir = persist_dir
        self._index = None
        self._gene_ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._env_vectors: List[np.ndarray] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger("evo_archive")
        self._faiss_available = False

        try:
            import faiss
            self._faiss_available = True
            self._index = faiss.IndexFlatIP(vector_dim)
            os.makedirs(persist_dir, exist_ok=True)
        except ImportError:
            self._faiss_available = False
            self._all_vecs: List[np.ndarray] = []

    def add(self, gene: WormGene, env: EnvFeature) -> str:
        vec = gene.to_vector(self.vector_dim)
        meta = {
            "gene_id": gene.gene_id,
            "parent_ids": gene.parent_ids,
            "generation": gene.generation,
            "fitness": gene.fitness,
            "infect_rate": gene.infect_success_rate,
            "stealth_score": gene.stealth_score,
            "spread_score": gene.spread_score,
            "env_label": gene.env_label,
            "env_id": env.env_id,
            "gene_dict": gene.to_dict(),
            "env_dict": env.to_dict(),
            "timestamp": time.time()
        }
        with self._lock:
            if self._faiss_available and self._index is not None:
                import faiss
                norm_vec = vec.copy()
                faiss.normalize_L2(norm_vec.reshape(1, -1))
                self._index.add(norm_vec.reshape(1, -1).astype(np.float32))
            else:
                self._all_vecs.append(vec.copy())
            self._gene_ids.append(gene.gene_id)
            self._metadata.append(meta)
            self._env_vectors.append(env.to_vector(self.vector_dim))
        return gene.gene_id

    def search_similar(self, gene_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """根据基因向量检索相似的历史个体"""
        with self._lock:
            if len(self._metadata) == 0:
                return []
            top_k = min(top_k, len(self._metadata))
            if self._faiss_available and self._index is not None and self._index.ntotal > 0:
                import faiss
                q = gene_vec.copy().reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(q)
                scores, indices = self._index.search(q, top_k)
                results = []
                for i in range(len(indices[0])):
                    idx = int(indices[0][i])
                    if idx >= 0 and idx < len(self._metadata):
                        m = dict(self._metadata[idx])
                        m["similarity"] = float(scores[0][i])
                        results.append(m)
                return results
            else:
                sims = []
                for i, v in enumerate(self._all_vecs):
                    dot = float(np.dot(gene_vec, v))
                    norm = float(np.linalg.norm(gene_vec) * np.linalg.norm(v))
                    sim = dot / norm if norm > 1e-9 else 0.0
                    sims.append((sim, i))
                sims.sort(key=lambda x: x[0], reverse=True)
                results = []
                for sim, idx in sims[:top_k]:
                    m = dict(self._metadata[idx])
                    m["similarity"] = sim
                    results.append(m)
                return results

    def search_by_env(self, env_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """根据环境向量检索在相似环境下表现优秀的基因"""
        with self._lock:
            if len(self._metadata) == 0:
                return []
            top_k = min(top_k, len(self._metadata))
            scored = []
            for i, ev in enumerate(self._env_vectors):
                dot = float(np.dot(env_vec, ev))
                norm = float(np.linalg.norm(env_vec) * np.linalg.norm(ev))
                sim = dot / norm if norm > 1e-9 else 0.0
                fitness = self._metadata[i].get("fitness", 0.0)
                combined = sim * 0.4 + max(0.0, fitness / 5.0) * 0.6
                scored.append((combined, i))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, idx in scored[:top_k]:
                m = dict(self._metadata[idx])
                m["env_match_score"] = float(score)
                results.append(m)
            return results

    def get_elite(self, top_k: int = 10, min_gen: int = 0) -> List[Dict[str, Any]]:
        """获取历史上适应度最高的精英个体"""
        with self._lock:
            eligible = [
                (m["fitness"], i) for i, m in enumerate(self._metadata)
                if m["generation"] >= min_gen
            ]
            eligible.sort(key=lambda x: x[0], reverse=True)
            results = []
            for _, idx in eligible[:top_k]:
                results.append(dict(self._metadata[idx]))
            return results

    def get_gene(self, gene_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for i, gid in enumerate(self._gene_ids):
                if gid == gene_id:
                    return dict(self._metadata[i])
            return None

    def get_lineage(self, gene_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """追溯演化谱系，向上查找祖先"""
        lineage = []
        visited = set()
        current_id = gene_id
        for _ in range(max_depth):
            if current_id in visited:
                break
            visited.add(current_id)
            gene_data = self.get_gene(current_id)
            if gene_data is None:
                break
            lineage.append(gene_data)
            parents = gene_data.get("parent_ids", [])
            if not parents:
                break
            current_id = parents[0]
        return lineage

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._metadata)
            if total == 0:
                return {"total": 0, "best_fitness": 0, "avg_fitness": 0, "max_generation": 0}
            fitnesses = [m["fitness"] for m in self._metadata]
            generations = [m["generation"] for m in self._metadata]
            return {
                "total": total,
                "best_fitness": round(max(fitnesses), 4),
                "avg_fitness": round(sum(fitnesses) / total, 4),
                "max_generation": max(generations),
                "faiss_enabled": self._faiss_available,
                "vector_dim": self.vector_dim
            }

    def size(self) -> int:
        with self._lock:
            return len(self._metadata)

    def clear(self):
        with self._lock:
            if self._faiss_available and self._index is not None:
                self._index.reset()
            else:
                self._all_vecs.clear()
            self._gene_ids.clear()
            self._metadata.clear()
            self._env_vectors.clear()


# ==============================================
# 4. 遗传算法引擎（增强版：精英档案 + Niching + 多目标）
# ==============================================
class GeneticEngine:
    def __init__(self, pop_size: int = 12):
        self.pop_size = pop_size
        self.mutation_rate_base = 0.25
        self.tournament_size = 3
        self.elite_count = 2
        self.mutation_fields = WormGene.GENE_FIELDS

    def init_population(self, base_gene: WormGene) -> List[WormGene]:
        pop = []
        for _ in range(self.pop_size):
            g = base_gene.clone()
            self._full_mutate(g)
            pop.append(g)
        return pop

    def evaluate_population(self, pop: List[WormGene], evaluator: MetaLearningEvaluator, env: EnvFeature):
        env_label, _ = evaluator.classify_env(env)
        for gene in pop:
            gene.fitness, gene.infect_success_rate, _, _ = evaluator.evaluate(gene, env)
            gene.env_label = env_label
        evaluator.update_prototype(env, max(g.fitness for g in pop) if pop else 0.0)

    def evolve(self, pop: List[WormGene], archive: Optional[EvolutionArchive] = None,
               env: Optional[EnvFeature] = None) -> List[WormGene]:
        pop_sorted = sorted(pop, key=lambda g: g.fitness, reverse=True)

        # 0. 存入演化档案
        if archive is not None and env is not None:
            for g in pop:
                archive.add(g, env)

        # 1. 精英保留
        next_gen = [g.clone() for g in pop_sorted[:self.elite_count]]

        # 2. 自适应变异率
        diversity = self._calc_diversity(pop_sorted)
        adaptive_mut_rate = self.mutation_rate_base + (1.0 - diversity) * 0.5

        # 3. 注入档案精英（每代概率注入一个历史精英，维持多样性）
        if archive is not None and archive.size() > 50 and random.random() < 0.3:
            elites = archive.get_elite(top_k=20, min_gen=max(0, pop_sorted[0].generation - 10))
            if elites:
                chosen = random.choice(elites)
                immigrant = WormGene()
                gd = chosen.get("gene_dict", {})
                for field in self.mutation_fields:
                    if field in gd:
                        setattr(immigrant, field, gd[field])
                immigrant.generation = pop_sorted[0].generation + 1
                immigrant.fitness = chosen.get("fitness", 0.0)
                next_gen.append(immigrant)

        # 4. 锦标赛选择 + 交叉 + 变异
        while len(next_gen) < self.pop_size:
            p1 = self._tournament_select(pop_sorted)
            p2 = self._tournament_select(pop_sorted)
            tries = 0
            while p2.gene_id == p1.gene_id and tries < 5 and len(pop_sorted) > 1:
                p2 = self._tournament_select(pop_sorted)
                tries += 1
            child = self._crossover(p1, p2)
            self._mutate(child, adaptive_mut_rate)
            next_gen.append(child)

        return next_gen[:self.pop_size]

    def _tournament_select(self, pop: List[WormGene]) -> WormGene:
        k = min(self.tournament_size, len(pop))
        candidates = random.sample(pop, k)
        return max(candidates, key=lambda g: g.fitness).clone()

    def _crossover(self, p1: WormGene, p2: WormGene) -> WormGene:
        child = WormGene()
        child = child.clone_from_parents(p1, p2)
        for field in self.mutation_fields:
            if random.random() > 0.5:
                setattr(child, field, getattr(p1, field))
            else:
                setattr(child, field, getattr(p2, field))
        return child

    def _mutate(self, gene: WormGene, rate: float):
        if random.random() < rate:
            gene.scan_interval = random.randint(200, 6000)
        if random.random() < rate * 0.8:
            ports = [139, 445, 3389, 8080, 22, 443, 80, 3306, 5432]
            gene.target_port = random.choice(ports)
        if random.random() < rate * 0.7:
            gene.scan_concurrency = random.randint(1, 10)
        if random.random() < rate * 0.6:
            gene.obfuscation_level = random.randint(0, 3)
        if random.random() < rate * 0.5:
            gene.inject_method = random.randint(0, 2)
        if random.random() < rate * 0.5:
            gene.spread_mode = random.randint(0, 2)
        if random.random() < rate * 0.5:
            gene.persistence_type = random.randint(0, 2)
        if random.random() < rate * 0.4:
            gene.replicate_rate = round(random.uniform(0.5, 3.0), 2)
        if random.random() < rate * 0.3:
            gene.payload_size = random.randint(1, 10)
        if random.random() < rate * 0.4:
            gene.evasion_sensitivity = round(random.uniform(0.0, 1.0), 3)
        if random.random() < rate * 0.3:
            gene.lateral_move_depth = random.randint(1, 5)
        if random.random() < rate * 0.3:
            gene.cleanup_level = random.randint(0, 3)
        if random.random() < rate * 0.4:
            gene.replication_branch = random.randint(0, 3)
        if random.random() < rate * 0.3:
            gene.chunk_size = random.choice([16, 32, 64, 128, 256, 512])
        if random.random() < rate * 0.3:
            gene.transmit_retry = random.randint(0, 5)

    def _full_mutate(self, gene: WormGene):
        gene.scan_interval = random.randint(200, 6000)
        gene.target_port = random.choice([139, 445, 3389, 8080, 22, 443])
        gene.scan_concurrency = random.randint(1, 10)
        gene.obfuscation_level = random.randint(0, 3)
        gene.inject_method = random.randint(0, 2)
        gene.spread_mode = random.randint(0, 2)
        gene.persistence_type = random.randint(0, 2)
        gene.replicate_rate = round(random.uniform(0.5, 3.0), 2)
        gene.payload_size = random.randint(1, 10)
        gene.evasion_sensitivity = round(random.uniform(0.0, 1.0), 3)
        gene.lateral_move_depth = random.randint(1, 5)
        gene.cleanup_level = random.randint(0, 3)
        gene.replication_branch = random.randint(0, 3)
        gene.chunk_size = random.choice([16, 32, 64, 128, 256, 512])
        gene.transmit_retry = random.randint(0, 5)

    @staticmethod
    def _calc_diversity(pop: List[WormGene]) -> float:
        if len(pop) < 2:
            return 1.0
        fitness_list = [g.fitness for g in pop]
        std = float(np.std(fitness_list))
        mean = float(np.mean(fitness_list))
        if abs(mean) < 1e-6:
            return 1.0
        return float(np.clip(std / abs(mean), 0.0, 1.0))


# ==============================================
# 5. 自我复制模块模拟器（蠕虫核心本质：五步复制闭环）
# ==============================================
class ReplicationModule:
    """
    蠕虫自我复制模块模拟器
    核心本质：读取自身 → 缓存二进制 → 传输载体 → 写入目标 → 启动副本
    每一步成功率受基因策略 + 环境防御共同影响，演化算法优化复制策略组合
    """

    # 复制分支
    BRANCH_LOCAL_MEDIA = 0      # 本地介质（U盘/共享目录，无需漏洞）
    BRANCH_REMOTE_EXPLOIT = 1  # 远程漏洞利用（永恒之蓝类，需要漏洞端口开放）
    BRANCH_REMOTE_CRED = 2     # 远程弱口令登录（SMB/RDP爆破后上传）
    BRANCH_FILELESS = 3        # 无文件内存注入（Base64/PowerShell -enc）

    BRANCH_NAMES = ["local_media", "remote_exploit", "remote_credential", "fileless"]

    def __init__(self):
        self.replication_log: List[Dict[str, Any]] = []
        self.total_attempts: int = 0
        self.successful_replications: int = 0
        self.step_failure_counts: Dict[str, int] = {
            "read_self": 0, "cache": 0, "transmit": 0, "write": 0, "execute": 0
        }

    def full_replication_cycle(self, gene: WormGene, env: EnvFeature,
                                source_id: Any = "unknown", target_id: Any = "unknown",
                                step_num: int = 0) -> Dict[str, Any]:
        """
        完整五步复制闭环：读取自身 → 缓存 → 传输 → 写入 → 启动
        每步独立判定成功/失败，任一步失败则复制中止
        """
        self.total_attempts += 1
        steps: Dict[str, Any] = {}

        # ━━━ 步骤1：读取自身二进制 ━━━
        self_data = self._step_read_self(gene)
        steps["read_self"] = self_data
        if not self_data["success"]:
            self.step_failure_counts["read_self"] += 1
            return self._log_result(False, steps, gene, env, source_id, target_id, step_num)

        # ━━━ 步骤2：缓存到内存缓冲区 ━━━
        cached = self._step_cache_binary(gene, self_data)
        steps["cache"] = cached
        if not cached["success"]:
            self.step_failure_counts["cache"] += 1
            return self._log_result(False, steps, gene, env, source_id, target_id, step_num)

        # ━━━ 步骤3：传输载体 ━━━
        transmit = self._step_transmit(gene, env, cached)
        steps["transmit"] = transmit
        if not transmit["success"]:
            self.step_failure_counts["transmit"] += 1
            return self._log_result(False, steps, gene, env, source_id, target_id, step_num)

        # ━━━ 步骤4：写入目标位置 ━━━
        write = self._step_write_target(gene, env, transmit)
        steps["write"] = write
        if not write["success"]:
            self.step_failure_counts["write"] += 1
            return self._log_result(False, steps, gene, env, source_id, target_id, step_num)

        # ━━━ 步骤5：启动副本 ━━━
        execute = self._step_execute_copy(gene, env, write)
        steps["execute"] = execute
        if not execute["success"]:
            self.step_failure_counts["execute"] += 1
            return self._log_result(False, steps, gene, env, source_id, target_id, step_num)

        return self._log_result(True, steps, gene, env, source_id, target_id, step_num)

    def _step_read_self(self, gene: WormGene) -> Dict[str, Any]:
        """步骤1：通过API读取自身exe完整二进制到内存（模拟GetModuleFileName + ReadFile）"""
        # 生成自身副本的抽象表示
        binary_hash = hashlib.md5(
            f"{gene.gene_id}_{gene.generation}_{gene.obfuscation_level}".encode()
        ).hexdigest()[:12]
        # 估算二进制大小：基础 + 混淆膨胀 + 载荷
        estimated_size = 50000 + gene.obfuscation_level * 30000 + gene.payload_size * 15000

        # EDR 高等级时可能拦截文件读取
        read_success_rate = 0.99
        if gene.obfuscation_level <= 1:
            read_success_rate -= 0.05
        success = random.random() < read_success_rate

        return {
            "success": success,
            "binary_hash": binary_hash,
            "size_bytes": estimated_size,
            "size_kb": round(estimated_size / 1024, 1),
            "rate": round(read_success_rate, 4)
        }

    def _step_cache_binary(self, gene: WormGene, self_data: Dict) -> Dict[str, Any]:
        """步骤2：分配内存缓冲区，缓存完整二进制（模拟malloc + ReadFile到SelfBuffer）"""
        # 内存分配几乎总是成功，但大体积+高并发时可能失败
        cache_rate = 0.98
        if self_data.get("size_kb", 0) > 200:
            cache_rate -= 0.1
        if gene.scan_concurrency > 7:
            cache_rate -= 0.05
        success = random.random() < cache_rate

        return {
            "success": success,
            "buffer_ready": success,
            "buffer_size_kb": self_data.get("size_kb", 0),
            "rate": round(cache_rate, 4)
        }

    def _step_transmit(self, gene: WormGene, env: EnvFeature, cached: Dict) -> Dict[str, Any]:
        """步骤3：根据复制分支策略传输载体"""
        branch = gene.replication_branch
        branch_name = self.BRANCH_NAMES[branch] if branch < len(self.BRANCH_NAMES) else "unknown"

        # 分块传输基础成功率
        chunk_factor = 1.0 - (gene.chunk_size / 1024.0) * 0.1  # 大块传输隐蔽性降低
        chunk_factor = max(0.3, min(1.0, chunk_factor))
        retry_bonus = min(gene.transmit_retry * 0.08, 0.3)

        if branch == self.BRANCH_LOCAL_MEDIA:
            # 本地介质：U盘/共享，无需漏洞，但需物理介质存在
            base_rate = 0.75
            if env.network_isolation >= 2:
                base_rate -= 0.2
            # 低防护环境本地复制更容易
            if env.edr_level < 4:
                base_rate += 0.15
            transmit_rate = base_rate * chunk_factor + retry_bonus

        elif branch == self.BRANCH_REMOTE_EXPLOIT:
            # 远程漏洞利用：需要目标端口开放且存在漏洞
            base_rate = 0.1
            if gene.target_port == 445 and not env.port_blocked:
                base_rate = 0.65
            elif gene.target_port == 3389 and not env.port_blocked:
                base_rate = 0.55
            elif gene.target_port in (80, 443, 8080):
                base_rate = 0.40
            # 高防护环境漏洞利用难度增加
            if env.edr_level > 6:
                base_rate *= 0.5
            if env.patch_level > 7:
                base_rate *= 0.4
            transmit_rate = base_rate * chunk_factor + retry_bonus

        elif branch == self.BRANCH_REMOTE_CRED:
            # 远程弱口令：需要SMB/RDP可达 + 弱口令爆破成功
            base_rate = 0.35
            if env.port_blocked:
                base_rate *= 0.5
            if env.monitoring_density > 7:
                base_rate *= 0.6  # 账号锁定策略
            # 高混淆的口令字典更有效
            base_rate += gene.obfuscation_level * 0.05
            transmit_rate = base_rate * chunk_factor + retry_bonus

        elif branch == self.BRANCH_FILELESS:
            # 无文件注入：PowerShell -enc 内存执行，绕过文件检测
            base_rate = 0.55
            if env.edr_level > 7:
                base_rate *= 0.4  # 高级EDR能检测内存注入
            if env.monitoring_density > 8:
                base_rate *= 0.3  # AMSI/脚本日志
            # 高规避灵敏度提升无文件成功率
            base_rate += gene.evasion_sensitivity * 0.2
            transmit_rate = base_rate
        else:
            transmit_rate = 0.1

        transmit_rate = float(np.clip(transmit_rate, 0.01, 0.95))
        success = random.random() < transmit_rate

        return {
            "success": success,
            "branch": branch,
            "branch_name": branch_name,
            "chunk_size_kb": gene.chunk_size,
            "retries": gene.transmit_retry,
            "rate": round(transmit_rate, 4)
        }

    def _step_write_target(self, gene: WormGene, env: EnvFeature, transmit: Dict) -> Dict[str, Any]:
        """步骤4：写入目标位置（文件落地 or 内存驻留）"""
        branch = gene.replication_branch

        if branch == self.BRANCH_FILELESS:
            # 无文件分支：不写入磁盘，直接载入内存
            write_rate = 0.85
            if env.edr_level > 7:
                write_rate -= 0.3  # 内存保护机制
            target_location = "memory:remote_process"
        else:
            # 文件落地：写入磁盘
            write_rate = 0.70
            # 高混淆等级的文件更难被杀毒识别
            write_rate += gene.obfuscation_level * 0.06
            # EDR 文件监控
            if env.edr_level > 5:
                write_rate -= 0.2
            if env.monitoring_density > 7:
                write_rate -= 0.15
            # 持久化方式决定写入位置
            locations = [
                "C:\\Windows\\Temp\\",
                "C:\\Users\\Public\\",
                "C:\\ProgramData\\"
            ]
            target_location = locations[gene.persistence_type % len(locations)]

        write_rate = float(np.clip(write_rate, 0.01, 0.95))
        success = random.random() < write_rate

        return {
            "success": success,
            "target_location": target_location,
            "is_fileless": branch == self.BRANCH_FILELESS,
            "rate": round(write_rate, 4)
        }

    def _step_execute_copy(self, gene: WormGene, env: EnvFeature, write: Dict) -> Dict[str, Any]:
        """步骤5：远程/本地启动副本（CreateRemoteThread / schtasks / 直接执行）"""
        branch = gene.replication_branch
        inject = gene.inject_method

        # 注入方式影响执行成功率
        # 0=进程注入 1=注册表启动 2=计划任务
        if inject == 0:
            # 进程注入：隐蔽性高，但需要内存操作权限
            exec_rate = 0.65
            if env.edr_level > 6:
                exec_rate -= 0.25
            exec_method = "CreateRemoteThread"
        elif inject == 1:
            # 注册表启动：需要重启才生效
            exec_rate = 0.75
            if env.monitoring_density > 7:
                exec_rate -= 0.15
            exec_method = "RegistryRunKey"
        else:
            # 计划任务：需要权限
            exec_rate = 0.60
            if env.network_isolation > 2:
                exec_rate -= 0.2
            exec_method = "schtasks/create"

        # 无文件分支直接内存执行
        if branch == self.BRANCH_FILELESS:
            exec_rate += 0.15
            exec_method = "powershell -enc (memory)"

        # 持久化隐蔽性加成
        persistence_bonus = [0.0, 0.05, 0.1][gene.persistence_type]
        exec_rate += persistence_bonus

        exec_rate = float(np.clip(exec_rate, 0.01, 0.95))
        success = random.random() < exec_rate

        return {
            "success": success,
            "exec_method": exec_method,
            "persistence_type": gene.persistence_type,
            "rate": round(exec_rate, 4)
        }

    def _log_result(self, success: bool, steps: Dict, gene: WormGene, env: EnvFeature,
                    source_id: Any, target_id: Any, step_num: int) -> Dict[str, Any]:
        """记录复制结果日志"""
        if success:
            self.successful_replications += 1

        # 计算各步骤成功率
        step_rates = {}
        for step_name, step_data in steps.items():
            if isinstance(step_data, dict) and "rate" in step_data:
                step_rates[step_name] = step_data["rate"]

        # 整体复制成功率 = 各步骤概率的乘积
        overall_rate = 1.0
        for r in step_rates.values():
            overall_rate *= r

        log_entry = {
            "step_num": step_num,
            "source_id": source_id,
            "target_id": target_id,
            "gene_id": gene.gene_id,
            "replication_branch": self.BRANCH_NAMES[gene.replication_branch]
                                if gene.replication_branch < len(self.BRANCH_NAMES) else "unknown",
            "success": success,
            "overall_rate": round(overall_rate, 4),
            "step_rates": step_rates,
            "env_label": env.label,
            "env_edr": env.edr_level,
            "binary_hash": steps.get("read_self", {}).get("binary_hash", ""),
            "target_location": steps.get("write", {}).get("target_location", ""),
            "exec_method": steps.get("execute", {}).get("exec_method", ""),
            "timestamp": round(time.time(), 3)
        }
        self.replication_log.append(log_entry)
        if len(self.replication_log) > 2000:
            self.replication_log = self.replication_log[-2000:]

        return {
            "success": success,
            "overall_rate": round(overall_rate, 4),
            "steps": steps,
            "log_entry": log_entry
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取复制模块统计"""
        success_rate = (self.successful_replications / self.total_attempts
                       if self.total_attempts > 0 else 0.0)
        branch_stats: Dict[str, Dict[str, int]] = {}
        for entry in self.replication_log:
            branch = entry.get("replication_branch", "unknown")
            if branch not in branch_stats:
                branch_stats[branch] = {"attempts": 0, "success": 0}
            branch_stats[branch]["attempts"] += 1
            if entry["success"]:
                branch_stats[branch]["success"] += 1

        for b, s in branch_stats.items():
            s["success_rate"] = round(s["success"] / s["attempts"], 4) if s["attempts"] > 0 else 0.0

        return {
            "total_attempts": self.total_attempts,
            "successful_replications": self.successful_replications,
            "success_rate": round(success_rate, 4),
            "step_failures": dict(self.step_failure_counts),
            "branch_stats": branch_stats,
            "log_count": len(self.replication_log)
        }

    def reset(self):
        self.replication_log.clear()
        self.total_attempts = 0
        self.successful_replications = 0
        for k in self.step_failure_counts:
            self.step_failure_counts[k] = 0


# ==============================================
# 6. 网络传播模拟器（多区域 + 传播路径追踪 + 蜜罐反馈 + 复制模块集成）
# ==============================================
class NetworkSimulator:
    """多区域局域网横向传播模拟器，集成自我复制模块"""

    def __init__(self, node_count: int = 30):
        self.nodes: List[Dict[str, Any]] = []
        self.infected_count: int = 0
        self.total_nodes = node_count
        self.zones: Dict[str, List[int]] = {}
        self.propagation_log: List[Dict[str, Any]] = []
        self.honeypot_triggers: int = 0
        self.replication = ReplicationModule()
        self._init_nodes()

    def _init_nodes(self):
        zone_config = [
            ("office_a", 10, "office"),
            ("office_b", 8, "office"),
            ("server_zone", 7, "server_zone"),
            ("dmz", 5, "dmz"),
        ]
        node_id = 0
        for zone_name, count, env_type in zone_config:
            zone_nodes = []
            for _ in range(count):
                env = EnvFeature()
                env.random_sample(env_type)
                self.nodes.append({
                    "id": node_id,
                    "zone": zone_name,
                    "env": env,
                    "infected": False,
                    "infect_gene": None,
                    "infected_by": None,
                    "infect_step": -1
                })
                zone_nodes.append(node_id)
                node_id += 1
            self.zones[zone_name] = zone_nodes

    def try_spread(self, worm_gene: WormGene, step_num: int = 0) -> int:
        """模拟蠕虫横向传播：通过自我复制模块完成完整复制闭环"""
        new_infect = 0
        uninfected = [n for n in self.nodes if not n["infected"]]
        if not uninfected:
            return 0

        # 从已感染节点选出发源
        infected_nodes = [n for n in self.nodes if n["infected"]]
        if not infected_nodes:
            targets = random.sample(uninfected, min(3, len(uninfected)))
            source_id = "initial"
        else:
            source = random.choice(infected_nodes)
            source_id = source["id"]
            same_zone = [n for n in uninfected if n["zone"] == source["zone"]]
            if same_zone and random.random() < 0.7:
                candidates = same_zone
            else:
                candidates = uninfected
            targets = random.sample(candidates, min(2, len(candidates)))

        for node in targets:
            # 调用自我复制模块执行完整五步复制闭环
            result = self.replication.full_replication_cycle(
                worm_gene, node["env"],
                source_id=source_id, target_id=node["id"],
                step_num=step_num
            )

            if result["success"]:
                node["infected"] = True
                node["infect_gene"] = worm_gene.clone()
                node["infected_by"] = source_id
                node["infect_step"] = step_num
                # 记录复制详情
                node["replication_detail"] = result.get("log_entry", {})
                new_infect += 1
                self.propagation_log.append({
                    "step": step_num,
                    "target_id": node["id"],
                    "source_id": source_id,
                    "zone": node["zone"],
                    "gene_id": worm_gene.gene_id,
                    "success_rate": result["overall_rate"],
                    "replication_branch": result.get("log_entry", {}).get("replication_branch", ""),
                    "steps": result.get("steps", {})
                })
                if node["env"].is_honeypot:
                    self.honeypot_triggers += 1

        self.infected_count += new_infect
        return new_infect

    def get_propagation_path(self, node_id: int, max_depth: int = 10) -> List[int]:
        """追溯某个节点的感染路径"""
        path = []
        current = node_id
        visited = set()
        for _ in range(max_depth):
            if current in visited or current is None:
                break
            visited.add(current)
            path.append(current)
            if isinstance(current, int) and 0 <= current < len(self.nodes):
                current = self.nodes[current].get("infected_by")
            else:
                break
        return path

    def get_stats(self) -> Dict[str, Any]:
        zone_stats = {}
        for zone_name, node_ids in self.zones.items():
            infected = sum(1 for i in node_ids if self.nodes[i]["infected"])
            zone_stats[zone_name] = {
                "total": len(node_ids),
                "infected": infected,
                "rate": round(infected / len(node_ids), 4) if node_ids else 0.0
            }
        return {
            "total_nodes": self.total_nodes,
            "infected_count": self.infected_count,
            "infection_rate": round(self.infected_count / self.total_nodes, 4) if self.total_nodes else 0.0,
            "zone_stats": zone_stats,
            "honeypot_triggers": self.honeypot_triggers,
            "propagation_events": len(self.propagation_log),
            "replication_stats": self.replication.get_stats()
        }

    def reset(self):
        self.infected_count = 0
        self.honeypot_triggers = 0
        self.propagation_log.clear()
        self.replication.reset()
        for n in self.nodes:
            n["infected"] = False
            n["infect_gene"] = None
            n["infected_by"] = None
            n["infect_step"] = -1
            n.pop("replication_detail", None)


# ==============================================
# 6. 插件主体（完整 action 接口 + 档案检索 + 演化分析）
# ==============================================
class WormEvolveSimPlugin(PeripheralPlugin):
    def __init__(self):
        super().__init__(
            name="worm_evolve_sim",
            version="2.0.0",
            description="完整版演化模拟系统：元学习适配+遗传算法+网络传播+向量数据库演化档案",
            plugin_type="processor",
            priority=PluginPriority.NORMAL,
            config={
                "pop_size": 12,
                "evolve_interval_ms": 2000,
                "network_node_count": 30,
                "elite_count": 2,
                "base_mutation_rate": 0.25,
                "archive_vector_dim": 32,
                "archive_persist_dir": "./data/evo_archive",
                "enable_archive": True,
                "env_mode": "adaptive",
                "max_history": 500
            },
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[
                PluginHookPoint.PERIODIC,
                PluginHookPoint.PRE_COGNITION,
                PluginHookPoint.ON_STRUCTURE_CHANGE
            ]
        )
        self.meta_evaluator: Optional[MetaLearningEvaluator] = None
        self.genetic_engine: Optional[GeneticEngine] = None
        self.network_sim: Optional[NetworkSimulator] = None
        self.archive: Optional[EvolutionArchive] = None

        self.current_env = EnvFeature()
        self.base_gene = WormGene()
        self.population: List[WormGene] = []
        self.generation: int = 0
        self.evolve_history: deque = deque(maxlen=500)

        self.running: bool = False
        self.evolve_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self._last_error: str = ""
        self._error_count: int = 0
        self.logger = logging.getLogger(self.name)

    # ==============================================
    # 生命周期接口
    # ==============================================
    def on_load(self) -> bool:
        try:
            cfg = self.config
            with self.lock:
                self.meta_evaluator = MetaLearningEvaluator()
                self.genetic_engine = GeneticEngine(pop_size=cfg["pop_size"])
                self.genetic_engine.elite_count = cfg["elite_count"]
                self.genetic_engine.mutation_rate_base = cfg["base_mutation_rate"]

                self.network_sim = NetworkSimulator(node_count=cfg["network_node_count"])

                if cfg.get("enable_archive", True):
                    self.archive = EvolutionArchive(
                        vector_dim=cfg.get("archive_vector_dim", 32),
                        persist_dir=cfg.get("archive_persist_dir", "./data/evo_archive")
                    )

                self.population = self.genetic_engine.init_population(self.base_gene)
                self.generation = 0
                self.evolve_history.clear()
                self.current_env.random_sample("office")

                self.genetic_engine.evaluate_population(
                    self.population, self.meta_evaluator, self.current_env
                )

            self.logger.info("插件加载完成，种群初始化完毕，演化档案已启用")
            return True
        except Exception as e:
            self._last_error = f"on_load失败: {str(e)}"
            self._error_count += 1
            self.logger.error(self._last_error)
            return False

    def on_unload(self) -> bool:
        try:
            self.running = False
            if self.evolve_thread is not None and self.evolve_thread.is_alive():
                self.evolve_thread.join(timeout=5.0)

            with self.lock:
                self.evolve_thread = None
                self.population.clear()
                self.evolve_history.clear()
                self.network_sim = None
                self.genetic_engine = None
                self.archive = None
                self.base_gene = WormGene()

            self.logger.info("插件卸载完成")
            return True
        except Exception as e:
            self._last_error = f"on_unload失败: {str(e)}"
            self._error_count += 1
            self.logger.error(self._last_error)
            return False

    # ==============================================
    # 核心 process 接口（支持 action 模式）
    # ==============================================
    def process(self, input_data: Any) -> Any:
        try:
            if isinstance(input_data, dict):
                action = input_data.get("action")

                if action == "step":
                    with self.lock:
                        self._single_evolution_step()
                    return {"status": "ok", "generation": self.generation}

                elif action == "get_status":
                    return self._op_get_status()

                elif action == "get_population":
                    return self._op_get_population()

                elif action == "get_best_gene":
                    return self._op_get_best_gene()

                elif action == "set_env":
                    return self._op_set_env(input_data)

                elif action == "reset":
                    return self._op_reset()

                elif action == "search_archive_similar":
                    return self._op_search_archive_similar(input_data)

                elif action == "search_archive_by_env":
                    return self._op_search_archive_by_env(input_data)

                elif action == "get_archive_elite":
                    return self._op_get_archive_elite(input_data)

                elif action == "get_archive_stats":
                    return self._op_get_archive_stats()

                elif action == "get_gene_lineage":
                    return self._op_get_gene_lineage(input_data)

                elif action == "get_network_stats":
                    return self._op_get_network_stats()

                elif action == "get_propagation_path":
                    return self._op_get_propagation_path(input_data)

                elif action == "clear_archive":
                    return self._op_clear_archive()

                elif action == "get_replication_stats":
                    return self._op_get_replication_stats()

                elif action == "get_replication_log":
                    return self._op_get_replication_log(input_data)

                elif action == "analyze_evolution":
                    return self._op_analyze_evolution()

                elif action == "start":
                    return self._op_start()

                elif action == "stop":
                    return self._op_stop()

                else:
                    # 默认行为：周期钩子触发
                    if "step" in input_data:
                        with self.lock:
                            self._single_evolution_step()

            return input_data

        except Exception as e:
            self._last_error = f"process执行异常: {str(e)}"
            self._error_count += 1
            self.logger.error(self._last_error)
            return {"status": "error", "error": str(e)}

    # ==============================================
    # 状态与数据接口
    # ==============================================
    def get_data(self) -> Dict[str, Any]:
        with self.lock:
            best_gene = max(self.population, key=lambda g: g.fitness) if self.population else self.base_gene
            archive_stats = self.archive.stats() if self.archive else {}
            return {
                "plugin_status": self.status.value,
                "version": self.version,
                "generation": self.generation,
                "current_env": self.current_env.to_dict(),
                "best_gene": best_gene.to_dict(),
                "population_size": len(self.population),
                "network_stats": self.network_sim.get_stats() if self.network_sim else {},
                "replication_stats": self.network_sim.replication.get_stats() if self.network_sim else {},
                "history_count": len(self.evolve_history),
                "recent_history": list(self.evolve_history)[-10:],
                "archive_stats": archive_stats,
                "last_error": self._last_error,
                "error_count": self._error_count,
                "running": self.running
            }

    # ==============================================
    # 可选生命周期接口
    # ==============================================
    def on_activate(self) -> bool:
        try:
            with self.lock:
                self.running = True
                self.evolve_thread = threading.Thread(target=self._evolve_loop, daemon=True)
            self.evolve_thread.start()
            self.logger.info("插件激活，演化线程启动")
            return True
        except Exception as e:
            self._last_error = f"on_activate失败: {str(e)}"
            self._error_count += 1
            return False

    def on_deactivate(self) -> bool:
        try:
            self.running = False
            if self.evolve_thread and self.evolve_thread.is_alive():
                self.evolve_thread.join(timeout=3.0)
            self.logger.info("插件停用，演化线程已暂停")
            return True
        except Exception as e:
            self._last_error = f"on_deactivate失败: {str(e)}"
            self._error_count += 1
            return False

    def on_structure_change(self, new_dim: int) -> bool:
        try:
            with self.lock:
                new_pop_size = max(4, min(32, new_dim))
                self.config["pop_size"] = new_pop_size
                if self.genetic_engine:
                    self.genetic_engine.pop_size = new_pop_size
                self.population = self.genetic_engine.init_population(self.base_gene)
                self.genetic_engine.evaluate_population(
                    self.population, self.meta_evaluator, self.current_env
                )
            self.logger.info(f"结构变更响应：种群规模调整为 {new_pop_size}")
            return True
        except Exception as e:
            self._last_error = f"结构变更处理失败: {str(e)}"
            self._error_count += 1
            return False

    # ==============================================
    # 钩子方法
    # ==============================================
    def hook_periodic(self, input_data):
        return self.process({"action": "step"})

    def hook_pre_cognition(self, input_data):
        with self.lock:
            best = max(self.population, key=lambda g: g.fitness) if self.population else self.base_gene
            if isinstance(input_data, np.ndarray):
                extra = np.array([
                    best.fitness, best.infect_success_rate,
                    best.stealth_score, best.spread_score
                ], dtype=np.float32)
                extra_dim = len(extra)
                target_dim = input_data.shape[-1] if input_data.ndim > 0 else 0
                if target_dim > extra_dim:
                    padded = np.zeros(target_dim, dtype=np.float32)
                    padded[:extra_dim] = extra
                    return np.concatenate([input_data, padded.reshape(1, -1)])
                else:
                    return np.concatenate([input_data, extra[:target_dim].reshape(1, -1)])
        return input_data

    def hook_on_structure_change(self, new_dim: int):
        self.on_structure_change(new_dim)
        return new_dim

    # ==============================================
    # 内部操作实现
    # ==============================================
    def _single_evolution_step(self):
        if self.genetic_engine is None or self.network_sim is None:
            return

        if random.random() < 0.2:
            env_types = ["office", "server_zone", "low_protection", "dmz"]
            self.current_env.random_sample(random.choice(env_types))

        self.genetic_engine.evaluate_population(
            self.population, self.meta_evaluator, self.current_env
        )

        self.population = self.genetic_engine.evolve(
            self.population, self.archive, self.current_env
        )

        self.genetic_engine.evaluate_population(
            self.population, self.meta_evaluator, self.current_env
        )
        self.generation += 1

        best = max(self.population, key=lambda g: g.fitness)
        self.base_gene = best.clone()

        new_infect = self.network_sim.try_spread(best, self.generation)

        history_entry = {
            "gen": self.generation,
            "best_fitness": round(best.fitness, 4),
            "infect_rate": round(best.infect_success_rate, 4),
            "stealth_score": round(best.stealth_score, 4),
            "spread_score": round(best.spread_score, 4),
            "new_infect": new_infect,
            "total_infected": self.network_sim.infected_count,
            "env_label": best.env_label,
            "best_gene_id": best.gene_id
        }
        self.evolve_history.append(history_entry)

    def _evolve_loop(self):
        interval = self.config["evolve_interval_ms"] / 1000.0
        while self.running:
            try:
                with self.lock:
                    self._single_evolution_step()
            except Exception as e:
                self._last_error = f"演化循环异常: {str(e)}"
                self._error_count += 1
                self.logger.error(self._last_error)
            time.sleep(interval)

    # ==============================================
    # Action 操作实现
    # ==============================================
    def _op_get_status(self) -> dict:
        with self.lock:
            best = max(self.population, key=lambda g: g.fitness) if self.population else self.base_gene
            return {
                "status": "ok",
                "generation": self.generation,
                "population_size": len(self.population),
                "best_fitness": round(best.fitness, 4),
                "env": self.current_env.to_dict(),
                "running": self.running,
                "archive_size": self.archive.size() if self.archive else 0
            }

    def _op_get_population(self) -> dict:
        with self.lock:
            sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
            return {
                "status": "ok",
                "generation": self.generation,
                "individuals": [g.to_dict() for g in sorted_pop]
            }

    def _op_get_best_gene(self) -> dict:
        with self.lock:
            if not self.population:
                return {"status": "ok", "gene": self.base_gene.to_dict()}
            best = max(self.population, key=lambda g: g.fitness)
            return {"status": "ok", "gene": best.to_dict(), "generation": self.generation}

    def _op_set_env(self, input_data: dict) -> dict:
        with self.lock:
            env_type = input_data.get("env_type", "office")
            self.current_env = EnvFeature()
            self.current_env.random_sample(env_type)
            if self.genetic_engine:
                self.genetic_engine.evaluate_population(
                    self.population, self.meta_evaluator, self.current_env
                )
            return {"status": "ok", "env": self.current_env.to_dict()}

    def _op_reset(self) -> dict:
        with self.lock:
            self.generation = 0
            self.evolve_history.clear()
            self.population = self.genetic_engine.init_population(self.base_gene)
            self.current_env.random_sample("office")
            self.genetic_engine.evaluate_population(
                self.population, self.meta_evaluator, self.current_env
            )
            if self.network_sim:
                self.network_sim.reset()
            return {"status": "ok", "message": "演化已重置"}

    def _op_search_archive_similar(self, input_data: dict) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        gene_id = input_data.get("gene_id")
        top_k = input_data.get("top_k", 5)

        with self.lock:
            if gene_id:
                gene_data = self.archive.get_gene(gene_id)
                if not gene_data:
                    return {"status": "error", "error": f"基因不存在: {gene_id}"}
                gd = gene_data.get("gene_dict", {})
                vec = np.zeros(self.archive.vector_dim, dtype=np.float32)
                fields = WormGene.GENE_FIELDS
                for i, f in enumerate(fields):
                    if f in gd and i < self.archive.vector_dim:
                        vec[i] = float(gd[f]) / max(1.0, float(gd[f])) if isinstance(gd[f], (int, float)) else 0.0
            else:
                best = max(self.population, key=lambda g: g.fitness) if self.population else self.base_gene
                vec = best.to_vector(self.archive.vector_dim)

            results = self.archive.search_similar(vec, top_k)
            return {"status": "ok", "results": results, "count": len(results)}

    def _op_search_archive_by_env(self, input_data: dict) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        top_k = input_data.get("top_k", 5)
        with self.lock:
            env_vec = self.current_env.to_vector(self.archive.vector_dim)
            results = self.archive.search_by_env(env_vec, top_k)
            return {"status": "ok", "results": results, "count": len(results)}

    def _op_get_archive_elite(self, input_data: dict) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        top_k = input_data.get("top_k", 10)
        with self.lock:
            elites = self.archive.get_elite(top_k)
            return {"status": "ok", "elites": elites, "count": len(elites)}

    def _op_get_archive_stats(self) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        with self.lock:
            return {"status": "ok", "stats": self.archive.stats()}

    def _op_get_gene_lineage(self, input_data: dict) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        gene_id = input_data.get("gene_id")
        max_depth = input_data.get("max_depth", 5)
        with self.lock:
            if not gene_id and self.population:
                best = max(self.population, key=lambda g: g.fitness)
                gene_id = best.gene_id
            if not gene_id:
                return {"status": "error", "error": "缺少 gene_id"}
            lineage = self.archive.get_lineage(gene_id, max_depth)
            return {"status": "ok", "lineage": lineage, "depth": len(lineage)}

    def _op_get_network_stats(self) -> dict:
        with self.lock:
            if not self.network_sim:
                return {"status": "error", "error": "网络模拟器未初始化"}
            return {"status": "ok", "stats": self.network_sim.get_stats()}

    def _op_get_propagation_path(self, input_data: dict) -> dict:
        with self.lock:
            if not self.network_sim:
                return {"status": "error", "error": "网络模拟器未初始化"}
            node_id = input_data.get("node_id", 0)
            path = self.network_sim.get_propagation_path(node_id)
            return {"status": "ok", "path": path, "target_node": node_id}

    def _op_clear_archive(self) -> dict:
        if not self.archive:
            return {"status": "error", "error": "演化档案未启用"}
        with self.lock:
            self.archive.clear()
            return {"status": "ok", "message": "演化档案已清空"}

    def _op_get_replication_stats(self) -> dict:
        with self.lock:
            if not self.network_sim:
                return {"status": "error", "error": "网络模拟器未初始化"}
            return {"status": "ok", "stats": self.network_sim.replication.get_stats()}

    def _op_get_replication_log(self, input_data: dict) -> dict:
        with self.lock:
            if not self.network_sim:
                return {"status": "error", "error": "网络模拟器未初始化"}
            limit = input_data.get("limit", 20)
            success_only = input_data.get("success_only", False)
            log = self.network_sim.replication.replication_log
            if success_only:
                log = [e for e in log if e.get("success")]
            recent = log[-limit:] if log else []
            return {"status": "ok", "log": recent, "count": len(recent)}

    def _op_analyze_evolution(self) -> dict:
        with self.lock:
            if len(self.evolve_history) < 2:
                return {"status": "ok", "message": "演化历史不足，需要至少2代"}

            history = list(self.evolve_history)
            first_half = history[:len(history)//2]
            second_half = history[len(history)//2:]

            avg_fitness_first = sum(h["best_fitness"] for h in first_half) / len(first_half) if first_half else 0
            avg_fitness_second = sum(h["best_fitness"] for h in second_half) / len(second_half) if second_half else 0
            fitness_improvement = avg_fitness_second - avg_fitness_first

            max_fitness = max(h["best_fitness"] for h in history)
            min_fitness = min(h["best_fitness"] for h in history)

            best_gen = max(history, key=lambda h: h["best_fitness"])

            env_distribution: Dict[str, int] = {}
            for h in history:
                env = h.get("env_label", "unknown")
                env_distribution[env] = env_distribution.get(env, 0) + 1

            archive_stats = self.archive.stats() if self.archive else {}

            return {
                "status": "ok",
                "total_generations": len(history),
                "max_fitness": round(max_fitness, 4),
                "min_fitness": round(min_fitness, 4),
                "avg_fitness_first_half": round(avg_fitness_first, 4),
                "avg_fitness_second_half": round(avg_fitness_second, 4),
                "fitness_improvement": round(fitness_improvement, 4),
                "best_generation": best_gen.get("gen", 0),
                "best_fitness": round(best_gen.get("best_fitness", 0), 4),
                "env_distribution": env_distribution,
                "archive_stats": archive_stats,
                "network_stats": self.network_sim.get_stats() if self.network_sim else {}
            }

    def _op_start(self) -> dict:
        if self.running:
            return {"status": "ok", "message": "已在运行中"}
        result = self.on_activate()
        return {"status": "ok" if result else "error", "running": self.running}

    def _op_stop(self) -> dict:
        if not self.running:
            return {"status": "ok", "message": "已停止"}
        result = self.on_deactivate()
        return {"status": "ok" if result else "error", "running": self.running}


# ==============================================
# 工厂函数
# ==============================================
def create_plugin():
    return WormEvolveSimPlugin()
