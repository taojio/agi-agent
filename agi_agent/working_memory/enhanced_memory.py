import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import hashlib


class MemoryImportance(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


class MemoryStatus(Enum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    FORGOTTEN = "forgotten"
    CONFLICTED = "conflicted"


class MemoryUnit:
    def __init__(self, memory_id: str, content: str, 
                 timestamp: Optional[float] = None,
                 source: str = "system",
                 importance: MemoryImportance = MemoryImportance.MEDIUM):
        self.memory_id = memory_id
        self.content = content
        self.timestamp = timestamp or datetime.now().timestamp()
        self.source = source
        self.importance = importance
        self.access_count = 0
        self.last_access_time = self.timestamp
        self.status = MemoryStatus.ACTIVE
        self.embedding: Optional[np.ndarray] = None
        self.tags: List[str] = []
        self.cluster_id: Optional[int] = None
        self.conflicts: List[str] = []
        self.resolution: Optional[str] = None

    def access(self):
        self.access_count += 1
        self.last_access_time = datetime.now().timestamp()

    def update_content(self, new_content: str):
        self.content = new_content
        self.timestamp = datetime.now().timestamp()
        self.embedding = None

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)

    def mark_conflicted(self, conflict_id: str):
        if conflict_id not in self.conflicts:
            self.conflicts.append(conflict_id)
        self.status = MemoryStatus.CONFLICTED

    def resolve_conflict(self, resolution: str):
        self.resolution = resolution
        self.status = MemoryStatus.ACTIVE


class MemoryAssociationDiscovery:
    def __init__(self, similarity_threshold: float = 0.5, 
                 min_cluster_size: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.clusters: Dict[int, List[MemoryUnit]] = {}
        self.association_graph: Dict[str, List[Tuple[str, float]]] = {}

    def discover_associations(self, memories: List[MemoryUnit]) -> Dict[str, Any]:
        if len(memories) < 2:
            return {"clusters": [], "associations": [], "quality": {}}
        
        contents = [m.content for m in memories]
        memory_ids = [m.memory_id for m in memories]
        
        tfidf_matrix = self.vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        self._build_association_graph(memory_ids, similarity_matrix)
        
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric="precomputed"
        )
        
        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)
        
        self.clusters = {}
        for idx, label in enumerate(labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(memories[idx])
            memories[idx].cluster_id = label
        
        quality = self._assess_clustering_quality(similarity_matrix, labels)
        
        return {
            "clusters": [
                {
                    "cluster_id": cid,
                    "size": len(mems),
                    "memory_ids": [m.memory_id for m in mems],
                    "representative": self._get_cluster_representative(mems)
                }
                for cid, mems in self.clusters.items() if cid >= 0
            ],
            "associations": self._get_association_list(),
            "quality": quality
        }

    def _build_association_graph(self, memory_ids: List[str], 
                                 similarity_matrix: np.ndarray):
        self.association_graph = {mid: [] for mid in memory_ids}
        
        n = len(memory_ids)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    self.association_graph[memory_ids[i]].append((memory_ids[j], similarity))
                    self.association_graph[memory_ids[j]].append((memory_ids[i], similarity))

    def _get_cluster_representative(self, memories: List[MemoryUnit]) -> str:
        if not memories:
            return ""
        
        contents = [m.content for m in memories]
        tfidf_matrix = self.vectorizer.transform(contents)
        
        centroid = tfidf_matrix.mean(axis=0).A[0]
        
        max_similarity = -1
        representative = ""
        for mem in memories:
            mem_tfidf = self.vectorizer.transform([mem.content]).A[0]
            similarity = np.dot(mem_tfidf, centroid) / (
                np.linalg.norm(mem_tfidf) * np.linalg.norm(centroid) + 1e-8
            )
            if similarity > max_similarity:
                max_similarity = similarity
                representative = mem.content[:100]
        
        return representative

    def _assess_clustering_quality(self, similarity_matrix: np.ndarray, 
                                   labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "num_clusters": 1}
        
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "num_clusters": len(np.unique(labels))}
        
        distance_matrix = 1 - similarity_matrix
        
        try:
            silhouette = silhouette_score(distance_matrix[valid_mask][:, valid_mask], 
                                        labels[valid_mask], metric="precomputed")
        except:
            silhouette = 0.0
        
        try:
            ch_score = calinski_harabasz_score(similarity_matrix[valid_mask], labels[valid_mask])
        except:
            ch_score = 0.0
        
        return {
            "silhouette": float(silhouette),
            "calinski_harabasz": float(ch_score),
            "num_clusters": len(np.unique(labels))
        }

    def _get_association_list(self) -> List[Dict[str, Any]]:
        associations = []
        seen_pairs = set()
        
        for source_id, targets in self.association_graph.items():
            for target_id, similarity in targets:
                pair = tuple(sorted([source_id, target_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    associations.append({
                        "source": source_id,
                        "target": target_id,
                        "similarity": similarity
                    })
        
        associations.sort(key=lambda x: -x["similarity"])
        return associations

    def update_clusters(self, new_memories: List[MemoryUnit]) -> Dict[str, Any]:
        all_memories = []
        for cluster in self.clusters.values():
            all_memories.extend(cluster)
        all_memories.extend(new_memories)
        
        return self.discover_associations(all_memories)


class MemoryCompressor:
    def __init__(self, compression_ratio: float = 0.5,
                 max_summary_length: int = 200):
        self.compression_ratio = compression_ratio
        self.max_summary_length = max_summary_length
        self.compressed_memories: Dict[str, MemoryUnit] = {}

    def compress_memory(self, memory: MemoryUnit, 
                       preserve_key_info: bool = True) -> MemoryUnit:
        if len(memory.content) <= self.max_summary_length:
            return memory
        
        summary = self._generate_summary(memory.content, preserve_key_info)
        
        compressed = MemoryUnit(
            memory_id=f"compressed_{memory.memory_id}",
            content=summary,
            timestamp=memory.timestamp,
            source=f"compressed_{memory.source}",
            importance=memory.importance
        )
        compressed.status = MemoryStatus.COMPRESSED
        compressed.tags = memory.tags.copy()
        compressed.cluster_id = memory.cluster_id
        
        self.compressed_memories[memory.memory_id] = compressed
        
        memory.status = MemoryStatus.COMPRESSED
        
        return compressed

    def _generate_summary(self, content: str, preserve_key_info: bool) -> str:
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return content[:self.max_summary_length]
        
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = tfidf.fit_transform(sentences)
        except:
            return content[:self.max_summary_length]
        
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        num_sentences = max(1, int(len(sentences) * self.compression_ratio))
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices.sort()
        
        summary_sentences = [sentences[i] for i in top_indices]
        summary = '. '.join(summary_sentences)
        
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length]
        
        return summary

    def compress_cluster(self, cluster: List[MemoryUnit]) -> MemoryUnit:
        combined_content = " ".join([m.content for m in cluster])
        
        summary = self._generate_summary(combined_content, preserve_key_info=True)
        
        cluster_id = cluster[0].cluster_id if cluster else None
        
        compressed = MemoryUnit(
            memory_id=f"cluster_compressed_{cluster_id}",
            content=summary,
            timestamp=datetime.now().timestamp(),
            source="cluster_compression",
            importance=max(m.importance for m in cluster) if cluster else MemoryImportance.MEDIUM
        )
        compressed.status = MemoryStatus.COMPRESSED
        compressed.cluster_id = cluster_id
        
        for mem in cluster:
            mem.status = MemoryStatus.COMPRESSED
            self.compressed_memories[mem.memory_id] = compressed
        
        return compressed

    def decompress_memory(self, memory_id: str) -> Optional[MemoryUnit]:
        return self.compressed_memories.get(memory_id)

    def get_compression_stats(self) -> Dict[str, Any]:
        total_compressed = len(self.compressed_memories)
        return {
            "total_compressed": total_compressed,
            "compression_ratio": self.compression_ratio,
            "max_summary_length": self.max_summary_length
        }


class MemoryConflictResolver:
    def __init__(self):
        self.conflicts: Dict[str, Dict[str, Any]] = {}
        self.resolution_history: List[Dict[str, Any]] = []

    def detect_conflicts(self, memories: List[MemoryUnit]) -> List[Dict[str, Any]]:
        detected_conflicts = []
        
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mem1 = memories[i]
                mem2 = memories[j]
                
                conflict_score = self._calculate_conflict_score(mem1, mem2)
                
                if conflict_score > 0.5:
                    conflict_id = f"conflict_{mem1.memory_id}_{mem2.memory_id}"
                    
                    self.conflicts[conflict_id] = {
                        "conflict_id": conflict_id,
                        "memory_ids": [mem1.memory_id, mem2.memory_id],
                        "conflict_score": conflict_score,
                        "status": "detected",
                        "resolution": None,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    mem1.mark_conflicted(conflict_id)
                    mem2.mark_conflicted(conflict_id)
                    
                    detected_conflicts.append(self.conflicts[conflict_id])
        
        return detected_conflicts

    def _calculate_conflict_score(self, mem1: MemoryUnit, mem2: MemoryUnit) -> float:
        similarity = self._text_similarity(mem1.content, mem2.content)
        
        if similarity < 0.3:
            return 0.0
        
        contradiction_score = self._detect_contradiction(mem1.content, mem2.content)
        
        return similarity * contradiction_score

    def _text_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        tfidf = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = tfidf.fit_transform([text1, text2])
            return float(cosine_similarity(tfidf_matrix)[0, 1])
        except:
            return 0.0

    def _detect_contradiction(self, text1: str, text2: str) -> float:
        contradiction_keywords = [
            ("yes", "no"), ("true", "false"), ("correct", "wrong"),
            ("exists", "does not exist"), ("is", "is not"),
            ("has", "does not have"), ("can", "cannot"), ("will", "will not")
        ]
        
        score = 0.0
        count = 0
        
        for pos, neg in contradiction_keywords:
            has_pos1 = pos.lower() in text1.lower()
            has_neg1 = neg.lower() in text1.lower()
            has_pos2 = pos.lower() in text2.lower()
            has_neg2 = neg.lower() in text2.lower()
            
            if (has_pos1 and has_neg2) or (has_neg1 and has_pos2):
                score += 1.0
                count += 1
        
        if count > 0:
            return score / count
        
        return 0.0

    def resolve_conflict(self, conflict_id: str, strategy: str = "automatic") -> Dict[str, Any]:
        if conflict_id not in self.conflicts:
            return {"error": "Conflict not found"}
        
        conflict = self.conflicts[conflict_id]
        
        resolution = self._apply_resolution_strategy(conflict, strategy)
        
        conflict["status"] = "resolved"
        conflict["resolution"] = resolution
        conflict["resolved_at"] = datetime.now().timestamp()
        
        for mem_id in conflict["memory_ids"]:
            self._resolve_memory(mem_id, resolution)
        
        self.resolution_history.append(conflict)
        
        return conflict

    def _apply_resolution_strategy(self, conflict: Dict[str, Any], 
                                   strategy: str) -> str:
        memory_ids = conflict["memory_ids"]
        
        if strategy == "latest":
            return f"保留最新记忆: {memory_ids[0]}"
        elif strategy == "source":
            return f"基于来源可靠性选择记忆: {memory_ids[0]}"
        elif strategy == "importance":
            return f"基于重要性选择记忆: {memory_ids[0]}"
        else:
            return f"自动解决冲突，保留较高可信度记忆: {memory_ids[0]}"

    def _resolve_memory(self, memory_id: str, resolution: str):
        pass

    def get_conflict_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.resolution_history)[-limit:]


class ActiveForgettingManager:
    def __init__(self, retention_policy: str = "importance",
                 max_memories: int = 1000, decay_rate: float = 0.95):
        self.retention_policy = retention_policy
        self.max_memories = max_memories
        self.decay_rate = decay_rate
        self.forgotten_memories: List[MemoryUnit] = []

    def calculate_priority(self, memory: MemoryUnit) -> float:
        now = datetime.now().timestamp()
        age_days = (now - memory.timestamp) / (24 * 3600)
        
        importance_weight = memory.importance.value / 5.0
        
        recency_weight = 1.0 / (1 + age_days)
        
        frequency_weight = min(memory.access_count / 10, 1.0)
        
        if self.retention_policy == "importance":
            return importance_weight * 0.5 + recency_weight * 0.3 + frequency_weight * 0.2
        elif self.retention_policy == "recency":
            return importance_weight * 0.2 + recency_weight * 0.5 + frequency_weight * 0.3
        elif self.retention_policy == "frequency":
            return importance_weight * 0.2 + recency_weight * 0.3 + frequency_weight * 0.5
        else:
            return importance_weight * 0.33 + recency_weight * 0.33 + frequency_weight * 0.34

    def forget_low_priority(self, memories: List[MemoryUnit], 
                           target_count: Optional[int] = None) -> List[MemoryUnit]:
        if target_count is None:
            target_count = self.max_memories
        
        if len(memories) <= target_count:
            return []
        
        priorities = [(self.calculate_priority(m), m) for m in memories]
        priorities.sort(key=lambda x: x[0])
        
        to_forget = [m for _, m in priorities[:len(memories) - target_count]]
        
        for memory in to_forget:
            memory.status = MemoryStatus.FORGOTTEN
            self.forgotten_memories.append(memory)
        
        return to_forget

    def apply_decay(self, memories: List[MemoryUnit]):
        now = datetime.now().timestamp()
        
        for memory in memories:
            if memory.status == MemoryStatus.ACTIVE:
                age_days = (now - memory.last_access_time) / (24 * 3600)
                decay_factor = self.decay_rate ** age_days
                
                new_importance_value = max(1, memory.importance.value - int(decay_factor < 0.5))
                memory.importance = MemoryImportance(new_importance_value)

    def restore_memory(self, memory_id: str) -> Optional[MemoryUnit]:
        for memory in self.forgotten_memories:
            if memory.memory_id == memory_id:
                memory.status = MemoryStatus.ACTIVE
                return memory
        return None

    def get_forgetting_stats(self) -> Dict[str, Any]:
        return {
            "total_forgotten": len(self.forgotten_memories),
            "retention_policy": self.retention_policy,
            "max_memories": self.max_memories,
            "decay_rate": self.decay_rate
        }


class EnhancedMemorySystem:
    def __init__(self):
        self.memories: Dict[str, MemoryUnit] = {}
        self.association_discovery = MemoryAssociationDiscovery()
        self.compressor = MemoryCompressor()
        self.conflict_resolver = MemoryConflictResolver()
        self.forgetting_manager = ActiveForgettingManager()
        self.history = deque(maxlen=1000)

    def add_memory(self, content: str, source: str = "system",
                   importance: MemoryImportance = MemoryImportance.MEDIUM) -> MemoryUnit:
        memory_id = hashlib.md5(f"{content}{datetime.now().timestamp()}".encode()).hexdigest()[:16]
        
        memory = MemoryUnit(
            memory_id=memory_id,
            content=content,
            source=source,
            importance=importance
        )
        
        self.memories[memory_id] = memory
        
        self.history.append({
            "action": "add",
            "memory_id": memory_id,
            "timestamp": memory.timestamp
        })
        
        return memory

    def get_memory(self, memory_id: str) -> Optional[MemoryUnit]:
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory

    def search_memories(self, query: str, top_k: int = 10) -> List[MemoryUnit]:
        if not self.memories:
            return []
        
        contents = [m.content for m in self.memories.values()]
        memory_list = list(self.memories.values())
        
        tfidf = TfidfVectorizer(max_features=512)
        try:
            tfidf_matrix = tfidf.fit_transform(contents + [query])
            similarities = cosine_similarity(tfidf_matrix)[-1, :-1]
        except:
            return []
        
        results = sorted(zip(memory_list, similarities), key=lambda x: -x[1])
        return [m for m, _ in results[:top_k]]

    def discover_associations(self) -> Dict[str, Any]:
        active_memories = [m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]
        return self.association_discovery.discover_associations(active_memories)

    def compress_memory(self, memory_id: str) -> Optional[MemoryUnit]:
        memory = self.memories.get(memory_id)
        if not memory:
            return None
        
        return self.compressor.compress_memory(memory)

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        active_memories = [m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]
        return self.conflict_resolver.detect_conflicts(active_memories)

    def resolve_conflict(self, conflict_id: str, strategy: str = "automatic") -> Dict[str, Any]:
        return self.conflict_resolver.resolve_conflict(conflict_id, strategy)

    def apply_forgetting(self) -> List[MemoryUnit]:
        active_memories = [m for m in self.memories.values() if m.status == MemoryStatus.ACTIVE]
        return self.forgetting_manager.forget_low_priority(active_memories)

    def get_statistics(self) -> Dict[str, Any]:
        active = sum(1 for m in self.memories.values() if m.status == MemoryStatus.ACTIVE)
        compressed = sum(1 for m in self.memories.values() if m.status == MemoryStatus.COMPRESSED)
        conflicted = sum(1 for m in self.memories.values() if m.status == MemoryStatus.CONFLICTED)
        forgotten = len(self.forgetting_manager.forgotten_memories)
        
        return {
            "total_memories": len(self.memories),
            "active": active,
            "compressed": compressed,
            "conflicted": conflicted,
            "forgotten": forgotten,
            "association_clusters": len(self.association_discovery.clusters),
            "detected_conflicts": len(self.conflict_resolver.conflicts)
        }
