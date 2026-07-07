from agi_agent.plugins.plugin_base import (
    PeripheralPlugin,
    PluginPriority,
    PluginHookPoint,
    PluginStatus
)
from typing import Dict, Any, List, Optional, Callable, Set, Tuple, Union
from abc import ABC, abstractmethod
import threading
import logging
import time
import os
import json
import hashlib
import gc
import uuid
from collections import deque
import numpy as np


# ==================================================
# 嵌入函数协议：支持外部接入与本地回退
# ==================================================
class EmbeddingFunction(ABC):
    """嵌入函数抽象基类，统一文本到向量的编码接口"""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """编码单条文本为向量"""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本为向量"""
        pass


class HashingEmbedding(EmbeddingFunction):
    """
    基于哈希的回退嵌入实现，无任何外部依赖
    通过MD5哈希将文本映射为定长浮点向量，适合无模型环境下的占位使用
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        text_bytes = text.encode("utf-8")
        # 用 SHA-512 + 多次种子迭代生成伪随机序列，避免每维一次哈希
        seed = int.from_bytes(hashlib.sha256(text_bytes).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        # L2归一化，便于余弦相似度
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode(t) for t in texts])


# ==================================================
# 存储抽象层：统一本地/远端向量存储接口
# ==================================================
class VectorStoreBase(ABC):
    """向量存储抽象基类，所有存储引擎统一接口"""

    def __init__(self, collection_name: str, dim: int, config: dict):
        self.name = collection_name
        self.dim = dim
        self.config = config
        self._lock = threading.Lock()

    @abstractmethod
    def create(self) -> bool:
        """创建集合"""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: List[str], metadata: List[dict]) -> None:
        """批量添加向量"""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int, filter_func: Optional[Callable] = None):
        """相似度检索，返回 (ids, scores, metadata)"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """按ID删除，返回删除数量"""
        pass

    @abstractmethod
    def drop(self) -> None:
        """销毁集合，释放所有资源"""
        pass

    @abstractmethod
    def count(self) -> int:
        """返回向量总数"""
        pass

    @property
    @abstractmethod
    def index_type(self) -> str:
        """当前索引类型"""
        pass


# ==================================================
# 本地FAISS存储引擎：支持多索引、量化、自动升级
# ==================================================
class LocalFAISSStore(VectorStoreBase):
    """
    本地嵌入式FAISS存储
    支持索引类型: Flat, IVF, HNSW
    支持量化: None, SQ8, PQ
    """

    INDEX_THRESHOLDS = {
        "Flat": 10000,
        "IVF": 1000000,
        "HNSW": 10000000
    }

    def __init__(self, collection_name: str, dim: int, config: dict):
        super().__init__(collection_name, dim, config)
        self._index = None
        self.ids: List[str] = []
        self.metadata: List[dict] = []
        self._index_type = config.get("index_type", "Flat")
        self._quantization = config.get("quantization", "None")  # None / SQ8 / PQ
        self._pq_m = config.get("pq_m", 8)  # PQ子空间数量
        self._nlist = config.get("ivf_nlist", 1024)  # IVF聚类中心数
        self._hnsw_m = config.get("hnsw_m", 32)  # HNSW邻居数
        self._metric = config.get("metric", "cosine")
        self._trained = False
        self._persist_dir = config.get("persist_dir", "./data/vectors")

    @property
    def index_type(self) -> str:
        return f"{self._index_type}_{self._quantization}".replace("_None", "")

    def _build_index(self):
        """构建对应类型的FAISS索引"""
        import faiss

        metric_type = faiss.METRIC_INNER_PRODUCT if self._metric == "cosine" else faiss.METRIC_L2

        if self._index_type == "Flat":
            if self._quantization == "None":
                index = faiss.IndexFlat(self.dim, metric_type)
            elif self._quantization == "SQ8":
                index = faiss.IndexScalarQuantizer(self.dim, faiss.ScalarQuantizer.QT_8bit, metric_type)
            elif self._quantization == "PQ":
                index = faiss.IndexPQ(self.dim, self._pq_m, 8, metric_type)
            else:
                index = faiss.IndexFlat(self.dim, metric_type)

        elif self._index_type == "IVF":
            quantizer = faiss.IndexFlat(self.dim, metric_type)
            if self._quantization == "None":
                index = faiss.IndexIVFFlat(quantizer, self.dim, self._nlist, metric_type)
            elif self._quantization == "SQ8":
                index = faiss.IndexIVFScalarQuantizer(quantizer, self.dim, self._nlist, faiss.ScalarQuantizer.QT_8bit, metric_type)
            elif self._quantization == "PQ":
                index = faiss.IndexIVFPQ(quantizer, self.dim, self._nlist, self._pq_m, 8, metric_type)
            else:
                index = faiss.IndexIVFFlat(quantizer, self.dim, self._nlist, metric_type)
            index.nprobe = min(20, self._nlist // 4)

        elif self._index_type == "HNSW":
            if self._quantization == "None":
                index = faiss.IndexHNSWFlat(self.dim, self._hnsw_m, metric_type)
            elif self._quantization == "SQ8":
                index = faiss.IndexHNSWSQ(self.dim, faiss.ScalarQuantizer.QT_8bit, self._hnsw_m, metric_type)
            elif self._quantization == "PQ":
                index = faiss.IndexHNSWPQ(self.dim, self._pq_m, self._hnsw_m, metric_type)
            else:
                index = faiss.IndexHNSWFlat(self.dim, self._hnsw_m, metric_type)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 200

        else:
            index = faiss.IndexFlat(self.dim, metric_type)

        return index

    def _ensure_index(self):
        if self._index is None:
            self._index = self._build_index()
            self._trained = self._index_type == "Flat"

    def _normalize_if_needed(self, vectors: np.ndarray):
        if self._metric == "cosine":
            import faiss
            # 注意：faiss.normalize_L2 是 in-place 操作
            faiss.normalize_L2(vectors)

    def create(self) -> bool:
        self._ensure_index()
        self._trained = self._index_type == "Flat"
        return True

    def add(self, vectors: np.ndarray, ids: List[str], metadata: List[dict]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        else:
            # 避免修改调用方数组（归一化是 in-place 操作）
            vectors = vectors.copy()
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"维度不匹配：预期 {self.dim}，实际 {vectors.shape[1]}")

        self._normalize_if_needed(vectors)

        with self._lock:
            self._ensure_index()

            # 需要训练的索引，数据量足够时自动训练
            if not self._trained and self._index_type != "Flat":
                min_train = self._nlist * 39
                total_after = self.count() + len(vectors)
                if total_after >= min_train:
                    all_vecs = []
                    if self._index.ntotal > 0:
                        all_vecs.append(self._index.reconstruct_n(0, self._index.ntotal))
                    all_vecs.append(vectors)
                    train_data = np.vstack(all_vecs)
                    self._index.train(train_data)
                    self._trained = True
                    # 训练后重新添加已有数据
                    if self._index.ntotal > 0:
                        self._index.reset()
                        self._index.add(all_vecs[0])

            self._index.add(vectors)
            self.ids.extend(ids)
            self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int, filter_func: Optional[Callable] = None):
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        else:
            query = query.copy()
        if query.ndim == 1:
            query = query.reshape(1, -1)
        self._normalize_if_needed(query)

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return [], [], []
            search_k = min(top_k * 5, self._index.ntotal)
            if search_k <= 0:
                return [], [], []
            scores, indices = self._index.search(query, search_k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
            local_ids = self.ids
            local_meta = self.metadata

        result_ids, result_scores, result_meta = [], [], []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(local_ids):
                continue
            meta = local_meta[idx]
            if filter_func and not filter_func(meta):
                continue
            result_ids.append(local_ids[idx])
            result_scores.append(float(score))
            result_meta.append(meta)
            if len(result_ids) >= top_k:
                break

        return result_ids, result_scores, result_meta

    def delete(self, ids: List[str]) -> int:
        id_set = set(ids)
        with self._lock:
            keep_idx = [i for i, vid in enumerate(self.ids) if vid not in id_set]
            deleted = len(self.ids) - len(keep_idx)
            if deleted == 0:
                return 0

            if not keep_idx:
                if self._index is not None:
                    self._index.reset()
                self.ids = []
                self.metadata = []
                return deleted

            if self._index is not None:
                vecs = self._index.reconstruct_n(0, self._index.ntotal)
                vecs = vecs[keep_idx]
                self._index.reset()
                self._index.add(vecs)

            self.ids = [self.ids[i] for i in keep_idx]
            self.metadata = [self.metadata[i] for i in keep_idx]
            return deleted

    def drop(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.reset()
                self._index = None
            self.ids.clear()
            self.metadata.clear()

    def count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    def upgrade_index(self, new_type: str, new_quant: str = "None") -> bool:
        """平滑升级索引类型，数据零丢失"""
        if new_type == self._index_type and new_quant == self._quantization:
            return True

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                self._index_type = new_type
                self._quantization = new_quant
                self._index = None
                self._trained = False
                return True

            # 备份原有数据
            old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
            old_ids = self.ids.copy()
            old_meta = self.metadata.copy()
            old_trained = self._trained

            # 构建新索引
            old_type, old_quant = self._index_type, self._quantization
            needs_training = new_type != "Flat"
        
            if needs_training and len(old_vecs) < self._nlist * 39:
                # 数据量不足，暂不升级
                return False

            self._index_type = new_type
            self._quantization = new_quant
            try:
                self._index = self._build_index()
                self._trained = new_type == "Flat"
                # 训练并添加数据
                if not self._trained:
                    self._index.train(old_vecs)
                    self._trained = True
                self._index.add(old_vecs)
                return True
            except Exception as e:
                # 失败回滚
                self._index_type = old_type
                self._quantization = old_quant
                self._trained = old_trained
                self._index = self._build_index()
                self._index.add(old_vecs)
                self.ids = old_ids
                self.metadata = old_meta
                raise e

    def save(self, path: str):
        """持久化到磁盘"""
        import faiss
        with self._lock:
            if self._index is not None:
                faiss.write_index(self._index, f"{path}.index")
            with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "dim": self.dim,
                    "index_type": self._index_type,
                    "quantization": self._quantization,
                    "metric": self._metric,
                    "ids": self.ids,
                    "metadata": self.metadata,
                    "trained": self._trained,
                    "nlist": self._nlist,
                    "hnsw_m": self._hnsw_m,
                    "pq_m": self._pq_m
                }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, collection_name: str, config: dict) -> "LocalFAISSStore":
        import faiss
        with open(f"{path}.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        store = cls(collection_name, meta["dim"], {
            **config,
            "index_type": meta.get("index_type", "Flat"),
            "quantization": meta.get("quantization", "None"),
            "metric": meta.get("metric", "cosine"),
            "ivf_nlist": meta.get("nlist", 1024),
            "hnsw_m": meta.get("hnsw_m", 32),
            "pq_m": meta.get("pq_m", 8)
        })

        index_path = f"{path}.index"
        if os.path.exists(index_path):
            store._index = faiss.read_index(index_path)
            store._trained = meta.get("trained", True)
        store.ids = meta["ids"]
        store.metadata = meta["metadata"]
        return store


# ==================================================
# 外部向量库适配：Milvus / Qdrant / Chroma
# ==================================================
class MilvusStore(VectorStoreBase):
    """Milvus分布式向量库适配"""

    def __init__(self, collection_name: str, dim: int, config: dict):
        super().__init__(collection_name, dim, config)
        self.host = config.get("milvus_host", "localhost")
        self.port = config.get("milvus_port", 19530)
        # 修复：避免与 index_type 属性冲突，重命名为 _milvus_index_type
        self._milvus_index_type = config.get("index_type", "HNSW")
        self.metric = "IP" if config.get("metric", "cosine") == "cosine" else "L2"
        self._alias = f"plugin_{collection_name}"
        self._conn = False  # 连接状态标志
        # 修复：在 __init__ 中声明 _collection
        self._collection = None

    @property
    def index_type(self) -> str:
        return f"Milvus_{self._milvus_index_type}"

    def _connect(self):
        # 修复：使用公开API或标志位代替私有API _fetch_handler
        if not self._conn:
            from pymilvus import connections
            connections.connect(alias=self._alias, host=self.host, port=self.port)
            # 使用 get_connection_addr 验证连接，失败则抛异常
            addr = connections.get_connection_addr(self._alias)
            if addr is None:
                raise RuntimeError(f"Milvus 连接失败: {self.host}:{self.port}")
            self._conn = True

    def create(self) -> bool:
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
        self._connect()

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description=f"Plugin collection: {self.name}")
        self._collection = Collection(self.name, schema, using=self._alias)

        # 使用 _milvus_index_type 构建 index_params
        index_params = {
            "index_type": self._milvus_index_type,
            "metric_type": self.metric,
            "params": {"M": 32, "efConstruction": 200} if self._milvus_index_type == "HNSW" else {"nlist": 1024}
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        self._collection.load()
        return True

    def add(self, vectors: np.ndarray, ids: List[str], metadata: List[dict]) -> None:
        from pymilvus import Collection
        self._connect()
        self._collection = Collection(self.name, using=self._alias)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        entities = [ids, vectors.tolist(), metadata]
        self._collection.insert(entities)
        self._collection.flush()

    def search(self, query: np.ndarray, top_k: int, filter_func: Optional[Callable] = None):
        from pymilvus import Collection
        self._connect()
        self._collection = Collection(self.name, using=self._alias)

        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self._milvus_index_type == "HNSW":
            search_params = {"metric_type": self.metric, "params": {"ef": 128}}
        elif self._milvus_index_type == "IVF_FLAT" or self._milvus_index_type.startswith("IVF"):
            search_params = {"metric_type": self.metric, "params": {"nprobe": 10}}
        else:
            search_params = {"metric_type": self.metric, "params": {}}
        results = self._collection.search(
            query.tolist(),
            "vector",
            search_params,
            limit=top_k,
            output_fields=["metadata"]
        )

        ids, scores, meta = [], [], []
        for hit in results[0]:
            m = hit.entity.get("metadata")
            if filter_func and not filter_func(m):
                continue
            ids.append(hit.id)
            scores.append(float(hit.score))
            meta.append(m)
        return ids, scores, meta

    def delete(self, ids: List[str]) -> int:
        from pymilvus import Collection
        self._connect()
        self._collection = Collection(self.name, using=self._alias)
        quoted = [f'"{sid}"' for sid in ids]
        expr = f'id in [{", ".join(quoted)}]'
        self._collection.delete(expr)
        return len(ids)

    def drop(self) -> None:
        from pymilvus import utility
        self._connect()
        if utility.has_collection(self.name, using=self._alias):
            utility.drop_collection(self.name, using=self._alias)
        self._collection = None

    def count(self) -> int:
        from pymilvus import Collection
        self._connect()
        self._collection = Collection(self.name, using=self._alias)
        return self._collection.num_entities


class QdrantStore(VectorStoreBase):
    """Qdrant向量数据库适配"""
    def __init__(self, collection_name: str, dim: int, config: dict):
        super().__init__(collection_name, dim, config)
        self.host = config.get("qdrant_host", "localhost")
        self.port = config.get("qdrant_port", 6333)
        self._client = None

    @property
    def index_type(self) -> str:
        return "Qdrant_HNSW"

    def _connect(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=self.host, port=self.port)

    def create(self) -> bool:
        from qdrant_client.models import Distance, VectorParams
        self._connect()
        dist = Distance.COSINE if self.config.get("metric", "cosine") == "cosine" else Distance.EUCLID
        self._client.create_collection(
            collection_name=self.name,
            vectors_config=VectorParams(size=self.dim, distance=dist)
        )
        return True

    def add(self, vectors: np.ndarray, ids: List[str], metadata: List[dict]) -> None:
        from qdrant_client.models import PointStruct
        self._connect()
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # 修复：Qdrant 期望 UUID 或整数ID，通过 uuid5 从字符串ID确定性生成UUID
        points = []
        for str_id, v, m in zip(ids, vectors, metadata):
            point_id = uuid.uuid5(uuid.NAMESPACE_URL, str(str_id))
            points.append(PointStruct(id=point_id, vector=v.tolist(), payload=m))
        self._client.upsert(collection_name=self.name, points=points)

    def search(self, query: np.ndarray, top_k: int, filter_func: Optional[Callable] = None):
        self._connect()
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        results = self._client.search(
            collection_name=self.name,
            query_vector=query[0].tolist(),
            limit=top_k * 3
        )
        ids, scores, meta = [], [], []
        for hit in results:
            if filter_func and not filter_func(hit.payload or {}):
                continue
            ids.append(str(hit.id))
            scores.append(float(hit.score))
            meta.append(hit.payload or {})
            if len(ids) >= top_k:
                break
        return ids, scores, meta

    def delete(self, ids: List[str]) -> int:
        from qdrant_client.models import PointIdsList
        self._connect()
        point_ids = [uuid.uuid5(uuid.NAMESPACE_URL, str(sid)) for sid in ids]
        self._client.delete(collection_name=self.name, points_selector=PointIdsList(points=point_ids))
        return len(ids)

    def drop(self) -> None:
        self._connect()
        self._client.delete_collection(collection_name=self.name)

    def count(self) -> int:
        self._connect()
        info = self._client.get_collection(collection_name=self.name)
        return info.points_count


class ChromaStore(VectorStoreBase):
    """Chroma嵌入式/分布式向量库适配"""
    def __init__(self, collection_name: str, dim: int, config: dict):
        super().__init__(collection_name, dim, config)
        self._client = None
        self._collection = None

    @property
    def index_type(self) -> str:
        return "Chroma_HNSW"

    def _connect(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.config.get("persist_dir", "./data/chroma"))

    def create(self) -> bool:
        self._connect()
        self._collection = self._client.create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine" if self.config.get("metric") == "cosine" else "l2"}
        )
        return True

    def add(self, vectors: np.ndarray, ids: List[str], metadata: List[dict]) -> None:
        self._connect()
        if self._collection is None:
            self._collection = self._client.get_collection(self.name)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=metadata
        )

    def search(self, query: np.ndarray, top_k: int, filter_func: Optional[Callable] = None):
        self._connect()
        if self._collection is None:
            self._collection = self._client.get_collection(self.name)
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        res = self._collection.query(
            query_embeddings=query.tolist(),
            n_results=top_k * 3
        )
        # 修复：增加安全检查，distances[0] 可能为空
        if not res or "ids" not in res or not res["ids"] or not res["ids"][0]:
            return [], [], []
        distances = res.get("distances", [[]])
        if not distances or not distances[0]:
            return [], [], []
        ids = res["ids"][0]
        # 距离转相似度：使用倒数公式，适用于所有度量类型
        scores = [1.0 / (1.0 + float(s)) for s in distances[0]]
        metadatas = res.get("metadatas", [[]])
        meta = metadatas[0] if metadatas and metadatas[0] else []

        final_ids, final_scores, final_meta = [], [], []
        for i in range(min(len(ids), len(scores), len(meta))):
            m = meta[i] or {}
            if filter_func and not filter_func(m):
                continue
            final_ids.append(ids[i])
            final_scores.append(float(scores[i]))
            final_meta.append(m)
            if len(final_ids) >= top_k:
                break
        return final_ids, final_scores, final_meta

    def delete(self, ids: List[str]) -> int:
        self._connect()
        if self._collection is None:
            self._collection = self._client.get_collection(self.name)
        self._collection.delete(ids=ids)
        return len(ids)

    def drop(self) -> None:
        self._connect()
        self._client.delete_collection(self.name)
        self._collection = None

    def count(self) -> int:
        self._connect()
        if self._collection is None:
            self._collection = self._client.get_collection(self.name)
        return self._collection.count()


# ==================================================
# 存储工厂：统一创建入口
# ==================================================
class VectorStoreFactory:
    @staticmethod
    def create(store_type: str, collection_name: str, dim: int, config: dict) -> VectorStoreBase:
        if store_type == "faiss":
            store = LocalFAISSStore(collection_name, dim, config)
            store.create()
            return store
        elif store_type == "milvus":
            store = MilvusStore(collection_name, dim, config)
            store.create()
            return store
        elif store_type == "qdrant":
            store = QdrantStore(collection_name, dim, config)
            store.create()
            return store
        elif store_type == "chroma":
            store = ChromaStore(collection_name, dim, config)
            store.create()
            return store
        else:
            raise ValueError(f"不支持的存储类型: {store_type}")

    @staticmethod
    def load_local(path: str, collection_name: str, config: dict) -> LocalFAISSStore:
        return LocalFAISSStore.load(path, collection_name, config)


# ==================================================
# 主插件类：全功能增强版
# ==================================================
class VectorDatabaseProPlugin(PeripheralPlugin):
    """
    企业级全功能向量数据库插件
    支持多索引、量化、多后端、自动升级、多模态、嵌入函数集成
    完全兼容 AGI Agent v1.0.0 插件规范
    """

    def __init__(self):
        super().__init__(
            name="vector_database_pro",
            version="1.2.0",
            description="企业级向量数据库，多索引/量化/多后端/自动升级/多模态/嵌入集成，智能体统一记忆底座",
            plugin_type="processor",
            priority=PluginPriority.NORMAL,
            config={
                # 基础配置
                "default_dim": 768,
                "default_metric": "cosine",
                "default_store_type": "faiss",  # faiss / milvus / qdrant / chroma
                "default_index_type": "Flat",
                "default_quantization": "None",  # None / SQ8 / PQ

                # 自动索引升级
                "enable_auto_upgrade": True,
                "auto_upgrade_thresholds": {
                    10000: ("Flat", "None"),
                    100000: ("IVF", "None"),
                    1000000: ("IVF", "SQ8"),
                    10000000: ("HNSW", "SQ8")
                },
                "upgrade_check_interval": 600,  # 升级检查间隔(秒)

                # 持久化
                "persist_dir": "./data/vectors",
                "auto_persist": True,
                "persist_interval": 300,

                # 检索默认值
                "default_top_k": 10,
                "max_collections": 100,
                "auto_create_collection": True,

                # 多模态
                "enable_modality": True,
                "default_modality": "text",

                # 外部服务连接
                "milvus_host": "localhost",
                "milvus_port": 19530,
                "qdrant_host": "localhost",
                "qdrant_port": 6333,

                # 认知钩子
                "enable_cognition_hook": True,
                "memory_collection": "memory",

                # 嵌入函数配置（可传入 EmbeddingFunction 实例）
                "embedding_function": None,
                "latency_window": 100  # 查询延迟统计窗口
            },
            dependencies=[],
            compatible_versions=["1.0.0"],
            hook_points=[
                PluginHookPoint.PRE_COGNITION,
                PluginHookPoint.POST_COGNITION,
                PluginHookPoint.PERIODIC,
                PluginHookPoint.ON_STRUCTURE_CHANGE
            ]
        )

        self._collections: Dict[str, VectorStoreBase] = {}
        # 使用可重入锁：_op_add 持锁时会调用 _op_create_collection，需支持重入
        self._global_lock = threading.RLock()
        self._persist_thread: Optional[threading.Thread] = None
        self._upgrade_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(self.name)

        # 性能统计：查询延迟队列与计数器
        self._query_latencies: deque = deque(maxlen=self.config.get("latency_window", 100))
        self._total_queries: int = 0
        self._total_inserts: int = 0

        # 自动升级失败退避：集合名 -> (失败时间戳, 连续失败次数)
        self._upgrade_failures: Dict[str, tuple] = {}
        self._upgrade_backoff_base = 600  # 基础退避秒数
        self._upgrade_backoff_max = 86400  # 最大退避秒数（1天）

        # 嵌入函数初始化
        emb_fn = self.config.get("embedding_function")
        if emb_fn is not None and isinstance(emb_fn, EmbeddingFunction):
            self._embedding_fn: EmbeddingFunction = emb_fn
        else:
            self._embedding_fn = HashingEmbedding(dim=self.config["default_dim"])

    # ==================================================
    # 强制生命周期方法
    # ==================================================
    def on_load(self) -> bool:
        if self.status == PluginStatus.LOADED:
            return True

        # 基础依赖检查
        try:
            import faiss
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False

        if not self._faiss_available and self.config["default_store_type"] == "faiss":
            self._last_error = "默认存储为FAISS，但缺少依赖：pip install faiss-cpu"
            self._error_count += 1
            self.status = PluginStatus.ERROR
            return False

        try:
            os.makedirs(self.config["persist_dir"], exist_ok=True)
        except Exception as e:
            self._last_error = f"目录创建失败: {str(e)}"
            self._error_count += 1
            self.status = PluginStatus.ERROR
            return False

        self.status = PluginStatus.LOADED
        self._logger.info("向量数据库Pro版加载完成")
        return True

    def on_unload(self) -> bool:
        try:
            if self.status == PluginStatus.ACTIVE:
                self.on_deactivate()

            # 先持久化本地集合，再清理内存引用
            # 注意：不调用 col.drop()，卸载不等于删除数据
            with self._global_lock:
                persist_dir = self.config["persist_dir"]
                for name, col in self._collections.items():
                    if isinstance(col, LocalFAISSStore):
                        try:
                            path = os.path.join(persist_dir, name)
                            col.save(path)
                        except Exception as e:
                            self._logger.warning(f"卸载时持久化集合 {name} 失败: {str(e)}")
                self._collections.clear()

            self._persist_thread = None
            self._upgrade_thread = None
            self._stop_event.clear()
            self._query_latencies.clear()
            gc.collect()

            self.status = PluginStatus.UNLOADED
            self._logger.info("向量数据库Pro版已完全卸载")
            return True

        except Exception as e:
            self._last_error = f"卸载异常: {str(e)}"
            self._error_count += 1
            self.status = PluginStatus.ERROR
            return False

    def process(self, input_data: Any) -> Any:
        if self.status != PluginStatus.ACTIVE:
            return {"status": "error", "error": "向量数据库未处于活跃状态"}

        try:
            if not isinstance(input_data, dict) or "action" not in input_data:
                return {"status": "error", "error": "参数必须包含 action 字段"}

            action = input_data["action"]
            collection = input_data.get("collection", "default")
            store_type = input_data.get("store_type", self.config["default_store_type"])

            if action == "create_collection":
                return self._op_create_collection(
                    collection,
                    input_data.get("dim", self.config["default_dim"]),
                    store_type,
                    input_data.get("index_type", self.config["default_index_type"]),
                    input_data.get("quantization", self.config["default_quantization"]),
                    input_data.get("metric", self.config["default_metric"])
                )

            elif action == "add":
                # 支持文本自动编码
                if "text" in input_data:
                    texts = input_data["text"]
                    if isinstance(texts, str):
                        texts = [texts]
                    vectors = self._embedding_fn.encode_batch(texts)
                    if vectors.size == 0:
                        return {"status": "error", "error": "文本编码结果为空"}
                else:
                    vectors = np.array(input_data["vectors"], dtype=np.float32)

                modality = input_data.get("modality", self.config["default_modality"])
                metadata = input_data.get("metadata", [])
                # 注入模态字段
                if self.config["enable_modality"] and metadata:
                    for m in metadata:
                        m.setdefault("modality", modality)
                return self._op_add(
                    collection, vectors,
                    input_data.get("ids"), metadata
                )

            elif action == "update":
                # 更新操作：删除旧向量并重新添加
                if "text" in input_data:
                    texts = input_data["text"]
                    if isinstance(texts, str):
                        texts = [texts]
                    vectors = self._embedding_fn.encode_batch(texts)
                else:
                    vectors = np.array(input_data["vectors"], dtype=np.float32)
                ids = input_data.get("ids", [])
                metadata = input_data.get("metadata", [])
                if not ids:
                    return {"status": "error", "error": "更新操作必须提供 ids"}
                if not metadata:
                    metadata = [{} for _ in range(len(ids))]
                return self._op_update(collection, ids, vectors, metadata)

            elif action == "search":
                # 支持文本查询自动编码
                if "query_text" in input_data:
                    query_vec = self._embedding_fn.encode(input_data["query_text"])
                    query = query_vec.reshape(1, -1).astype(np.float32)
                else:
                    query = np.array([input_data["query_vector"]], dtype=np.float32)
                filter_meta = input_data.get("filter", {})
                # 模态过滤
                if "modality" in input_data:
                    filter_meta["modality"] = input_data["modality"]
                return self._op_search(
                    collection, query,
                    input_data.get("top_k", self.config["default_top_k"]),
                    filter_meta
                )

            elif action == "batch_search":
                # 批量查询
                if "query_texts" in input_data:
                    queries = self._embedding_fn.encode_batch(input_data["query_texts"]).astype(np.float32)
                else:
                    queries = np.array(input_data["query_vectors"], dtype=np.float32)
                if queries.ndim == 1:
                    queries = queries.reshape(1, -1)
                filter_meta = input_data.get("filter", {})
                return self._op_batch_search(
                    collection, queries,
                    input_data.get("top_k", self.config["default_top_k"]),
                    filter_meta
                )

            elif action == "hybrid_search":
                # 混合检索：向量相似度 + 关键词匹配
                if "query_text" in input_data:
                    query_vec = self._embedding_fn.encode(input_data["query_text"])
                    query = query_vec.reshape(1, -1).astype(np.float32)
                    query_text = input_data["query_text"]
                else:
                    query = np.array([input_data["query_vector"]], dtype=np.float32)
                    query_text = input_data.get("query_text", "")
                return self._op_hybrid_search(
                    collection, query,
                    input_data.get("top_k", self.config["default_top_k"]),
                    input_data.get("keyword_fields", ["content"]),
                    query_text
                )

            elif action == "range_filter":
                # 数值范围过滤检索
                field = input_data.get("field")
                if not field:
                    return {"status": "error", "error": "缺少 field 参数"}
                cond: Dict[str, Union[int, float]] = {}
                for op_key in ("gte", "gt", "lte", "lt"):
                    if op_key in input_data:
                        cond[op_key] = input_data[op_key]
                if not cond:
                    return {"status": "error", "error": "至少需要提供 gte/gt/lte/lt 中的一个"}
                range_filter = {"__range__": {field: cond}}
                if "query_text" in input_data:
                    query_vec = self._embedding_fn.encode(input_data["query_text"])
                    query = query_vec.reshape(1, -1).astype(np.float32)
                else:
                    query = np.array([input_data["query_vector"]], dtype=np.float32)
                return self._op_search(
                    collection, query,
                    input_data.get("top_k", self.config["default_top_k"]),
                    range_filter
                )

            elif action == "count":
                return self._op_count(collection)

            elif action == "get_collection_info":
                return self._op_collection_info(collection)

            elif action == "delete":
                return self._op_delete(collection, input_data.get("ids", []))

            elif action == "list_collections":
                return self._op_list_collections()

            elif action == "drop_collection":
                return self._op_drop_collection(collection)

            elif action == "upgrade_index":
                return self._op_upgrade_index(
                    collection,
                    input_data["new_index_type"],
                    input_data.get("new_quantization", "None")
                )

            elif action == "save_collection":
                return self._op_save_collection(collection)

            elif action == "load_collection":
                return self._op_load_collection(collection)

            else:
                return {"status": "error", "error": f"不支持的操作: {action}"}

        except Exception as e:
            self._last_error = f"操作失败: {str(e)}"
            self._error_count += 1
            return {"status": "error", "error": str(e)}

    def get_data(self) -> Dict[str, Any]:
        with self._global_lock:
            total_vectors = sum(col.count() for col in self._collections.values())

            # 查询延迟统计
            latencies = list(self._query_latencies)
            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                latency_stats = {
                    "count": n,
                    "avg": sum(latencies) / n,
                    "min": sorted_lat[0],
                    "max": sorted_lat[-1],
                    "p50": sorted_lat[n // 2],
                    "p95": sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[0]
                }
            else:
                latency_stats = {"count": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}

            # 集合详情：增加内存估算与索引健康状态
            collection_info = {}
            for name, col in self._collections.items():
                count = col.count()
                vector_bytes = count * col.dim * 4  # float32 每元素4字节
                meta_bytes = 0
                if hasattr(col, "metadata") and col.metadata:
                    sample_count = min(10, len(col.metadata))
                    total_sample_bytes = 0
                    for i in range(sample_count):
                        sample = json.dumps(col.metadata[i], ensure_ascii=False)
                        total_sample_bytes += len(sample.encode("utf-8"))
                    avg_meta_bytes = total_sample_bytes / sample_count
                    meta_bytes = int(avg_meta_bytes * count)
                idx = col.index_type
                if "HNSW" in idx:
                    overhead = vector_bytes * 0.3
                elif "IVF" in idx:
                    overhead = vector_bytes * 0.1
                else:
                    overhead = 0
                memory = vector_bytes + meta_bytes + overhead

                # 索引健康状态
                health = self._index_health(col)

                store_type = ("faiss" if isinstance(col, LocalFAISSStore) else
                              "milvus" if isinstance(col, MilvusStore) else
                              "qdrant" if isinstance(col, QdrantStore) else "chroma")

                collection_info[name] = {
                    "dim": col.dim,
                    "vector_count": count,
                    "index_type": col.index_type,
                    "store_type": store_type,
                    "memory_estimate": memory,
                    "memory_estimate_human": self._format_bytes(memory),
                    "index_health": health
                }

            # 多模态统计
            modality_stats: Dict[str, int] = {}
            if self.config["enable_modality"]:
                for col in self._collections.values():
                    if hasattr(col, "metadata"):
                        for m in col.metadata:
                            mod = m.get("modality", "unknown")
                            modality_stats[mod] = modality_stats.get(mod, 0) + 1

        return {
            "plugin_name": self.name,
            "version": self.version,
            "status": self.status.value,
            "statistics": {
                "total_vectors": total_vectors,
                "collection_count": len(self._collections),
                "modality_distribution": modality_stats,
                "total_queries": self._total_queries,
                "total_inserts": self._total_inserts,
                "query_latency": latency_stats
            },
            "config": {
                "default_store_type": self.config["default_store_type"],
                "auto_upgrade_enabled": self.config["enable_auto_upgrade"],
                "multi_modality": self.config["enable_modality"],
                "embedding_function": type(self._embedding_fn).__name__
            },
            "collections": collection_info,
            "last_error": self._last_error,
            "error_count": self._error_count
        }

    # ==================================================
    # 可选生命周期方法
    # ==================================================
    def on_activate(self) -> bool:
        if self.status == PluginStatus.ACTIVE:
            return True

        try:
            # 加载本地持久化集合
            if self.config["default_store_type"] == "faiss":
                self._load_local_collections()

            # 启动持久化线程
            if self.config["auto_persist"] and self.config["default_store_type"] == "faiss":
                self._stop_event.clear()
                self._persist_thread = threading.Thread(
                    target=self._persist_worker, daemon=True, name=f"{self.name}-persist"
                )
                self._persist_thread.start()

            # 启动自动索引升级线程
            if self.config["enable_auto_upgrade"] and self.config["default_store_type"] == "faiss":
                self._upgrade_thread = threading.Thread(
                    target=self._auto_upgrade_worker, daemon=True, name=f"{self.name}-upgrade"
                )
                self._upgrade_thread.start()

            self.status = PluginStatus.ACTIVE
            self._logger.info(f"向量数据库Pro版激活成功，已加载 {len(self._collections)} 个集合")
            return True

        except Exception as e:
            self._last_error = f"激活失败: {str(e)}"
            self._error_count += 1
            self.status = PluginStatus.ERROR
            return False

    def on_deactivate(self) -> bool:
        if self.status != PluginStatus.ACTIVE:
            return True

        try:
            self._stop_event.set()

            for thread in [self._persist_thread, self._upgrade_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=10)

            if self.config["auto_persist"] and self.config["default_store_type"] == "faiss":
                self._persist_all()

            self.status = PluginStatus.LOADED
            return True

        except Exception as e:
            self._last_error = f"停用异常: {str(e)}"
            self._error_count += 1
            return False

    def on_structure_change(self, new_dim: int) -> bool:
        try:
            mem_col = self.config["memory_collection"]
            default_col = "default"

            with self._global_lock:
                for col_name in [default_col, mem_col]:
                    if col_name in self._collections:
                        backup = f"{col_name}_backup_{int(time.time())}"
                        self._collections[backup] = self._collections.pop(col_name)

                self._collections[default_col] = VectorStoreFactory.create(
                    self.config["default_store_type"],
                    default_col, new_dim, self.config
                )

            self._logger.info(f"向量维度已适配为 {new_dim}")
            return True

        except Exception as e:
            self._last_error = f"维度变更失败: {str(e)}"
            self._error_count += 1
            return False

    # ==================================================
    # 钩子方法
    # ==================================================
    def hook_pre_cognition(self, input_data: np.ndarray) -> np.ndarray:
        if not self.config["enable_cognition_hook"]:
            return input_data
        try:
            mem_col = self.config["memory_collection"]
            with self._global_lock:
                if mem_col not in self._collections or self._collections[mem_col].count() == 0:
                    return input_data
            query = input_data.reshape(1, -1).astype(np.float32)
            self._collections[mem_col].search(query, top_k=3)
        except Exception:
            pass
        return input_data

    def hook_post_cognition(self, input_data: Any) -> Any:
        if not self.config["enable_cognition_hook"]:
            return input_data
        try:
            # 修复：优雅处理非dict输入或缺少vector键的情况
            if not isinstance(input_data, dict):
                return input_data
            if "vector" not in input_data:
                return input_data
            vec = np.array([input_data["vector"]], dtype=np.float32)
            mid = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:12]
            meta = {
                "content": input_data.get("content", ""),
                "timestamp": time.time(),
                "type": "cognition_memory",
                "modality": "text",
                **input_data.get("metadata", {})
            }
            self._op_add(self.config["memory_collection"], vec, [mid], [meta])
        except Exception:
            pass
        return input_data

    def hook_periodic(self, input_data: dict) -> dict:
        return input_data

    def hook_on_structure_change(self, input_data: int) -> int:
        self.on_structure_change(input_data)
        return input_data

    # ==================================================
    # 内部操作实现
    # ==================================================
    def _op_create_collection(self, name: str, dim: int, store_type: str, index_type: str, quantization: str, metric: str) -> dict:
        with self._global_lock:
            if name in self._collections:
                return {"status": "ok", "message": "集合已存在", "name": name}
            if len(self._collections) >= self.config["max_collections"]:
                return {"status": "error", "error": "已达最大集合数量限制"}

            col_config = self.config.copy()
            col_config["index_type"] = index_type
            col_config["quantization"] = quantization
            col_config["metric"] = metric

            store = VectorStoreFactory.create(store_type, name, dim, col_config)
            self._collections[name] = store
            return {"status": "ok", "name": name, "dim": dim, "store_type": store_type, "index_type": index_type}

    def _op_add(self, collection: str, vectors: np.ndarray, ids: Optional[List[str]], metadata: List[dict]) -> dict:
        with self._global_lock:
            if collection not in self._collections:
                if self.config["auto_create_collection"]:
                    if vectors.ndim == 1:
                        dim = vectors.shape[0]
                    else:
                        dim = vectors.shape[1]
                    self._op_create_collection(
                        collection, dim,
                        self.config["default_store_type"],
                        self.config["default_index_type"],
                        self.config["default_quantization"],
                        self.config["default_metric"]
                    )
                else:
                    return {"status": "error", "error": f"集合不存在: {collection}"}

            col = self._collections[collection]
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            count = len(vectors)
            if ids is None:
                ids = [uuid.uuid4().hex[:16] for _ in range(count)]
            if not metadata:
                metadata = [{} for _ in range(count)]

            col.add(vectors, ids, metadata)
            self._total_inserts += count

        return {"status": "ok", "added_count": count, "collection": collection}

    def _op_update(self, collection: str, ids: List[str], vectors: np.ndarray, metadata: List[dict]) -> dict:
        """更新操作：原子性删除旧向量并重新添加新向量"""
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "error", "error": f"集合不存在: {collection}"}
            col = self._collections[collection]
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.shape[0] != len(ids):
                return {"status": "error", "error": "vectors 数量与 ids 数量不匹配"}
            if vectors.shape[1] != col.dim:
                return {"status": "error", "error": f"维度不匹配：预期 {col.dim}，实际 {vectors.shape[1]}"}

            deleted = col.delete(ids)
            try:
                col.add(vectors, ids, metadata)
                self._total_inserts += len(ids)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"更新失败：删除成功但添加失败: {str(e)}",
                    "deleted_count": deleted
                }
        return {
            "status": "ok",
            "updated_count": len(ids),
            "deleted_old": deleted,
            "collection": collection
        }

    def _op_search(self, collection: str, query: np.ndarray, top_k: int, filter_meta: Optional[dict]):
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "ok", "results": [], "count": 0}
            col = self._collections[collection]

        # 解析等值过滤与范围过滤
        range_filters: Dict[str, dict] = {}
        equality_filters: Dict[str, Any] = {}
        if filter_meta:
            filter_copy = dict(filter_meta)
            range_filters = filter_copy.pop("__range__", {}) or {}
            equality_filters = filter_copy

        filter_func = None
        if equality_filters or range_filters:
            def filter_func(meta, _eq=equality_filters, _rg=range_filters):
                # 等值过滤
                for k, v in _eq.items():
                    if meta.get(k) != v:
                        return False
                # 范围过滤
                for field, cond in _rg.items():
                    val = meta.get(field)
                    if val is None:
                        return False
                    if "gte" in cond and val < cond["gte"]:
                        return False
                    if "gt" in cond and val <= cond["gt"]:
                        return False
                    if "lte" in cond and val > cond["lte"]:
                        return False
                    if "lt" in cond and val >= cond["lt"]:
                        return False
                return True

        start = time.time()
        ids, scores, meta = col.search(query, top_k, filter_func)
        latency = time.time() - start
        self._query_latencies.append(latency)
        self._total_queries += 1

        results = [{"id": i, "score": s, "metadata": m} for i, s, m in zip(ids, scores, meta)]
        return {"status": "ok", "results": results, "count": len(results)}

    def _op_batch_search(self, collection: str, queries: np.ndarray, top_k: int, filter_meta: Optional[dict]):
        """批量查询：一次提交多个查询向量，高效检索"""
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "ok", "batch_results": [], "count": 0}
            col = self._collections[collection]

        # 解析过滤条件
        range_filters: Dict[str, dict] = {}
        equality_filters: Dict[str, Any] = {}
        if filter_meta:
            filter_copy = dict(filter_meta)
            range_filters = filter_copy.pop("__range__", {}) or {}
            equality_filters = filter_copy

        filter_func = None
        if equality_filters or range_filters:
            def filter_func(meta, _eq=equality_filters, _rg=range_filters):
                for k, v in _eq.items():
                    if meta.get(k) != v:
                        return False
                for field, cond in _rg.items():
                    val = meta.get(field)
                    if val is None:
                        return False
                    if "gte" in cond and val < cond["gte"]:
                        return False
                    if "gt" in cond and val <= cond["gt"]:
                        return False
                    if "lte" in cond and val > cond["lte"]:
                        return False
                    if "lt" in cond and val >= cond["lt"]:
                        return False
                return True

        batch_results = []
        start = time.time()
        for q in queries:
            q_arr = q.reshape(1, -1).astype(np.float32)
            ids, scores, meta = col.search(q_arr, top_k, filter_func)
            batch_results.append([
                {"id": i, "score": s, "metadata": m}
                for i, s, m in zip(ids, scores, meta)
            ])
        latency = time.time() - start
        self._query_latencies.append(latency)
        self._total_queries += len(queries)

        total_count = sum(len(r) for r in batch_results)
        return {"status": "ok", "batch_results": batch_results, "count": total_count}

    def _op_hybrid_search(self, collection: str, query: np.ndarray, top_k: int, keyword_fields: List[str], query_text: str = ""):
        """混合检索：向量相似度 + 元数据关键词匹配重排"""
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "ok", "results": [], "count": 0}
            col = self._collections[collection]

        # 先做向量检索，扩大候选池
        candidate_k = min(top_k * 5, col.count()) if col.count() > 0 else top_k
        if candidate_k <= 0:
            return {"status": "ok", "results": [], "count": 0}

        start = time.time()
        ids, scores, meta = col.search(query, candidate_k, None)
        latency = time.time() - start
        self._query_latencies.append(latency)
        self._total_queries += 1

        # 获取度量类型用于分数归一化
        vec_metric = getattr(col, "_metric", getattr(col, "metric", "cosine"))
        if hasattr(col, "metric") and isinstance(col.metric, str):
            vec_metric = "cosine" if col.metric in ("IP", "COSINE") else "l2"

        # 关键词匹配重排
        query_keywords: Set[str] = set(query_text.lower().split()) if query_text else set()
        reranked = []
        for rid, score, m in zip(ids, scores, meta):
            keyword_score = 0.0
            if query_keywords:
                matched = 0
                total_fields = 0
                for field in keyword_fields:
                    field_val = str(m.get(field, "")).lower()
                    field_keywords = set(field_val.split())
                    if field_keywords:
                        total_fields += 1
                        intersection = query_keywords & field_keywords
                        if intersection:
                            union = query_keywords | field_keywords
                            matched += len(intersection) / len(union)
                if total_fields > 0:
                    keyword_score = matched / total_fields
            # 归一化向量分数到 [0, 1]
            if vec_metric == "cosine":
                normalized_vec_score = (score + 1) / 2
            else:
                normalized_vec_score = 1.0 / (1.0 + max(score, 0.0))
            combined_score = 0.7 * normalized_vec_score + 0.3 * keyword_score
            reranked.append({
                "id": rid,
                "score": float(score),
                "keyword_score": float(keyword_score),
                "combined_score": float(combined_score),
                "metadata": m
            })

        # 按组合分数降序排列
        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        results = reranked[:top_k]
        return {"status": "ok", "results": results, "count": len(results)}

    def _op_delete(self, collection: str, ids: List[str]) -> dict:
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "error", "error": f"集合不存在: {collection}"}
            deleted = self._collections[collection].delete(ids)
        return {"status": "ok", "deleted_count": deleted}

    def _op_count(self, collection: str) -> dict:
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "ok", "count": 0, "collection": collection}
            col = self._collections[collection]
            count = col.count()
        return {"status": "ok", "count": count, "collection": collection}

    def _op_collection_info(self, name: str) -> dict:
        """获取集合详细信息：维度、数量、索引、存储类型、内存估算、元数据样本"""
        with self._global_lock:
            if name not in self._collections:
                return {"status": "error", "error": f"集合不存在: {name}"}
            col = self._collections[name]
            count = col.count()

            # 内存估算
            vector_bytes = count * col.dim * 4  # float32 每元素4字节
            meta_bytes = 0
            if hasattr(col, "metadata") and col.metadata:
                sample_count = min(10, len(col.metadata))
                total_sample_bytes = 0
                for i in range(sample_count):
                    sample = json.dumps(col.metadata[i], ensure_ascii=False)
                    total_sample_bytes += len(sample.encode("utf-8"))
                avg_meta_bytes = total_sample_bytes / sample_count
                meta_bytes = int(avg_meta_bytes * count)
            idx = col.index_type
            if "HNSW" in idx:
                overhead = vector_bytes * 0.3
            elif "IVF" in idx:
                overhead = vector_bytes * 0.1
            else:
                overhead = 0
            memory_estimate = vector_bytes + meta_bytes + overhead

            # 存储类型
            if isinstance(col, LocalFAISSStore):
                store_type = "faiss"
            elif isinstance(col, MilvusStore):
                store_type = "milvus"
            elif isinstance(col, QdrantStore):
                store_type = "qdrant"
            elif isinstance(col, ChromaStore):
                store_type = "chroma"
            else:
                store_type = "unknown"

            # 元数据样本
            metadata_sample = []
            if hasattr(col, "metadata") and col.metadata:
                metadata_sample = col.metadata[:3]

            # 索引健康状态
            health = self._index_health(col)

        return {
            "status": "ok",
            "name": name,
            "dim": col.dim,
            "count": count,
            "index_type": idx,
            "store_type": store_type,
            "memory_estimate": memory_estimate,
            "memory_estimate_human": self._format_bytes(memory_estimate),
            "index_health": health,
            "metadata_sample": metadata_sample
        }

    def _op_list_collections(self) -> dict:
        with self._global_lock:
            info = {
                name: {
                    "dim": col.dim,
                    "vector_count": col.count(),
                    "index_type": col.index_type
                }
                for name, col in self._collections.items()
            }
        return {"status": "ok", "collections": info}

    def _op_drop_collection(self, collection: str) -> dict:
        with self._global_lock:
            if collection in self._collections:
                self._collections[collection].drop()
                del self._collections[collection]
                # 清理本地持久化文件
                path = os.path.join(self.config["persist_dir"], collection)
                try:
                    for ext in [".index", ".meta.json"]:
                        if os.path.exists(path + ext):
                            os.remove(path + ext)
                except Exception:
                    pass
        return {"status": "ok", "dropped": collection}

    def _op_upgrade_index(self, collection: str, new_type: str, new_quant: str) -> dict:
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "error", "error": f"集合不存在: {collection}"}
            col = self._collections[collection]
            if not isinstance(col, LocalFAISSStore):
                return {"status": "error", "error": "仅本地FAISS存储支持在线索引升级"}
            try:
                col.upgrade_index(new_type, new_quant)
                return {"status": "ok", "collection": collection, "new_index": f"{new_type}_{new_quant}".replace("_None", "")}
            except Exception as e:
                return {"status": "error", "error": f"升级失败: {str(e)}"}

    def _op_save_collection(self, collection: str) -> dict:
        """持久化指定集合到磁盘"""
        path = os.path.join(self.config["persist_dir"], collection)
        with self._global_lock:
            if collection not in self._collections:
                return {"status": "error", "error": f"集合不存在: {collection}"}
            col = self._collections[collection]
            if not isinstance(col, LocalFAISSStore):
                return {"status": "error", "error": "仅本地FAISS存储支持持久化保存"}
            try:
                col.save(path)
            except Exception as e:
                return {"status": "error", "error": f"保存失败: {str(e)}"}
        return {"status": "ok", "saved": collection, "path": path}

    def _op_load_collection(self, collection: str) -> dict:
        """从磁盘加载指定集合"""
        with self._global_lock:
            if collection in self._collections:
                return {"status": "ok", "message": "集合已加载", "name": collection}
            path = os.path.join(self.config["persist_dir"], collection)
            if not os.path.exists(f"{path}.meta.json"):
                return {"status": "error", "error": f"持久化文件不存在: {collection}"}
            try:
                self._collections[collection] = VectorStoreFactory.load_local(path, collection, self.config)
            except Exception as e:
                return {"status": "error", "error": f"加载失败: {str(e)}"}
        return {"status": "ok", "loaded": collection}

    # ==================================================
    # 辅助方法：内存估算与索引健康检查
    # ==================================================
    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        """字节数格式化为人类可读字符串"""
        size = float(num_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    @staticmethod
    def _index_health(col: VectorStoreBase) -> str:
        """索引健康状态检查"""
        if isinstance(col, LocalFAISSStore):
            if col._index is None:
                return "no_index"
            if not col._trained:
                return "untrained"
            if col.count() == 0:
                return "empty"
            if len(col.ids) != col._index.ntotal:
                return "inconsistent"
            return "healthy"
        elif isinstance(col, MilvusStore):
            return "healthy" if getattr(col, "_conn", False) else "disconnected"
        elif isinstance(col, QdrantStore):
            return "healthy" if getattr(col, "_client", None) is not None else "disconnected"
        elif isinstance(col, ChromaStore):
            return "healthy" if getattr(col, "_collection", None) is not None else "no_collection"
        return "unknown"

    # ==================================================
    # 后台线程：持久化 + 自动索引升级
    # ==================================================
    def _persist_worker(self):
        while not self._stop_event.is_set():
            try:
                time.sleep(self.config["persist_interval"])
                if self._stop_event.is_set():
                    break
                self._persist_all()
            except Exception as e:
                self._logger.error(f"持久化线程异常: {str(e)}")
                time.sleep(60)

    def _persist_all(self):
        with self._global_lock:
            for name, col in self._collections.items():
                if isinstance(col, LocalFAISSStore):
                    try:
                        path = os.path.join(self.config["persist_dir"], name)
                        col.save(path)
                    except Exception as e:
                        self._logger.warning(f"集合 {name} 持久化失败: {str(e)}")

    def _auto_upgrade_worker(self):
        while not self._stop_event.is_set():
            try:
                time.sleep(self.config["upgrade_check_interval"])
                if self._stop_event.is_set():
                    break

                thresholds = sorted(self.config["auto_upgrade_thresholds"].items(), key=lambda x: x[0])
                now = time.time()

                with self._global_lock:
                    for name, col in self._collections.items():
                        if not isinstance(col, LocalFAISSStore):
                            continue
                        # 失败退避检查
                        if name in self._upgrade_failures:
                            fail_time, fail_count = self._upgrade_failures[name]
                            backoff = min(self._upgrade_backoff_base * (2 ** (fail_count - 1)), self._upgrade_backoff_max)
                            if now - fail_time < backoff:
                                continue

                        count = col.count()
                        target_idx, target_quant = "Flat", "None"
                        for thresh, (idx, quant) in thresholds:
                            if count >= thresh:
                                target_idx, target_quant = idx, quant
                        if target_idx != col._index_type or target_quant != col._quantization:
                            self._logger.info(f"集合 {name} 自动升级索引: {col.index_type} -> {target_idx}_{target_quant}")
                            try:
                                col.upgrade_index(target_idx, target_quant)
                                self._upgrade_failures.pop(name, None)
                            except Exception as e:
                                self._logger.warning(f"自动升级失败: {str(e)}")
                                old_fail = self._upgrade_failures.get(name, (now, 0))
                                self._upgrade_failures[name] = (now, old_fail[1] + 1)
            except Exception as e:
                self._logger.error(f"索引升级线程异常: {str(e)}")
                time.sleep(300)

    def _load_local_collections(self):
        persist_dir = self.config["persist_dir"]
        if not os.path.isdir(persist_dir):
            return

        with self._global_lock:
            for filename in os.listdir(persist_dir):
                if filename.endswith(".meta.json"):
                    name = filename[:-10]
                    try:
                        path = os.path.join(persist_dir, name)
                        self._collections[name] = VectorStoreFactory.load_local(path, name, self.config)
                    except Exception as e:
                        self._logger.warning(f"加载集合 {name} 失败: {str(e)}")


# ==================================================
# 工厂函数入口
# ==================================================
def create_plugin():
    return VectorDatabaseProPlugin()
