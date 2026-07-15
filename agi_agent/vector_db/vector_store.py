"""
vector_db/vector_store.py - 向量库写入存储 (T014)

接收向量与元数据（时间、场景、标签、用户ID），支持批量/单条写入。
优先使用 chromadb / faiss，降级使用内存 numpy 矩阵 + pickle 持久化到
./data/vector_store.pkl。
"""
import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.vector_db")

# 可选依赖：faiss
try:  # pragma: no cover - 环境相关
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False

# 可选依赖：chromadb
try:  # pragma: no cover - 环境相关
    import chromadb  # type: ignore

    _HAS_CHROMA = True
except Exception:  # noqa: BLE001
    chromadb = None  # type: ignore
    _HAS_CHROMA = False


@dataclass
class VectorStoreConfig:
    """向量库存储配置"""
    backend: str = "auto"  # auto | chroma | faiss | numpy
    dim: int = 384
    collection_name: str = "agi_default"
    persist_path: str = "./data/vector_store.pkl"
    chroma_path: str = "./data/chroma"
    normalize_on_write: bool = False


@dataclass
class VectorRecord:
    """单条向量记录"""
    vector_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float


class VectorStoreWriter(BaseModule):
    """向量库写入器 (T014)

    优先使用 chromadb；其次 faiss；均不可用时降级为内存 numpy 矩阵 +
    pickle 持久化。提供 write / write_batch / count / flush 方法。
    """

    name = "vector_store_writer"
    version = "1.0.0"
    description = "向量库写入存储 (T014)"

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        super().__init__()
        self._cfg = config or VectorStoreConfig()
        self._backend: str = "numpy"
        self._records: List[VectorRecord] = []
        self._index_by_id: Dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None
        self._chroma_collection = None
        self._faiss_index = None
        self._dirty: bool = False

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        backend = self._cfg.backend
        if backend in ("auto", "chroma") and _HAS_CHROMA:
            try:
                client = chromadb.PersistentClient(path=self._cfg.chroma_path)  # type: ignore[attr-defined]
                self._chroma_collection = client.get_or_create_collection(
                    name=self._cfg.collection_name
                )
                self._backend = "chroma"
                logger.info("VectorStoreWriter 使用 chromadb 后端: %s", self._cfg.chroma_path)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("chromadb 初始化失败: %s", e)
        if backend in ("auto", "faiss") and _HAS_FAISS:
            try:
                self._faiss_index = faiss.IndexFlatIP(self._cfg.dim)  # type: ignore[attr-defined]
                self._backend = "faiss"
                logger.info("VectorStoreWriter 使用 faiss 后端 (dim=%d)", self._cfg.dim)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("faiss 初始化失败: %s", e)
        self._backend = "numpy"
        self._load_pickle()
        logger.info("VectorStoreWriter 使用 numpy 后端 (dim=%d)", self._cfg.dim)

    def _shutdown(self) -> None:
        try:
            if self._dirty:
                self.flush()
        except Exception:  # noqa: BLE001
            pass
        self._chroma_collection = None
        self._faiss_index = None

    def _health_check(self) -> bool:
        return self._backend in ("chroma", "faiss", "numpy")

    # ====== 公共方法 ======
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return self._cfg.dim

    def write(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> str:
        """写入单条向量

        Args:
            vector: 向量
            metadata: 元数据（时间、场景、标签、用户ID 等）

        Returns:
            str: 生成的 vector_id
        """
        vec = self._prepare_vector(vector)
        meta = dict(metadata or {})
        meta.setdefault("timestamp", time.time())
        vector_id = str(meta.get("vector_id") or uuid.uuid4().hex)
        meta.setdefault("created_at", meta["timestamp"])

        if self._backend == "chroma" and self._chroma_collection is not None:
            try:
                self._chroma_collection.add(
                    ids=[vector_id],
                    embeddings=[vec.tolist()],
                    metadatas=[self._sanitize_meta(meta)],
                )
                self._dirty = True
                return vector_id
            except Exception as e:  # noqa: BLE001
                logger.warning("chromadb 写入失败，降级内存: %s", e)
                self._backend = "numpy"

        if self._backend == "faiss" and self._faiss_index is not None:
            try:
                self._faiss_index.add(vec.reshape(1, -1).astype(np.float32))
            except Exception as e:  # noqa: BLE001
                logger.warning("faiss 写入失败，降级内存: %s", e)
                self._backend = "numpy"

        # numpy 后端（含降级）
        idx = len(self._records)
        rec = VectorRecord(
            vector_id=vector_id,
            vector=vec,
            metadata=meta,
            created_at=float(meta.get("timestamp", time.time())),
        )
        self._records.append(rec)
        self._index_by_id[vector_id] = idx
        self._matrix = None  # 失效缓存
        self._dirty = True
        return vector_id

    def write_batch(self, items: List[Tuple[np.ndarray, Optional[Dict[str, Any]]]]) -> List[str]:
        """批量写入

        Args:
            items: (vector, metadata) 列表

        Returns:
            List[str]: vector_id 列表
        """
        ids: List[str] = []
        for vec, meta in items:
            ids.append(self.write(vec, meta))
        return ids

    def count(self) -> int:
        """返回当前存储向量数量"""
        if self._backend == "chroma" and self._chroma_collection is not None:
            try:
                return int(self._chroma_collection.count())
            except Exception:  # noqa: BLE001
                return len(self._records)
        if self._backend == "faiss" and self._faiss_index is not None:
            try:
                return int(self._faiss_index.ntotal)
            except Exception:  # noqa: BLE001
                return len(self._records)
        return len(self._records)

    def flush(self) -> None:
        """持久化落盘"""
        if self._backend == "chroma":
            self._dirty = False
            return
        if self._backend == "faiss":
            self._dirty = False
            return
        self._save_pickle()
        self._dirty = False

    # ====== 内部访问（供 SimilaritySearch / IndexMaintenance 使用） ======
    def get_records(self) -> List[VectorRecord]:
        """返回内存中的全部记录（numpy 后端）"""
        return self._records

    def get_matrix(self) -> Optional[np.ndarray]:
        """返回堆叠后的向量矩阵（numpy 后端，无记录时返回 None）"""
        if self._backend != "numpy":
            return None
        if not self._records:
            return None
        if self._matrix is None:
            self._matrix = np.vstack([r.vector for r in self._records]).astype(np.float32)
        return self._matrix

    def remove_ids(self, vector_ids: List[str]) -> int:
        """按 ID 删除记录（numpy 后端），返回实际删除数量"""
        if not vector_ids:
            return 0
        if self._backend == "chroma" and self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=list(vector_ids))
                self._dirty = True
                return len(vector_ids)
            except Exception:  # noqa: BLE001
                pass
        removed = 0
        keep: List[VectorRecord] = []
        for rec in self._records:
            if rec.vector_id in vector_ids:
                removed += 1
            else:
                keep.append(rec)
        if removed:
            self._records = keep
            self._index_by_id = {r.vector_id: i for i, r in enumerate(self._records)}
            self._matrix = None
            self._dirty = True
        return removed

    def rebuild_matrix(self) -> None:
        """重建内存矩阵缓存"""
        self._matrix = None
        _ = self.get_matrix()

    # ====== 工具 ======
    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self._cfg.dim:
            # 维度不一致时截断或补零，保证写入不抛异常
            if vec.shape[0] > self._cfg.dim:
                vec = vec[: self._cfg.dim]
            else:
                tmp = np.zeros(self._cfg.dim, dtype=np.float32)
                tmp[: vec.shape[0]] = vec
                vec = tmp
        if self._cfg.normalize_on_write:
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
        return vec

    @staticmethod
    def _sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        """chromadb 仅接受基础类型 metadata，做一次安全转换"""
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def _load_pickle(self) -> None:
        path = self._cfg.persist_path
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._records = list(data.get("records", []))
                self._index_by_id = {r.vector_id: i for i, r in enumerate(self._records)}
                self._matrix = None
                logger.info("VectorStoreWriter 从 %s 载入 %d 条记录", path, len(self._records))
        except Exception as e:  # noqa: BLE001
            logger.warning("加载向量持久化文件失败: %s", e)
            self._records = []
            self._index_by_id = {}

    def _save_pickle(self) -> None:
        path = self._cfg.persist_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"records": self._records, "dim": self._cfg.dim}, f)
        except Exception as e:  # noqa: BLE001
            logger.warning("保存向量持久化文件失败: %s", e)
