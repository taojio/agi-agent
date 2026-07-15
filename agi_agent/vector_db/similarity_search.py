"""
vector_db/similarity_search.py - 相似度检索 (T015)

按阈值与 TopN 检索向量。优先使用底层向量库自带检索能力，
降级使用 numpy 余弦相似度暴力检索。
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.vector_db")

# 可选依赖标记（仅用于能力判断，实际后端由 store 决定）
try:  # pragma: no cover - 环境相关
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False


@dataclass
class SearchResult:
    """单条检索结果"""
    vector_id: str
    score: float
    metadata: Dict[str, Any]


class SimilaritySearch(BaseModule):
    """向量相似度检索器 (T015)

    绑定一个 VectorStoreWriter 实例进行检索。后端为 chroma/faiss 时优先
    使用其原生检索；否则使用 numpy 余弦相似度暴力检索。
    """

    name = "similarity_search"
    version = "1.0.0"
    description = "向量相似度检索 (T015)"

    def __init__(self, store: Optional[Any] = None):
        super().__init__()
        self._store = store

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._store is None:
            logger.info("SimilaritySearch 未绑定 store，仅支持空检索")
        else:
            logger.info("SimilaritySearch 绑定 store 后端: %s", getattr(self._store, "backend", "unknown"))

    def _shutdown(self) -> None:
        pass

    def _health_check(self) -> bool:
        return True

    # ====== 公共方法 ======
    def bind(self, store: Any) -> None:
        """绑定向量库写入器"""
        self._store = store

    def search(
        self,
        query_vector: np.ndarray,
        topn: int = 5,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """按相似度检索 TopN

        Args:
            query_vector: 查询向量
            topn: 返回结果数上限
            threshold: 相似度阈值（仅返回 score >= threshold 的结果）

        Returns:
            List[SearchResult]: 命中结果，按分数降序
        """
        if self._store is None:
            return []
        topn = max(0, int(topn))
        if topn == 0:
            return []

        qv = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(qv))
        if norm > 1e-12:
            qv = qv / norm

        backend = getattr(self._store, "backend", "numpy")
        if backend == "chroma":
            results = self._search_chroma(qv, topn)
        elif backend == "faiss":
            results = self._search_faiss(qv, topn)
        else:
            results = self._search_numpy(qv, topn)

        return [r for r in results if r.score >= threshold]

    def search_by_metadata(self, filter: Dict[str, Any], topn: int = 100) -> List[SearchResult]:
        """按元数据过滤检索

        Args:
            filter: 元数据过滤条件键值对
            topn: 返回结果数上限

        Returns:
            List[SearchResult]: 命中结果
        """
        if self._store is None or not filter:
            return []
        topn = max(0, int(topn))

        backend = getattr(self._store, "backend", "numpy")
        if backend == "chroma" and self._store._chroma_collection is not None:
            try:
                chroma_filter = {k: v for k, v in filter.items() if isinstance(v, (str, int, float, bool))}
                payload = self._store._chroma_collection.get(
                    where=chroma_filter, limit=topn, include=["metadatas"]
                )
                ids = payload.get("ids", [])
                metas = payload.get("metadatas", [])
                return [SearchResult(vector_id=i, score=1.0, metadata=m or {}) for i, m in zip(ids, metas)]
            except Exception as e:  # noqa: BLE001
                logger.warning("chromadb 元数据检索失败，降级内存: %s", e)

        # numpy 后端逐条匹配
        results: List[SearchResult] = []
        for rec in self._store.get_records():
            meta = rec.metadata
            if all(meta.get(k) == v for k, v in filter.items()):
                results.append(SearchResult(vector_id=rec.vector_id, score=1.0, metadata=meta))
                if len(results) >= topn:
                    break
        return results

    # ====== 后端实现 ======
    def _search_chroma(self, qv: np.ndarray, topn: int) -> List[SearchResult]:
        try:
            payload = self._store._chroma_collection.query(
                query_embeddings=[qv.tolist()],
                n_results=topn,
                include=["metadatas", "distances"],
            )
            ids = (payload.get("ids") or [[]])[0]
            dists = (payload.get("distances") or [[]])[0]
            metas = (payload.get("metadatas") or [[]])[0]
            out: List[SearchResult] = []
            for vid, d, m in zip(ids, dists, metas):
                score = float(1.0 - d) if d is not None else 0.0
                out.append(SearchResult(vector_id=str(vid), score=score, metadata=m or {}))
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("chromadb 查询失败，降级 numpy: %s", e)
            return self._search_numpy(qv, topn)

    def _search_faiss(self, qv: np.ndarray, topn: int) -> List[SearchResult]:
        try:
            index = self._store._faiss_index
            k = min(topn, int(index.ntotal))
            if k <= 0:
                return []
            scores, idxs = index.search(qv.reshape(1, -1).astype(np.float32), k)
            records = self._store.get_records()
            out: List[SearchResult] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or idx >= len(records):
                    continue
                rec = records[idx]
                out.append(SearchResult(vector_id=rec.vector_id, score=float(score), metadata=rec.metadata))
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning("faiss 查询失败，降级 numpy: %s", e)
            return self._search_numpy(qv, topn)

    def _search_numpy(self, qv: np.ndarray, topn: int) -> List[SearchResult]:
        matrix = self._store.get_matrix()
        records = self._store.get_records()
        if matrix is None or not records:
            return []
        scores = matrix @ qv  # 已归一化即余弦相似度
        k = min(topn, len(records))
        if k <= 0:
            return []
        # 取 TopK（argpartition 加速，再排序）
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out: List[SearchResult] = []
        for i in top_idx:
            i = int(i)
            out.append(
                SearchResult(
                    vector_id=records[i].vector_id,
                    score=float(scores[i]),
                    metadata=records[i].metadata,
                )
            )
        return out
