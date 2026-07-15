"""
vector_db/__init__.py - 向量数据库引擎子模块 (T013-T016)

提供向量化编码、向量库写入存储、相似度检索、索引维护能力。
"""
from .embedding import EmbeddingConfig, EmbeddingEncoder
from .index_maintenance import IndexMaintenanceConfig, VectorIndexMaintenance
from .similarity_search import SearchResult, SimilaritySearch
from .vector_store import VectorRecord, VectorStoreConfig, VectorStoreWriter

__all__ = [
    "EmbeddingConfig",
    "EmbeddingEncoder",
    "VectorStoreConfig",
    "VectorStoreWriter",
    "VectorRecord",
    "SearchResult",
    "SimilaritySearch",
    "IndexMaintenanceConfig",
    "VectorIndexMaintenance",
]
