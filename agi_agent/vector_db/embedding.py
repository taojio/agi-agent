"""
vector_db/embedding.py - 文本/图像向量化编码 (T013)

接收原始文本/图片，输出标准化向量。优先使用 sentence-transformers，
降级使用 numpy 实现的哈希向量（固定维度 384）。统一 L2 归一化。
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.vector_db")

# 可选依赖：sentence-transformers
try:  # pragma: no cover - 环境相关
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_ST = True
except Exception:  # noqa: BLE001
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

# 可选依赖：PIL（图像读取降级时仍可用，但非必须）
try:  # pragma: no cover - 环境相关
    from PIL import Image  # type: ignore

    _HAS_PIL = True
except Exception:  # noqa: BLE001
    Image = None  # type: ignore
    _HAS_PIL = False


@dataclass
class EmbeddingConfig:
    """向量化编码配置"""
    model_name: str = "all-MiniLM-L6-v2"
    dim: int = 384
    use_sentence_transformers: bool = True
    device: Optional[str] = None
    normalize: bool = True


class EmbeddingEncoder(BaseModule):
    """文本/图像向量化编码器 (T013)

    优先使用 sentence-transformers 编码；当其不可用时降级为基于 numpy
    的哈希向量（文本使用带符号哈希，图像使用字节哈希种子生成确定性向量）。
    所有输出向量统一 L2 归一化。
    """

    name = "embedding_encoder"
    version = "1.0.0"
    description = "文本/图像向量化编码 (T013)"

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__()
        self._cfg = config or EmbeddingConfig()
        self._model = None
        self._backend: str = "numpy"

    # ====== 生命周期 ======
    def _initialize(self, config: dict) -> None:
        if self._cfg.use_sentence_transformers and _HAS_ST:
            try:
                kwargs: dict = {}
                if self._cfg.device:
                    kwargs["device"] = self._cfg.device
                self._model = SentenceTransformer(self._cfg.model_name, **kwargs)  # type: ignore[misc]
                self._backend = "sentence_transformers"
                logger.info("EmbeddingEncoder 使用 sentence-transformers 后端: %s", self._cfg.model_name)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("sentence-transformers 初始化失败，降级为 numpy 哈希向量: %s", e)
                self._model = None
        self._backend = "numpy"
        logger.info("EmbeddingEncoder 使用 numpy 哈希向量后端 (dim=%d)", self._cfg.dim)

    def _shutdown(self) -> None:
        self._model = None

    def _health_check(self) -> bool:
        return self._backend in ("sentence_transformers", "numpy")

    # ====== 核心方法 ======
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return self._cfg.dim

    def encode_text(self, text: str) -> np.ndarray:
        """对文本进行向量化编码

        Args:
            text: 原始文本

        Returns:
            np.ndarray: L2 归一化后的向量
        """
        if not isinstance(text, str):
            text = str(text)
        if self._backend == "sentence_transformers" and self._model is not None:
            try:
                vec = np.asarray(self._model.encode(text, convert_to_numpy=True), dtype=np.float32)
                return self._maybe_normalize(vec)
            except Exception as e:  # noqa: BLE001
                logger.warning("sentence-transformers 编码失败，降级单条: %s", e)
        return self._hash_text(text)

    def encode_image(self, image: Union[str, bytes, Any]) -> np.ndarray:
        """对图像进行向量化编码

        Args:
            image: 图像路径 / 二进制字节 / PIL.Image / numpy 数组

        Returns:
            np.ndarray: L2 归一化后的向量
        """
        if self._backend == "sentence_transformers" and self._model is not None and _HAS_PIL:
            try:
                img = self._to_pil_image(image)
                if img is not None:
                    vec = np.asarray(self._model.encode(img, convert_to_numpy=True), dtype=np.float32)
                    return self._maybe_normalize(vec)
            except Exception as e:  # noqa: BLE001
                logger.warning("sentence-transformers 图像编码失败，降级哈希: %s", e)
        return self._hash_image(image)

    def encode_batch(self, items: List[Union[str, Any]]) -> List[np.ndarray]:
        """批量编码

        Args:
            items: 文本或图像列表（统一类型效果更佳）

        Returns:
            List[np.ndarray]: 归一化向量列表
        """
        if not items:
            return []
        if self._backend == "sentence_transformers" and self._model is not None:
            texts = [x for x in items if isinstance(x, str)]
            if len(texts) == len(items):
                try:
                    mat = np.asarray(self._model.encode(items, convert_to_numpy=True), dtype=np.float32)
                    return [self._maybe_normalize(mat[i]) for i in range(mat.shape[0])]
                except Exception as e:  # noqa: BLE001
                    logger.warning("sentence-transformers 批量编码失败，逐条降级: %s", e)
        return [self._encode_one(x) for x in items]

    # ====== 内部工具 ======
    def _encode_one(self, item: Union[str, Any]) -> np.ndarray:
        if isinstance(item, str):
            return self.encode_text(item)
        return self.encode_image(item)

    def _maybe_normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if self._cfg.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
        return vec

    def _hash_text(self, text: str) -> np.ndarray:
        """基于带符号哈希的文本向量（降级方案）"""
        dim = self._cfg.dim
        vec = np.zeros(dim, dtype=np.float32)
        for raw in text.lower().split():
            token = raw.strip(".,;:!?\"'()[]{}，。；：！？、（）")
            if not token:
                continue
            h = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign
        if float(np.linalg.norm(vec)) < 1e-12:
            # 文本为空时给出确定性非零向量
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype(np.float32)
        return self._maybe_normalize(vec)

    def _hash_image(self, image: Union[str, bytes, Any]) -> np.ndarray:
        """基于字节哈希种子的图像向量（降级方案）"""
        dim = self._cfg.dim
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
        elif isinstance(image, str):
            payload = image.encode("utf-8")
        else:
            try:
                arr = np.asarray(image, dtype=np.uint8).flatten()
                payload = arr.tobytes()
            except Exception:  # noqa: BLE001
                payload = repr(image).encode("utf-8", errors="ignore")
        seed = int(hashlib.md5(payload).hexdigest()[:8], 16) if payload else 0
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        return self._maybe_normalize(vec)

    @staticmethod
    def _to_pil_image(image: Any):  # type: ignore[no-untyped-def]
        if not _HAS_PIL:
            return None
        if isinstance(image, Image.Image):  # type: ignore[attr-defined]
            return image
        if isinstance(image, (bytes, bytearray)):
            import io

            try:
                return Image.open(io.BytesIO(bytes(image)))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return None
        if isinstance(image, str):
            try:
                return Image.open(image)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return None
        if isinstance(image, np.ndarray):
            try:
                return Image.fromarray(image)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return None
        return None
