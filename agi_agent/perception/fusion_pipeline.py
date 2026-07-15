"""
perception/fusion_pipeline.py - 多模态融合流水线

实现任务 T041-T043：多模态时序对齐、感知信息结构化封装、无效模态过滤。

注意：本文件不修改已有的 multimodal_fusion.py，而是新增模块。
numpy 可用；其余依赖均为纯 Python 实现。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.perception")


# ============================ 数据结构 ============================

@dataclass
class AlignedBundle:
    """T041 多模态时序对齐后的报文包"""

    timestamp: float
    text: Optional[str] = None
    frame: Optional[Any] = None      # Frame 对象，使用 Any 避免循环引用
    audio: Optional[Any] = None      # AudioChunk 对象
    session_id: str = ""


@dataclass
class PerceptionReport:
    """T042 标准化感知报文

    可序列化为 JSON。numpy 数组在序列化时转为列表。
    """

    session_id: str
    timestamp: float
    text: str = ""
    visual: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典（numpy 数组/标量自动转换）"""
        def _conv(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, dict):
                return {k: _conv(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_conv(x) for x in v]
            return v
        d = asdict(self)
        return {k: _conv(val) for k, val in d.items()}


# ============================ T041 ============================

class MultimodalTimeAligner(BaseModule):
    """T041 多模态时序对齐

    动态调度模块。基于时间戳将文本/图像/音频对齐为同一时间窗口的报文包。
    """

    name = "multimodal_time_aligner"
    version = "1.0.0"
    description = "基于时间戳对齐文本/图像/音频"

    def __init__(self, time_window: float = 0.5) -> None:
        super().__init__()
        self.time_window = time_window

    def align(self,
              text_msgs: Optional[List[Any]] = None,
              frames: Optional[List[Any]] = None,
              audio_chunks: Optional[List[Any]] = None) -> AlignedBundle:
        """按时间窗口对齐三种模态

        Args:
            text_msgs: RawTextMessage 或字符串列表（带 timestamp 属性优先）
            frames: Frame 列表
            audio_chunks: AudioChunk 列表

        Returns:
            AlignedBundle: 对齐后的报文包
        """
        text_msgs = text_msgs or []
        frames = frames or []
        audio_chunks = audio_chunks or []
        ts_text = self._get_timestamp(text_msgs)
        ts_frame = self._get_timestamp(frames)
        ts_audio = self._get_timestamp(audio_chunks)
        candidates = [t for t in (ts_text, ts_frame, ts_audio) if t is not None]
        anchor = min(candidates) if candidates else time.time()
        text = self._pick_text(text_msgs, anchor)
        frame = self._pick(frames, anchor)
        audio = self._pick(audio_chunks, anchor)
        session_id = uuid.uuid4().hex
        return AlignedBundle(
            timestamp=anchor,
            text=text,
            frame=frame,
            audio=audio,
            session_id=session_id,
        )

    @staticmethod
    def _get_timestamp(items: List[Any]) -> Optional[float]:
        for it in items:
            ts = getattr(it, "timestamp", None) or getattr(it, "start_time", None)
            if ts is not None:
                return float(ts)
        return None

    def _pick(self, items: List[Any], anchor: float) -> Any:
        best: Any = None
        best_diff = float("inf")
        for it in items:
            ts = getattr(it, "timestamp", None) or getattr(it, "start_time", None)
            if ts is None:
                continue
            diff = abs(float(ts) - anchor)
            if diff <= self.time_window and diff < best_diff:
                best_diff = diff
                best = it
        return best

    def _pick_text(self, items: List[Any], anchor: float) -> Optional[str]:
        best: Any = None
        best_diff = float("inf")
        for it in items:
            if isinstance(it, str):
                return it
            ts = getattr(it, "timestamp", None)
            if ts is None:
                continue
            diff = abs(float(ts) - anchor)
            if diff <= self.time_window and diff < best_diff:
                best_diff = diff
                best = it
        if best is None:
            return None
        raw = getattr(best, "raw_text", None)
        return raw if raw is not None else str(best)


# ============================ T042 ============================

class PerceptionStructPackager(BaseModule):
    """T042 感知信息结构化封装

    动态调度模块。将对齐后的多模态报文统一封装为标准化 PerceptionReport，
    固定字段并支持序列化为 JSON。
    """

    name = "perception_struct_packager"
    version = "1.0.0"
    description = "封装标准化感知报文"

    def package(self, aligned_bundle: AlignedBundle) -> PerceptionReport:
        """封装对齐后的报文

        Args:
            aligned_bundle: AlignedBundle 对象

        Returns:
            PerceptionReport: 标准化感知报文
        """
        visual: Dict[str, Any] = {}
        if aligned_bundle.frame is not None:
            f = aligned_bundle.frame
            visual = {
                "width": getattr(f, "width", 0),
                "height": getattr(f, "height", 0),
                "channels": getattr(f, "channels", 0),
                "timestamp": getattr(f, "timestamp", 0.0),
                "has_data": getattr(f, "data", None) is not None,
            }
        audio: Dict[str, Any] = {}
        if aligned_bundle.audio is not None:
            a = aligned_bundle.audio
            samples = getattr(a, "samples", None)
            samples_count = 0
            if samples is not None:
                try:
                    samples_count = int(np.asarray(samples).shape[0])
                except Exception:
                    samples_count = 0
            audio = {
                "sample_rate": getattr(a, "sample_rate", 0),
                "start_time": getattr(a, "start_time", 0.0),
                "duration": getattr(a, "duration", 0.0),
                "samples_count": samples_count,
            }
        return PerceptionReport(
            session_id=aligned_bundle.session_id or uuid.uuid4().hex,
            timestamp=aligned_bundle.timestamp,
            text=aligned_bundle.text or "",
            visual=visual,
            audio=audio,
            metadata={
                "has_text": aligned_bundle.text is not None and bool(aligned_bundle.text),
                "has_frame": aligned_bundle.frame is not None,
                "has_audio": aligned_bundle.audio is not None,
            },
        )


# ============================ T043 ============================

class InvalidModalityFilter(BaseModule):
    """T043 无效模态过滤

    动态调度模块。过滤空白文本/黑屏画面/静音音频/重复数据。
    """

    name = "invalid_modality_filter"
    version = "1.0.0"
    description = "过滤无效模态（空白文本/黑屏/静音/重复）"

    def __init__(self,
                 text_min_len: int = 1,
                 black_pixel_ratio: float = 0.98,
                 silence_rms_threshold: float = 0.005,
                 duplicate_window: int = 10) -> None:
        super().__init__()
        self.text_min_len = text_min_len
        self.black_pixel_ratio = black_pixel_ratio
        self.silence_rms_threshold = silence_rms_threshold
        self.duplicate_window = duplicate_window
        # 最近报文指纹缓存，用于重复检测
        self._recent_signatures: List[str] = []

    def filter(self, report: PerceptionReport) -> bool:
        """判断单个报文是否有效

        Args:
            report: 待检测的感知报文

        Returns:
            bool: True 表示有效保留，False 表示应过滤掉
        """
        if report is None:
            return False
        text_ok = bool(report.text) and len(report.text.strip()) >= self.text_min_len
        visual_ok = bool(report.visual) and report.visual.get("has_data", False)
        audio_ok = bool(report.audio) and report.audio.get("samples_count", 0) > 0
        # 三种模态全空
        if not (text_ok or visual_ok or audio_ok):
            return False
        # 仅文本空白且无其他模态
        if report.text and self.is_blank_text(report.text) and not visual_ok and not audio_ok:
            return False
        # 仅视觉且黑屏
        if visual_ok and not text_ok and not audio_ok:
            if self._is_black_visual(report.visual):
                return False
        # 仅音频且静音
        if audio_ok and not text_ok and not visual_ok:
            if self._is_silent_visual(report.audio):
                return False
        # 重复检测
        sig = self._signature(report)
        if sig in self._recent_signatures:
            return False
        self._recent_signatures.append(sig)
        if len(self._recent_signatures) > self.duplicate_window:
            self._recent_signatures.pop(0)
        return True

    def filter_batch(self, reports: List[PerceptionReport]) -> List[PerceptionReport]:
        """批量过滤

        Args:
            reports: 报文列表

        Returns:
            List[PerceptionReport]: 通过过滤的报文列表
        """
        kept: List[PerceptionReport] = []
        for r in reports:
            if self.filter(r):
                kept.append(r)
        return kept

    def is_blank_text(self, t: str) -> bool:
        """判断文本是否空白

        Args:
            t: 文本

        Returns:
            bool: True 表示空白
        """
        if t is None:
            return True
        if not isinstance(t, str):
            t = str(t)
        return len(t.strip()) < self.text_min_len

    def is_black_frame(self, f: Any) -> bool:
        """判断帧是否为黑屏

        Args:
            f: Frame 对象或 numpy 数组

        Returns:
            bool: True 表示黑屏
        """
        img = getattr(f, "data", None)
        if img is None:
            img = f
        if img is None:
            return True
        try:
            arr = np.asarray(img)
            if arr.size == 0:
                return True
            dark_ratio = float(np.mean(arr < 15))
            return dark_ratio >= self.black_pixel_ratio
        except Exception:
            return False

    def is_silent_audio(self, a: Any) -> bool:
        """判断音频是否静音

        Args:
            a: AudioChunk 对象或 numpy 数组

        Returns:
            bool: True 表示静音
        """
        samples = getattr(a, "samples", None)
        if samples is None:
            samples = a
        if samples is None:
            return True
        try:
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return True
            rms = float(np.sqrt(np.mean(arr ** 2)))
            return rms < self.silence_rms_threshold
        except Exception:
            return True

    def _is_black_visual(self, visual: Dict[str, Any]) -> bool:
        # 没有 data 字段时无法判断像素，认为非黑屏
        return not visual.get("has_data", False)

    def _is_silent_visual(self, audio: Dict[str, Any]) -> bool:
        return audio.get("samples_count", 0) == 0

    def _signature(self, report: PerceptionReport) -> str:
        """生成报文指纹用于重复检测"""
        text_sig = (report.text or "").strip()[:200]
        visual_sig = f"{report.visual.get('width', 0)}x{report.visual.get('height', 0)}"
        audio_sig = f"{report.audio.get('samples_count', 0)}@{report.audio.get('sample_rate', 0)}"
        return f"{text_sig}|{visual_sig}|{audio_sig}"


__all__ = [
    "AlignedBundle",
    "PerceptionReport",
    "MultimodalTimeAligner",
    "PerceptionStructPackager",
    "InvalidModalityFilter",
]
