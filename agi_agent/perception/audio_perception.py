"""
perception/audio_perception.py - 音频感知子模块

实现任务 T038-T040：音频流采集降噪、ASR 语音转文字、声纹/情绪识别。

所有重型依赖（PyAudio/librosa/noisereduce/whisper）采用 try/except 可选导入，
保证无依赖环境可导入实例化。numpy 可用。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.perception")


# ============================ 数据结构 ============================

@dataclass
class AudioChunk:
    """T038 音频片段"""

    samples: np.ndarray        # 1D float32 数组
    sample_rate: int
    start_time: float
    duration: float


@dataclass
class ASRSegment:
    """T039 ASR 转写片段"""

    text: str
    start: float
    end: float
    confidence: float = 0.0


@dataclass
class SpeakerProfile:
    """T040 说话人画像"""

    speaker_id: str
    emotion: str
    confidence: float = 0.0


# ============================ T038 ============================

class AudioCaptureDenoiser(BaseModule):
    """T038 音频流采集降噪

    轮询执行模块。采集麦克风原始音频，降噪过滤环境噪音并截取有效人声片段。
    PyAudio/librosa/noisereduce 均为可选，缺失时返回空音频。
    """

    name = "audio_capture_denoiser"
    version = "1.0.0"
    description = "采集麦克风音频并降噪"

    def __init__(self,
                 sample_rate: int = 16000,
                 chunk_duration: float = 0.5,
                 silence_threshold: float = 0.01) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self._has_pyaudio: bool = False
        try:
            import pyaudio  # type: ignore  # noqa: F401
            self._has_pyaudio = True
        except Exception:
            logger.warning("PyAudio 不可用，音频采集将返回空片段")
        self._has_noisereduce: bool = False
        try:
            import noisereduce  # type: ignore  # noqa: F401
            self._has_noisereduce = True
        except Exception:
            pass
        self._captures: Dict[int, Any] = {}
        self._capture_start: Dict[int, float] = {}
        self._next_id: int = 1

    def start_capture(self) -> int:
        """开启麦克风采集

        Returns:
            int: capture_id，失败返回 -1
        """
        if not self._has_pyaudio:
            logger.warning("PyAudio 不可用，无法启动音频采集")
            return -1
        try:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=int(self.sample_rate * self.chunk_duration),
            )
            cap_id = self._next_id
            self._next_id += 1
            self._captures[cap_id] = {"pa": pa, "stream": stream}
            self._capture_start[cap_id] = time.time()
            return cap_id
        except Exception as e:
            logger.warning("启动音频采集失败 err=%s", e)
            return -1

    def read_chunk(self, capture_id: int) -> AudioChunk:
        """读取一段音频并降噪

        Args:
            capture_id: 由 start_capture 返回的句柄

        Returns:
            AudioChunk: 音频片段；不可用时返回空 samples
        """
        empty = AudioChunk(
            samples=np.zeros(
                int(self.sample_rate * self.chunk_duration), dtype=np.float32
            ),
            sample_rate=self.sample_rate,
            start_time=time.time(),
            duration=self.chunk_duration,
        )
        cap = self._captures.get(capture_id)
        if cap is None:
            return empty
        stream = cap["stream"]
        try:
            frames = stream.read(
                int(self.sample_rate * self.chunk_duration),
                exception_on_overflow=False,
            )
            samples = np.frombuffer(frames, dtype=np.float32)
        except Exception as e:
            logger.debug("读取音频失败 err=%s", e)
            return empty
        # 降噪
        if self._has_noisereduce and samples.size > 0:
            try:
                import noisereduce as nr  # type: ignore
                samples = nr.reduce_noise(y=samples, sr=self.sample_rate)
            except Exception as e:
                logger.debug("降噪失败 err=%s", e)
        start_time = self._capture_start.get(capture_id, time.time())
        self._capture_start[capture_id] = start_time + self.chunk_duration
        return AudioChunk(
            samples=samples,
            sample_rate=self.sample_rate,
            start_time=start_time,
            duration=self.chunk_duration,
        )

    def stop_capture(self, capture_id: int) -> None:
        """停止采集

        Args:
            capture_id: 由 start_capture 返回的句柄
        """
        cap = self._captures.pop(capture_id, None)
        self._capture_start.pop(capture_id, None)
        if cap is None:
            return
        try:
            cap["stream"].stop_stream()
            cap["stream"].close()
        except Exception:
            pass
        try:
            cap["pa"].terminate()
        except Exception:
            pass

    def _shutdown(self) -> None:
        for cid in list(self._captures.keys()):
            self.stop_capture(cid)


# ============================ T039 ============================

class ASRTranscriber(BaseModule):
    """T039 ASR 语音转文字

    动态调度模块。将连续人声转为带时间戳的分段文本。
    Whisper 为可选依赖，缺失时返回空列表。
    """

    name = "asr_transcriber"
    version = "1.0.0"
    description = "ASR 语音转文字（Whisper 可选）"

    def __init__(self, model_name: str = "base") -> None:
        super().__init__()
        self.model_name = model_name
        self._has_whisper: bool = False
        self._model: Any = None
        self._model_loaded: bool = False
        # 仅探测 whisper 是否可用，不在构造时下载权重
        try:
            import whisper  # type: ignore  # noqa: F401
            self._has_whisper = True
        except Exception:
            logger.warning("Whisper 不可用，ASR 将返回空结果")

    def _ensure_model(self) -> Any:
        """懒加载 Whisper 模型，避免构造时触发网络下载"""
        if self._model_loaded:
            return self._model
        self._model_loaded = True
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(self.model_name)
        except Exception as e:
            logger.warning("Whisper 模型加载失败 err=%s，返回空结果", e)
            self._model = None
        return self._model

    def transcribe(self, audio_chunk: AudioChunk) -> List[ASRSegment]:
        """转写音频片段

        Args:
            audio_chunk: 输入音频片段

        Returns:
            List[ASRSegment]: 转写分段
        """
        if not self._has_whisper:
            return []
        model = self._ensure_model()
        if model is None:
            return []
        if (audio_chunk is None
                or audio_chunk.samples is None
                or audio_chunk.samples.size == 0):
            return []
        try:
            result = model.transcribe(audio_chunk.samples, language="zh")
            segments: List[ASRSegment] = []
            for seg in result.get("segments", []):
                segments.append(ASRSegment(
                    text=seg.get("text", "").strip(),
                    start=float(seg.get("start", 0.0)) + audio_chunk.start_time,
                    end=float(seg.get("end", 0.0)) + audio_chunk.start_time,
                    confidence=float(seg.get("avg_logprob", 0.0)),
                ))
            return segments
        except Exception as e:
            logger.warning("ASR 转写失败 err=%s", e)
            return []


# ============================ T040 ============================

class VoiceprintEmotionRecognizer(BaseModule):
    """T040 声纹与情绪识别

    动态调度模块。提取声纹特征以区分说话人，并识别情绪。
    重型依赖（speechbrain/librosa/scikit-learn）可选，缺失时返回 unknown。
    """

    name = "voiceprint_emotion_recognizer"
    version = "1.0.0"
    description = "声纹特征提取与情绪识别"

    def __init__(self) -> None:
        super().__init__()
        self._has_librosa: bool = False
        try:
            import librosa  # type: ignore  # noqa: F401
            self._has_librosa = True
        except Exception:
            pass
        self._has_sklearn: bool = False
        try:
            import sklearn  # type: ignore  # noqa: F401
            self._has_sklearn = True
        except Exception:
            pass
        # 简易说话人缓存
        self._speaker_clusters: Dict[str, np.ndarray] = {}
        if not self._has_librosa:
            logger.warning("librosa 不可用，声纹/情绪识别将返回 unknown")

    def recognize(self,
                  audio_chunk: AudioChunk,
                  asr_text: str = "") -> SpeakerProfile:
        """识别说话人与情绪

        Args:
            audio_chunk: 输入音频片段
            asr_text: 对应的 ASR 文本（可选辅助）

        Returns:
            SpeakerProfile: 说话人画像，无法识别时返回 unknown
        """
        if (audio_chunk is None
                or audio_chunk.samples is None
                or audio_chunk.samples.size == 0):
            return SpeakerProfile(speaker_id="unknown", emotion="unknown", confidence=0.0)
        if not self._has_librosa:
            return SpeakerProfile(speaker_id="unknown", emotion="unknown", confidence=0.0)
        try:
            import librosa  # type: ignore
            samples = np.asarray(audio_chunk.samples, dtype=np.float32)
            mfcc = librosa.feature.mfcc(
                y=samples, sr=audio_chunk.sample_rate, n_mfcc=13
            )
            feat = mfcc.mean(axis=1)
            speaker_id = self._identify_speaker(feat)
            emotion = self._detect_emotion(samples, audio_chunk.sample_rate, asr_text)
            confidence = float(1.0 / (1.0 + float(np.var(feat))))
            return SpeakerProfile(
                speaker_id=speaker_id,
                emotion=emotion,
                confidence=confidence,
            )
        except Exception as e:
            logger.warning("声纹/情绪识别失败 err=%s", e)
            return SpeakerProfile(speaker_id="unknown", emotion="unknown", confidence=0.0)

    def _identify_speaker(self, feat: np.ndarray) -> str:
        """根据特征进行简单说话人匹配/注册"""
        if not self._speaker_clusters:
            sid = f"speaker_{len(self._speaker_clusters) + 1}"
            self._speaker_clusters[sid] = feat
            return sid
        best_sid = "unknown"
        best_sim = -1.0
        for sid, ref in self._speaker_clusters.items():
            denom = (float(np.linalg.norm(feat)) * float(np.linalg.norm(ref))) or 1.0
            sim = float(np.dot(feat, ref) / denom)
            if sim > best_sim:
                best_sim = sim
                best_sid = sid
        # 高于阈值则匹配，否则注册新说话人
        if best_sim > 0.9:
            return best_sid
        sid = f"speaker_{len(self._speaker_clusters) + 1}"
        self._speaker_clusters[sid] = feat
        return sid

    @staticmethod
    def _detect_emotion(samples: np.ndarray, sr: int, text: str) -> str:
        """简易情绪判别：基于能量+语速（文本长度/时长）"""
        try:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            duration = len(samples) / float(sr) if sr else 0.0
            text_len = len(text)
            if rms > 0.05 and (duration == 0 or text_len / duration > 5.0):
                return "excited"
            if rms < 0.005:
                return "calm"
            return "neutral"
        except Exception:
            return "unknown"


__all__ = [
    "AudioChunk",
    "ASRSegment",
    "SpeakerProfile",
    "AudioCaptureDenoiser",
    "ASRTranscriber",
    "VoiceprintEmotionRecognizer",
]
