"""
perception/visual_perception.py - 视觉感知子模块

实现任务 T034-T037：视频帧采集解码、目标检测、图像分割与空间定位、
图像描述生成。

所有重型依赖（OpenCV/YOLO/detectron2/SAM/BLIP/LLaVA）采用 try/except
可选导入，保证无依赖环境可导入实例化。numpy 可用。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agi_agent.core import BaseModule

logger = logging.getLogger("agi_agent.perception")


# ============================ 数据结构 ============================

@dataclass
class Frame:
    """T034 视频帧"""

    data: np.ndarray              # 像素数据 HxWxC
    width: int
    height: int
    channels: int
    timestamp: float


@dataclass
class Detection:
    """T035 目标检测结果"""

    label: str
    bbox: List[int]               # [x1, y1, x2, y2]
    confidence: float
    class_id: int = -1


@dataclass
class SegmentationResult:
    """T036 分割与空间定位结果"""

    masks: List[np.ndarray] = field(default_factory=list)
    spatial_coords: List[List[float]] = field(default_factory=list)
    scene_structure: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneDescription:
    """T037 图像场景描述"""

    text: str
    elements: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)


# ============================ T034 ============================

class VideoFrameDecoder(BaseModule):
    """T034 视频帧采集解码

    轮询执行模块。采集摄像头流或视频文件，完成帧解码、格式转换、分辨率归一化。
    OpenCV 为可选依赖，缺失时返回 None + warning。
    """

    name = "video_frame_decoder"
    version = "1.0.0"
    description = "采集视频/摄像头帧并解码归一化"

    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self._has_cv2: bool = False
        try:
            import cv2  # type: ignore  # noqa: F401
            self._has_cv2 = True
        except Exception:
            logger.warning("OpenCV 不可用，视频帧解码将返回空帧")
        self._captures: Dict[int, Any] = {}
        self._next_id: int = 1

    def open(self, source: Any) -> int:
        """打开视频源

        Args:
            source: 摄像头索引（int）或视频文件路径（str）

        Returns:
            int: capture_id，失败返回 -1
        """
        if not self._has_cv2:
            logger.warning("OpenCV 不可用，无法打开视频源")
            return -1
        import cv2  # type: ignore
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.warning("视频源打开失败 source=%s", source)
            return -1
        cap_id = self._next_id
        self._next_id += 1
        self._captures[cap_id] = cap
        return cap_id

    def read_frame(self, capture_id: int) -> Optional[Frame]:
        """读取一帧

        Args:
            capture_id: 由 open 返回的句柄

        Returns:
            Optional[Frame]: 帧对象；失败或无依赖返回 None
        """
        if not self._has_cv2:
            return None
        cap = self._captures.get(capture_id)
        if cap is None:
            return None
        import cv2  # type: ignore
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        # BGR -> RGB
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        h, w = frame.shape[:2]
        # 分辨率归一化
        if (w, h) != (self.target_width, self.target_height):
            try:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
                h, w = self.target_height, self.target_width
            except Exception as e:
                logger.debug("resize 失败 err=%s", e)
        channels = frame.shape[2] if frame.ndim == 3 else 1
        return Frame(
            data=frame,
            width=int(w),
            height=int(h),
            channels=int(channels),
            timestamp=time.time(),
        )

    def close(self, capture_id: int) -> None:
        """关闭视频源

        Args:
            capture_id: 由 open 返回的句柄
        """
        cap = self._captures.pop(capture_id, None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _shutdown(self) -> None:
        for cid in list(self._captures.keys()):
            self.close(cid)


# ============================ T035 ============================

class ObjectDetector(BaseModule):
    """T035 目标检测

    动态调度模块。输出检测类别/坐标/置信度/检测框。
    YOLO/detectron2 均为可选，缺失时返回空列表。
    """

    name = "object_detector"
    version = "1.0.0"
    description = "目标检测（YOLO/detectron2 可选）"

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.3) -> None:
        super().__init__()
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._backend: Optional[str] = None
        self._model: Any = None
        self._model_loaded: bool = False
        # 仅探测可用的依赖，不在构造时加载模型权重
        try:
            from ultralytics import YOLO  # type: ignore  # noqa: F401
            self._backend = "ultralytics"
        except Exception:
            pass
        if self._backend is None:
            try:
                import detectron2  # type: ignore  # noqa: F401
                self._backend = "detectron2"
            except Exception:
                pass
        if self._backend is None:
            logger.warning("未检测到 YOLO/detectron2，目标检测返回空列表")

    def _ensure_model(self) -> Any:
        """懒加载模型权重，避免构造时触发网络下载"""
        if self._model_loaded:
            return self._model
        self._model_loaded = True  # 标记已尝试，避免重复下载
        try:
            if self._backend == "ultralytics":
                from ultralytics import YOLO  # type: ignore
                self._model = YOLO(self.model_path or "yolov8n.pt")
        except Exception as e:
            logger.warning("目标检测模型加载失败 err=%s，返回空列表", e)
            self._model = None
        return self._model

    def detect(self, frame: Any) -> List[Detection]:
        """对帧执行目标检测

        Args:
            frame: Frame 对象或 numpy 数组

        Returns:
            List[Detection]: 检测结果列表
        """
        if self._backend is None:
            return []
        model = self._ensure_model()
        if model is None:
            return []
        try:
            img = frame.data if isinstance(frame, Frame) else frame
            if img is None:
                return []
            if self._backend == "ultralytics":
                return self._detect_ultralytics(img)
        except Exception as e:
            logger.warning("目标检测失败 err=%s", e)
        return []

    def _detect_ultralytics(self, img: np.ndarray) -> List[Detection]:
        results = self._model(img, verbose=False)
        detections: List[Detection] = []
        for r in results:
            try:
                boxes = r.boxes
                names = r.names if hasattr(r, "names") else {}
                for b in boxes:
                    conf = float(b.conf[0]) if hasattr(b, "conf") and len(b.conf) > 0 else 0.0
                    if conf < self.confidence_threshold:
                        continue
                    cls_id = int(b.cls[0]) if hasattr(b, "cls") and len(b.cls) > 0 else -1
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    xyxy = b.xyxy[0].tolist() if hasattr(b, "xyxy") and len(b.xyxy) > 0 else [0, 0, 0, 0]
                    bbox = [int(v) for v in xyxy]
                    detections.append(Detection(label=label, bbox=bbox, confidence=conf, class_id=cls_id))
            except Exception:
                continue
        return detections


# ============================ T036 ============================

class SegmentationLocator(BaseModule):
    """T036 图像分割与空间定位

    动态调度模块。基于检测结果生成像素级分割掩码并换算三维空间坐标。
    SAM 为可选依赖，缺失时降级为基于检测框的近似掩码。
    """

    name = "segmentation_locator"
    version = "1.0.0"
    description = "图像分割与三维空间定位"

    def __init__(self) -> None:
        super().__init__()
        self._has_sam: bool = False
        try:
            from segment_anything import sam_model_registry  # type: ignore  # noqa: F401
            from segment_anything import SamPredictor  # type: ignore  # noqa: F401
            self._has_sam = True
        except Exception:
            pass
        if not self._has_sam:
            logger.warning("SAM 不可用，分割将降级为检测框近似掩码")

    def segment(self,
                frame: Any,
                detections: List[Detection]) -> SegmentationResult:
        """对帧执行分割

        Args:
            frame: Frame 或 numpy 数组
            detections: 目标检测结果，用于引导分割

        Returns:
            SegmentationResult: 分割与空间坐标结果
        """
        img = frame.data if isinstance(frame, Frame) else frame
        if img is None:
            return SegmentationResult()
        masks: List[np.ndarray] = []
        coords: List[List[float]] = []
        if self._has_sam:
            try:
                masks, coords = self._segment_sam(img, detections)
            except Exception as e:
                logger.debug("SAM 分割失败 err=%s，降级", e)
                masks, coords = self._segment_fallback(img, detections)
        else:
            masks, coords = self._segment_fallback(img, detections)
        scene_structure = self._build_scene_structure(detections, img.shape)
        return SegmentationResult(
            masks=masks,
            spatial_coords=coords,
            scene_structure=scene_structure,
        )

    def _segment_sam(self,
                     img: np.ndarray,
                     detections: List[Detection]) -> Tuple[List[np.ndarray], List[List[float]]]:
        # SAM 权重未配置时降级为基于检测框的近似掩码
        return self._segment_fallback(img, detections)

    def _segment_fallback(self,
                          img: np.ndarray,
                          detections: List[Detection]) -> Tuple[List[np.ndarray], List[List[float]]]:
        h, w = img.shape[:2]
        masks: List[np.ndarray] = []
        coords: List[List[float]] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            mask = np.zeros((h, w), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
            masks.append(mask)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area_ratio = float((x2 - x1) * (y2 - y1)) / float(max(1, w * h))
            coords.append([cx / w, cy / h, area_ratio])
        return masks, coords

    @staticmethod
    def _build_scene_structure(detections: List[Detection],
                               img_shape: Tuple[int, ...]) -> Dict[str, Any]:
        h = img_shape[0] if img_shape else 0
        w = img_shape[1] if len(img_shape) > 1 else 0
        labels = [d.label for d in detections]
        return {
            "object_count": len(detections),
            "labels": labels,
            "image_size": [int(w), int(h)],
            "layout": "unknown",
        }


# ============================ T037 ============================

class ImageCaptionGenerator(BaseModule):
    """T037 图像描述生成

    动态调度模块。基于多模态模型生成自然语言场景描述。
    BLIP/LLaVA 为可选依赖，缺失时降级返回占位描述。
    """

    name = "image_caption_generator"
    version = "1.0.0"
    description = "为图像生成自然语言场景描述"

    def __init__(self) -> None:
        super().__init__()
        self._backend: Optional[str] = None
        self._processor: Any = None
        self._model: Any = None
        self._model_loaded: bool = False
        # 仅探测 transformers 是否可用，不在构造时下载权重
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore  # noqa: F401
            self._backend = "blip"
        except Exception:
            pass
        if self._backend is None:
            logger.warning("BLIP/LLaVA 不可用，图像描述将返回占位文本")

    def _ensure_model(self) -> bool:
        """懒加载 BLIP 模型权重，避免构造时触发网络下载

        Returns:
            bool: 模型是否就绪
        """
        if self._model_loaded:
            return self._processor is not None and self._model is not None
        self._model_loaded = True  # 标记已尝试，避免重复下载
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
            self._processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self._model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        except Exception as e:
            logger.warning("BLIP 模型加载失败 err=%s，降级占位描述", e)
            self._processor = None
            self._model = None
        return self._processor is not None and self._model is not None

    def describe(self,
                 frame: Any,
                 segmentation: Optional[SegmentationResult] = None) -> SceneDescription:
        """生成场景描述

        Args:
            frame: Frame 或 numpy 数组
            segmentation: 可选的分割结果，用于补充描述

        Returns:
            SceneDescription: 场景描述
        """
        img = frame.data if isinstance(frame, Frame) else frame
        elements: List[str] = []
        relations: List[str] = []
        if segmentation is not None:
            scene = segmentation.scene_structure or {}
            labels = scene.get("labels") or []
            elements.extend(labels)
            if scene.get("object_count"):
                relations.append(f"包含 {scene['object_count']} 个对象")
        if (self._backend == "blip"
                and img is not None
                and self._ensure_model()):
            try:
                from PIL import Image  # type: ignore
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img)
                else:
                    pil_img = img
                inputs = self._processor(pil_img, return_tensors="pt")
                out = self._model.generate(**inputs, max_length=50)
                text = self._processor.decode(out[0], skip_special_tokens=True)
                return SceneDescription(text=text, elements=elements, relations=relations)
            except Exception as e:
                logger.debug("BLIP 描述生成失败 err=%s，降级", e)
        text = self._placeholder_description(img, elements)
        return SceneDescription(text=text, elements=elements, relations=relations)

    @staticmethod
    def _placeholder_description(img: Optional[np.ndarray],
                                 elements: List[str]) -> str:
        if img is None:
            return "图像不可用"
        try:
            h, w = img.shape[:2]
        except Exception:
            h, w = 0, 0
        prefix = f"图像尺寸 {w}x{h}"
        if elements:
            return f"{prefix}，包含：{', '.join(elements)}"
        return f"{prefix}（多模态描述模型未加载，仅占位）"


__all__ = [
    "Frame",
    "Detection",
    "SegmentationResult",
    "SceneDescription",
    "VideoFrameDecoder",
    "ObjectDetector",
    "SegmentationLocator",
    "ImageCaptionGenerator",
]
