import numpy as np
from typing import Any, List, Union


def to_native(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(item) for item in obj]
    return obj


def cosine_similarity(vec1: Union[np.ndarray, List[float]], 
                      vec2: Union[np.ndarray, List[float]]) -> float:
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    min_len = min(len(vec1), len(vec2))
    if min_len == 0:
        return 0.0
    
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    return dot_product / norm_product if norm_product != 0 else 0.0


def normalize_vector(vec: Union[np.ndarray, List[float]]) -> np.ndarray:
    vec = np.array(vec).flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec
