import torch
from agi_agent.core import get_adaptive_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_adaptive_config = get_adaptive_config()

FREE_ENERGY_THRESHOLD = _adaptive_config.get("free_energy_threshold", 0.3)
EVOLVE_TRIGGER_STEP = _adaptive_config.get("evolve_trigger_step", 200)
MAX_INFERENCE_STEP = _adaptive_config.get("max_inference_step", 5)
NOVELTY_THRESHOLD = _adaptive_config.get("novelty_threshold", 0.5)

MEMORY_BUFFER_SIZE = _adaptive_config.get("memory_buffer_size", 200)
KNOWLEDGE_MAX_SIZE = _adaptive_config.get("knowledge_max_size", 1000)
HISTORY_MAX_LEN = _adaptive_config.get("history_max_len", 50)

LEARNING_RATE_POOL = _adaptive_config.get("learning_rate_pool", [1e-4, 5e-4, 1e-3, 2e-3])
INITIAL_LEARNING_RATE = _adaptive_config.get("initial_learning_rate", 1e-3)

MAX_HIDDEN_DIM = _adaptive_config.get("max_hidden_dim", 256)
MIN_HIDDEN_DIM = _adaptive_config.get("min_hidden_dim", 16)
GROWTH_STEP = _adaptive_config.get("growth_step", 8)
PRUNE_STEP = _adaptive_config.get("prune_step", 4)

SAFETY_MAX_ENERGY = 10.0
SAFETY_MAX_MEMORY_GB = _adaptive_config.get("safety_max_memory_gb", 4.0)
SAFETY_MAX_GPU_UTIL = _adaptive_config.get("safety_max_gpu_util", 0.95)
SAFETY_MAX_LATENCY_MS = _adaptive_config.get("safety_max_latency_ms", 1000)

LOG_INTERVAL = _adaptive_config.get("log_interval", 20)
SAVE_INTERVAL = _adaptive_config.get("save_interval", 1000)
EVAL_INTERVAL = _adaptive_config.get("eval_interval", 500)

STORAGE_DIR = "./data"
LOG_DIR = "./logs"
MODEL_DIR = "./models"

SEED = 42