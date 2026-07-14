import os
import logging
from datetime import datetime
from ..config.settings import LOG_DIR


def setup_logger(name: str = "agi_agent", log_level: str = "INFO") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_metrics(logger: logging.Logger, metrics: dict, step: int) -> None:
    metrics_str = " | ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
    logger.info(f"Step {step} | {metrics_str}")


def log_event(logger: logging.Logger, event_type: str, message: str) -> None:
    logger.info(f"[{event_type}] {message}")


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    logger.error(f"[{context}] Error: {str(error)}", exc_info=True)