"""
Logging configuration utilities
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str, 
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Parameters:
    -----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : int
        Logging level
    format_string : str, optional
        Custom format string

    Returns:
    --------
    logger : Logger
        Configured logger
    """
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max, 5 backups
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)

def set_log_level(logger: logging.Logger, level: int):
    """Set log level for logger and all its handlers"""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
