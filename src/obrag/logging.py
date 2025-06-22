"""
Setting up timestamped logging for each run for debugging purposes.
"""

import logging
import os
from datetime import datetime

def start_logger(name: str="OBRAG", log_dir: str="logs") -> logging.Logger:
    """
    Start a logger with a timestamped filename in the specified directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger