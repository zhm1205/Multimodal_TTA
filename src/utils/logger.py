"""
Logging management module
"""

import os
import sys
import logging
from typing import Optional
from datetime import datetime
import zoneinfo  # Python 3.9+ 


def setup_logger(
    name: str = 'main',
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger

    Args:
        name: Logger name
        log_file: Path to the log file (optional)
        level: Logging level
        format_str: Log format string

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set format
    if format_str is None:
        # MODIFIED: Added %(filename)s and %(lineno)d to the format string
        # %(filename)s: The filename where the log was made
        # %(lineno)d: The line number where the log was made
        format_str = '[%(asctime)s] %(name)s - %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    
    formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str = 'main') -> logging.Logger:
    """
    Get logger

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)



class LoggerWriter:
    """A class to redirect writes to the logger"""
    
    def __init__(self, logger: logging.Logger, level: str = 'INFO'):
        self.logger = logger
        self.level = getattr(logging, level.upper())
        
    def write(self, message: str):
        if message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        # This flush method is needed for compatibility with sys.stdout
        pass


def redirect_stdout_to_logger(logger: logging.Logger):
    """Redirect standard output to the logger"""
    sys.stdout = LoggerWriter(logger, 'INFO')


def redirect_stderr_to_logger(logger: logging.Logger):
    """Redirect standard error to the logger"""
    sys.stderr = LoggerWriter(logger, 'ERROR')