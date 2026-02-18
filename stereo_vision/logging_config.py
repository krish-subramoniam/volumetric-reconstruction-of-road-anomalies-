"""
Logging configuration for the stereo vision pipeline.

This module provides centralized logging configuration with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output
- Structured logging with context information
- Performance timing utilities
- Error tracking and reporting

Requirements: Error handling across all modules
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback
import functools
import time


class PipelineLogger:
    """
    Centralized logger for the stereo vision pipeline.
    
    Provides structured logging with context information and performance tracking.
    """
    
    def __init__(self, name: str = "stereo_vision", log_dir: Optional[str] = None):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files. If None, logs only to console.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log directory specified)
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_path / f"stereo_vision_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional exception info and context."""
        self._log_with_context(logging.ERROR, message, kwargs, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message with optional exception info and context."""
        self._log_with_context(logging.CRITICAL, message, kwargs, exc_info=exc_info)
    
    def _log_with_context(self, level: int, message: str, context: Dict[str, Any], 
                         exc_info: bool = False) -> None:
        """Log message with structured context information."""
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message, exc_info=exc_info)
    
    def log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an exception with full traceback and context.
        
        Args:
            exception: The exception to log
            context: Optional context information
        """
        exc_type = type(exception).__name__
        exc_message = str(exception)
        exc_traceback = ''.join(traceback.format_tb(exception.__traceback__))
        
        error_msg = f"{exc_type}: {exc_message}\nTraceback:\n{exc_traceback}"
        
        if context:
            self.error(error_msg, **context)
        else:
            self.error(error_msg)


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, logger: PipelineLogger, operation_name: str):
        """
        Initialize performance timer.
        
        Args:
            logger: PipelineLogger instance
            operation_name: Name of the operation being timed
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed: {self.operation_name}",
                duration_seconds=f"{duration:.3f}"
            )
        else:
            self.logger.error(
                f"Failed: {self.operation_name}",
                duration_seconds=f"{duration:.3f}",
                error=str(exc_val)
            )
        
        return False  # Don't suppress exceptions
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def log_function_call(logger: PipelineLogger):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: PipelineLogger instance
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log function call
            logger.debug(f"Calling {func_name}", args_count=len(args), kwargs_count=len(kwargs))
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Completed {func_name}", duration_seconds=f"{duration:.3f}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Error in {func_name}: {str(e)}",
                    duration_seconds=f"{duration:.3f}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Global logger instance
_global_logger: Optional[PipelineLogger] = None


def get_logger(name: str = "stereo_vision", log_dir: Optional[str] = None) -> PipelineLogger:
    """
    Get or create a global logger instance.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        PipelineLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PipelineLogger(name, log_dir)
    
    return _global_logger


def configure_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """
    Configure global logging settings.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    """
    global _global_logger
    
    _global_logger = PipelineLogger("stereo_vision", log_dir)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    _global_logger.logger.setLevel(level)
