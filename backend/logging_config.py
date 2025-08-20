import logging
import os
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration for the RAG system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs only to console.
    """
    # Create logs directory if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging format
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    # Configure handlers
    handlers = [logging.StreamHandler()]  # Always log to console

    if log_file:
        # Add file handler if log file specified
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing logging configuration
    )

    # Configure specific loggers for our modules
    loggers = [
        "rag_system",
        "document_processor",
        "vector_store",
        "search_tools",
        "ai_generator",
        "session_manager",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

    # Suppress overly verbose third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logging.info(
        f"Logging configured - Level: {log_level}, File: {log_file or 'Console only'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
