import logging
import logging.config
import os
from pythonjsonlogger import jsonlogger


def setup_logging():
    """
    Sets up logging using a dictionary configuration.
    This configuration sets up console and rotating file handlers
    with different formatters.
    """
    # Read configuration from environment variables with defaults
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR = os.getenv("LOG_DIR", "../logs")

    # Create the log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR, "app.log")

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            # JSON formatter for file logs (machine-readable)
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
            # Standard formatter for console logs (human-readable)
            "standard": {
                "format": "[%(asctime)s] - %(levelname)s - %(name)s - %(message)s",
            },
        },
        "handlers": {
            # Console handler to print logs to the terminal
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": LOG_LEVEL,
            },
            # Rotating file handler to write logs to a file
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOG_FILE,
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 10, # Keeps 5 old log files
                "level": LOG_LEVEL,
            },
        },
        "root": {
            # Send logs to both console and file handlers
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
        },
    }

    # Apply the configuration
    logging.config.dictConfig(LOGGING_CONFIG)



