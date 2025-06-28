import logging

logger = logging.getLogger(__name__)


def log_message(message):
    """Log a simple message using the module logger."""
    logger.info(message)

def validate_input(data, expected_type):
    if not isinstance(data, expected_type):
        raise ValueError(f"Invalid input: Expected {expected_type}, got {type(data)}")
