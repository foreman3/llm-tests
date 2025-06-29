"""Utility functions for logging and input validation."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def log_message(message: str) -> None:
    """Log a message with an ``INFO`` level."""
    logger.info(message)

def validate_input(data, expected_type):
    """Validate that ``data`` is an instance of ``expected_type``."""
    if not isinstance(data, expected_type):
        raise ValueError(
            f"Invalid input: Expected {expected_type}, got {type(data)}"
        )
