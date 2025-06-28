"""Utility functions for logging and input validation."""


def log_message(message):
    """Print a message with a log prefix."""
    print(f"[LOG] {message}")

def validate_input(data, expected_type):
    """Validate that ``data`` is an instance of ``expected_type``."""
    if not isinstance(data, expected_type):
        raise ValueError(
            f"Invalid input: Expected {expected_type}, got {type(data)}"
        )
