def log_message(message):
    print(f"[LOG] {message}")

def validate_input(data, expected_type):
    if not isinstance(data, expected_type):
        raise ValueError(
            f"Invalid input: Expected {expected_type}, got {type(data)}"
        )
