
import time

def get_time() -> float:
    """
    Returns the current time in seconds since the epoch (Unix timestamp).

    Returns:
        float: The current time in seconds since the epoch.
    """
    return time.time()

def timestamp_to_minutes(ts):
    return ts / 60000

def timestamp_to_seconds(ts):
    return ts / 1000
