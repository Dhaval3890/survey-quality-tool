import math
from typing import Any, Dict, List


def to_json_safe(value: Any) -> Any:
    """
    Recursively convert NaN / inf floats into None so standard JSON can serialize.
    Works for nested dicts/lists/tuples.
    """
    # floats: handle NaN / inf
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    # ints are already safe
    if isinstance(value, int):
        return int(value)

    # dict: recurse
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}

    # list / tuple: recurse
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    # everything else (str, bool, None, etc.)
    return value
