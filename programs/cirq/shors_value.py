from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .shors import find_factor

def run_simulation(config: Dict[str, Any]) -> int:
    n = int(config.get("N", 21))
    if n < 2:
        raise ValueError("N must be >= 2")
    max_attempts = int(config.get("max_attempts", 30))
    retries = int(config.get("retries", 3))

    for _ in range(retries):
        factor = find_factor(n=n, max_attempts=max_attempts)
        if factor is not None:
            return np.array(factor)

    raise ValueError(f"No non-trivial factor found for N={n}")
