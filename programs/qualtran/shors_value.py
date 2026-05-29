from __future__ import annotations
from typing import Any, Dict
import numpy as np
from . import shors
import fractions


def process_measurement(measurement: int, x: int, n: int, t: int) -> int | None:
    # TODO: eigenphase := measurement / 2**n_exp

    # Run the continued fractions algorithm to determine f = s / r.
    f = fractions.Fraction.from_float(measurement / 2**t).limit_denominator(n)

    # If the numerator is zero, the order finder failed.
    if f.numerator == 0:
        return None

    # Else, return the denominator if it is valid.
    r = f.denominator
    if x**r % n != 1:
        return None
    return r


def measure(state: np.ndarray) -> int:
    probs = np.abs(state)**2
    return np.random.choice(len(state), p=probs)

def quantum_order_finder(config: Dict) -> int | None:
    state = shors.run_simulation(config)
    measurement = measure(state)

    # Return the processed measurement result.
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    return process_measurement(measurement, a, N, t)


def run_simulation(config: Dict[str, Any]) -> int:
    n = int(config.get("N", 21))
    if n < 2:
        raise ValueError("N must be >= 2")

    max_attempts = int(config.get("max_attempts", 30))
    retries = int(config.get("retries", 3))

    for _ in range(retries):
        factor = shors.find_factor(n, quantum_order_finder, config, max_attempts)
        if factor is not None:
            return np.array(factor)

    raise ValueError(f"No non-trivial factor found for N={n}")
