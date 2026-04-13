from __future__ import annotations
from typing import Any, Callable, Dict
import numpy as np
from .shor.quantumorderfinding import quantum_order_finder
import sympy
import math
import random

def find_factor_of_prime_power(n: int) -> int | None:
    """Returns non-trivial factor of n if n is a prime power, else None."""
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1**k == n:
            return c1
        c2 = math.ceil(c)
        if c2**k == n:
            return c2
    return None


def find_factor(
    n: int,
    order_finder: Callable[[int, int], int | None] = quantum_order_finder,
    max_attempts: int = 30, ) -> int | None:
    """Returns a non-trivial factor of composite integer n.
        n: Integer to factor.
        order_finder: Function for finding the order
        max_attempts: number of random x's to try
    """
    # If the number is prime, there are no non-trivial factors
    if sympy.isprime(n):
        print("n is prime!")
        return None

    # If the number is even, 2 is a non-trivial factor
    if n % 2 == 0:
        return 2

    # If n is a prime power, we can find a non-trivial factor efficiently
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c

    for _ in range(max_attempts):
        # Choose a random number between 2 and n - 1
        x = random.randint(2, n - 1)

        # Most likely x and n will be relatively prime
        c = math.gcd(x, n)

        # If x and n are not relatively prime, we got lucky and found a non-trivial factor
        if 1 < c < n:
            return c

        # Compute the order r of x modulo n
        r = order_finder(x, n)
        if r is None:
            continue

        # If the order is even, try again
        if r % 2 != 0:
            continue

        # Compute the non-trivial factor
        y = x ** (r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c

    print(f"Failed to find a non-trivial factor in {max_attempts} attempts.")
    return None

def measure_vector(statevector: np.ndarray) -> int:
    """Sample a basis state from a statevector and return its integer index."""
    probs = np.abs(statevector) ** 2
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

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
