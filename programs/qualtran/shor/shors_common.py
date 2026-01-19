"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import math

"""Function for classically computing the order of an element of Z_n."""
def classical_order_finder(x: int, n: int) -> int | None:
    """Computes smallest positive r such that x**r mod n == 1.

    Args:
        x: Integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: Modulus of the multiplicative group.

    Returns:
        Smallest positive integer r such that x**r == 1 mod n.
        Always succeeds (and hence never returns None).

    Raises:
        ValueError when x is 1 or not an element of the multiplicative
        group of integers modulo n.
    """
    # Make sure x is both valid and in Z_n.
    if x < 2 or x >= n or math.gcd(x, n) > 1:
        raise ValueError(f"Invalid x={x} for modulus n={n}.")

    # Determine the order.
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r