"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import cirq
import math
import random
import sympy
import numpy as np
from math import ceil, gcd, log2
from typing import Callable, Sequence
from typing import Any, Dict, List, Optional, Tuple
from shor.modularexponentiation import ModularExp
from shor.quantumorderfinding import quantum_order_finder
from shor.common import classical_order_finder

"""Functions for factoring from start to finish."""
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
    max_attempts: int = 30,
) -> int | None:
    """Returns a non-trivial factor of composite integer n.

    Args:
        n: Integer to factor.
        order_finder: Function for finding the order of elements of the
            multiplicative group of integers modulo n.
        max_attempts: number of random x's to try, also an upper limit
            on the number of order_finder invocations.

    Returns:
        Non-trivial factor of n or None if no such factor was found.
        Factor k of n is trivial if it is 1 or n.
    """
    # If the number is prime, there are no non-trivial factors.
    if sympy.isprime(n):
        print("n is prime!")
        return None

    # If the number is even, two is a non-trivial factor.
    if n % 2 == 0:
        return 2

    # If n is a prime power, we can find a non-trivial factor efficiently.
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c

    for _ in range(max_attempts):
        # Choose a random number between 2 and n - 1.
        x = random.randint(2, n - 1)

        # Most likely x and n will be relatively prime.
        c = math.gcd(x, n)

        # If x and n are not relatively prime, we got lucky and found
        # a non-trivial factor.
        if 1 < c < n:
            return c

        # Compute the order r of x modulo n using the order finder.
        r = order_finder(x, n)

        # If the order finder failed, try again.
        if r is None:
            continue

        # If the order r is even, try again.
        if r % 2 != 0:
            continue

        # Compute the non-trivial factor.
        y = x ** (r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c

    print(f"Failed to find a non-trivial factor in {max_attempts} attempts.")
    return None

def _build_qpe_circuit(*, t: int, N: int, a: int) -> tuple[cirq.Circuit, list[cirq.Qid]]:
    """
    Build QPE circuit for the Ma: |x>->|a*x mod N> with NO measurements
    returns (circuit, qubit_order)
    """
    if N<=1:
        raise ValueError("N must be >1")
    if gcd(a,N)!=1:
        raise ValueError("QPE requires gcd(a,N)==1 for order finding")
    
    m = int(ceil(log2(N)))
    counting = list(cirq.LineQubit.range(t))
    target = list(cirq.LineQubit.range(t, t + m))

    # ModularExp expects registers as (target, exponent, base, modulus).
    mod_exp = ModularExp([2] * m, [2] * t, a, N)

    circuit = cirq.Circuit(
        cirq.H.on_each(*counting),
        cirq.X(target[-1]),
        mod_exp.on(*target, *counting),
        cirq.qft(*counting, inverse=True, without_reverse=Falseq),
    )

    qubit_order = counting + target
    return circuit, qubit_order


def run_simulation(config: Dict[str, Any]):
    """
    Build Shor QPE circuit for factoring 21 with a=2, without measurement.
    Returns the full final statevector.
    Expected config keys: t, N, a
    """
    print("--------Running Shor in Cirq--------")
    t=int(config.get("t",8))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    if N != 21 or a != 2:
        # using the wrong code, raise error
        raise ValueError(
            f"This implementation is specialized for N=21, a=2 (got N={N}, a={a})."
        )
        
    circuit, qubit_order = _build_qpe_circuit(t=t, N=N, a=a)
    sv = cirq.final_state_vector(
        circuit,
        qubit_order=qubit_order,
        dtype=np.complex128,
    )

    return np.asarray(sv, dtype=np.complex128)


if __name__ == "__main__":
    n = 15
    factor = find_factor(n, order_finder=classical_order_finder)
    if factor is not None:
        print(f"A non-trivial factor of n = {n} is {factor}.")
    else:
        print(f"Failed to find a non-trivial factor of n = {n}.")