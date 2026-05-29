"""
Source 1: https://qualtran.readthedocs.io/en/latest/bloqs/cryptography/rsa/rsa.html
Source 2: https://quantumai.google/cirq/experiments/shor
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import cirq
import numpy as np
import sympy
from math import ceil, gcd, log2
from qualtran import BloqBuilder, QUInt
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.bookkeeping import Split, Join
from qualtran.bloqs.qft import QFTTextBook
from qualtran.bloqs.cryptography.rsa import ModExp

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
    order_finder: Callable[[Dict], int | None],
    config: Dict,
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
        r = order_finder(config)

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

def _build_qpe_circuit(t: int, N: int, a: int):
    """
    Build QPE circuit for the Ma: |x>->|a*x mod N>
    """
    if N <= 1:
        raise ValueError("N must be > 1")
    if gcd(a, N) != 1:
        raise ValueError("QPE requires gcd(a, N) == 1")

    m = int(ceil(log2(N)))
    bb = BloqBuilder()
    exp = bb.add_register('exp', t)

    # Hadamards on counting qubits
    exps = bb.add(Split(QUInt(t)), reg=exp)
    for i in range(t):
        exps[i] = bb.add(Hadamard(), q=exps[i])
    exp = bb.add(Join(QUInt(t)), reg=exps)

    # Modular exponentiation: |exp>|1> -> |exp>|a^exp mod N>
    exp, x = bb.add(
        ModExp(base=a, mod=N, exp_bitsize=t, x_bitsize=m),
        exponent=exp,
    )

    # Inverse QFT on counting qubits
    exp = bb.add(QFTTextBook(bitsize=t).adjoint(), q=exp)

    return bb.finalize(exp=exp, x=x)

def run_simulation(config: Dict[str, Any]):
    """
    Build Shor QPE circuit for factoring, without measurement.
    Returns the full final statevector.
    Expected config keys: t, N, a
    """
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))

    m = math.ceil(math.log2(N))

    print(f"\t**Building Qualtran QPE Bloq for N={N}, a={a}, t={t}...")
    bloq = _build_qpe_circuit(t=t, N=N, a=a)
    print("\t**Bloq built; simulating...")

    circuit, _ = bloq.to_cirq_circuit_and_quregs(
        exp=np.array([cirq.LineQubit(i) for i in range(t)]),
    )
    state = cirq.Simulator().simulate(circuit).final_state_vector

    # Swap qubits around to match tester
    state = state.reshape([2]*(t+m))
    state = np.transpose(state, list(range(m, m+t)) + list(range(m)))
    state = state.reshape((-1,))

    # print(np.round(state, 3).reshape((-1, 4)))
    return state
