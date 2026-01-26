"""
Source 1: https://qualtran.readthedocs.io/en/latest/bloqs/cryptography/rsa/rsa.html
Source 2: https://quantumai.google/cirq/experiments/shor
"""

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple
import cirq
import numpy as np
import sympy
from math import ceil, gcd, log2
from qualtran.bloqs.cryptography.rsa import ModExp, RSAPhaseEstimate
from .shor.shors_common import classical_order_finder
from qualtran import BloqBuilder, QUInt
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.qft import QFTTextBook
from qualtran.simulation.tensor import initialize_from_zero
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager
from qualtran._infra.gate_with_registers import get_named_qubits, merge_qubits
# from qualtran.bloqs.mod_arithmetic import CModMulK
# from qualtran.bloqs.basic_gates import XGate

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
    order_finder: Callable[[int, int], int | None] = classical_order_finder,
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

def _build_qpe_circuit(t: int, N: int, a: int) -> np.ndarray:
    """
    Build QPE circuit for the Ma: |x>->|a*x mod N> with NO measurements
    """
    if N <= 1:
        raise ValueError("N must be > 1")
    if gcd(a, N) != 1:
        raise ValueError("QPE requires gcd(a, N) == 1")

    m = int(ceil(log2(N)))

    bb = BloqBuilder()

    # exponent register (t qubits)
    exponent = bb.add_register_from_dtype("exponent", QUInt(t))

    # H on each exponent qubit.
    exp_bits = bb.split(exponent)
    for i in range(t):
        exp_bits[i] = bb.add(Hadamard(), q=exp_bits[i])
    exponent = bb.join(exp_bits)

    # ModExp takes exponent and produces x = a^exponent mod N,
    # and internally prepares x starting from |1>.
    mod_exp = ModExp(base=a, mod=N, exp_bitsize=t, x_bitsize=m)
    exponent, x = bb.add(mod_exp, exponent=exponent)

    # iQFT on exponent (with_reverse=True by default)
    exponent = bb.add(QFTTextBook(t).adjoint(), q=exponent) 

    return bb.finalize(exponent=exponent, x=x)

def _simulate_bloq_and_project_ancillas(bloq) -> np.ndarray:
    """
    Simulate the decomposed bloq as a Cirq circuit and return the statevector
    on ONLY the (signature) registers, with all extra ancillas projected to |0...0>.
    """
    print("Checkpoint 1")
    cbloq = bloq.flatten()

    print("Checkpoint 2")
    init_quregs = get_named_qubits(bloq.signature)
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    circuit, quregs_out = cbloq.to_cirq_circuit_and_quregs(qubit_manager=qm, **init_quregs)

    # Signature qubits (ordered by signature)
    print("Checkpoint 3")
    sig_qubits = merge_qubits(bloq.signature, **quregs_out)
    sig_set = set(sig_qubits)

    # Any extra ancillas introduced by decomposition
    extra_qubits = sorted([q for q in circuit.all_qubits() if q not in sig_set], key=lambda q: str(q))

    qubit_order = list(sig_qubits) + extra_qubits
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    full_sv = np.asarray(result.final_state_vector, dtype=np.complex128)

    bits_per_register = [reg.total_bits() for reg in bloq.signature]
    if extra_qubits:
        bits_per_register.append(len(extra_qubits))

    # Reshape into registers (+ one final "extra ancilla" register if present),
    # then slice extra ancillas to |0...0>.
    shape = [1 << b for b in bits_per_register]
    reshaped = full_sv.reshape(shape)

    indexer = [slice(None)] * len(bloq.signature)
    if extra_qubits:
        indexer.append(0)

    projected = reshaped[tuple(indexer)].reshape(-1)

    # If the decomposition is perfectly clean, this norm is already 1.
    # Renormalize anyway for safety.
    norm = np.linalg.norm(projected)
    if norm == 0:
        raise ValueError("Projected state on ancillas=|0...0> has zero norm (unexpected).")

    return projected / norm
    

def run_simulation(config: Dict[str, Any]):
    """
    Build Shor QPE circuit for factoring 21 with a=2, without measurement.
    Returns the full final statevector.
    Expected config keys: t, N, a
    """
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))

    if N != 21 or a != 2:
        raise ValueError(
            f"This implementation is specialized for N=21, a=2 (got N={N}, a={a})."
        )

    print(f"\t**Building Qualtran QPE Bloq for N={N}, a={a}, t={t}...")
    qpe_bloq = _build_qpe_circuit(t=t, N=N, a=a)
    print("\t**Bloq built; simulating...")

    sv = _simulate_bloq_and_project_ancillas(qpe_bloq)
    return np.asarray(sv, dtype=np.complex128)
    
    return sv

if __name__ == "__main__":
    N = 21
    factor = find_factor(N, order_finder=classical_order_finder)
    if factor is not None:
        print(f"A non-trivial factor of N = {N} is {factor}.")
    else:
        print(f"Failed to find a non-trivial factor of N = {N}.")