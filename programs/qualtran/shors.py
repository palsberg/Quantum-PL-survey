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
    returns (circuit, qubit_order)
    """
    if N <= 1:
        raise ValueError("N must be > 1")
    if gcd(a, N) != 1:
        raise ValueError("QPE requires gcd(a, N) == 1")

    m = int(ceil(log2(N)))

    bb = BloqBuilder()

    # exponent register
    exponent = bb.add_register_from_dtype("exponent", QUInt(t))

    # Hadamard on each exponent bit
    bits = bb.split(exponent)
    for i in range(t):
        bits[i] = bb.add(Hadamard(), q=bits[i])
    exponent = bb.join(bits, dtype=QUInt(t))

    # modular exponentiation (creates RIGHT register x)
    modexp = ModExp(base=a, mod=N, exp_bitsize=t, x_bitsize=m)
    exponent, x = bb.add(modexp, exponent=exponent)

    # inverse QFT (with swaps, matching cirq.qft(... without_reverse=False))
    exponent = bb.add(QFTTextBook(bitsize=t, with_reverse=True).adjoint(), q=exponent)
    cbloq = bb.finalize(exponent=exponent, x=x)
    
    # --- Convert CompositeBloq -> Cirq circuit and simulate ---
    print("a1")
    init_quregs = get_named_qubits(cbloq.signature)
    print("a12")
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    print("a13")
    circuit, quregs_out = cbloq.to_cirq_circuit_and_quregs(qubit_manager=qm, **init_quregs)

    print("a2")
    # Put qubits in a stable order: signature qubits first, then any extra ancillas.
    sig_qubits = merge_qubits(cbloq.signature, **quregs_out)
    sig_set = set(sig_qubits)
    extra_qubits = sorted([q for q in circuit.all_qubits() if q not in sig_set], key=str)
    qubit_order = list(sig_qubits) + extra_qubits

    print("a3")
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    out = result.final_state_vector

    return np.asarray(out, dtype=np.complex128)
    

def run_simulation(config: Dict[str, Any]):
    """
    Build Shor QPE circuit for factoring 21 with a=2, without measurement.
    Returns the full final statevector.
    Expected config keys: t, N, a
    """
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    if N != 21 or a != 2:
        # using the wrong code, raise error
        raise ValueError(
            f"This implementation is specialized for N=21, a=2 (got N={N}, a={a})."
        )
    
    print(f"\t**Building QPE Circuit for N={N}, a={a}, t={t}...")
    sv = _build_qpe_circuit(t=t, N=N, a=a)
    print("\t**QPE Circuit built")
    
    return sv

if __name__ == "__main__":
    N = 21
    factor = find_factor(N, order_finder=classical_order_finder)
    if factor is not None:
        print(f"A non-trivial factor of N = {N} is {factor}.")
    else:
        print(f"Failed to find a non-trivial factor of N = {N}.")