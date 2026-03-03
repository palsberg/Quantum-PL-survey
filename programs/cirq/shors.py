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
from .shor.modularexponentiation import ModularExp
from .shor.quantumorderfinding import quantum_order_finder
from .shor.shors_common import classical_order_finder
from .shors_value import find_factor

def _build_qpe_circuit(t: int, N: int, a: int) -> tuple[cirq.Circuit, list[cirq.Qid]]:
    """
    Build QPE circuit for the Ma: |x>->|a*x mod N> with NO measurements
    returns (circuit, qubit_order)
    **Cirq requires an explicit qubit order
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
        cirq.qft(*counting, inverse=True, without_reverse=False),
    )

    qubit_order = counting + target
    return circuit, qubit_order


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
    circuit, qubit_order = _build_qpe_circuit(t=t, N=N, a=a)
    print("\t**QPE Circuit built")
    sv = cirq.final_state_vector(
        circuit,
        qubit_order=qubit_order,
        dtype=np.complex128,
    )

    return np.asarray(sv, dtype=np.complex128)


if __name__ == "__main__":
    n = 21
    factor = find_factor(n, order_finder=classical_order_finder)
    if factor is not None:
        print(f"A non-trivial factor of n = {n} is {factor}.")
    else:
        print(f"Failed to find a non-trivial factor of n = {n}.")