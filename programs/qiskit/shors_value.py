"""
Measured (value) Shor driver that reuses shors.py building blocks.

- Quantum: QPE circuit from shors.py, then measure counting register.
- Classical: continued fractions + gcd to extract a factor.

Required API:
    def run_simulation(config: Dict[str, Any]) -> int
"""

from __future__ import annotations

from math import ceil, gcd, log2
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile

# Aer sampler
from qiskit_aer import Aer

# Import what already exists in shors.py
# Adjust the import path to match your repo layout, e.g.:
#   from .shors import _build_qpe_circuit, _fraction_with_bounded_denominator_from_counts
# or
#   from programs.qiskit.shors import ...
from .shors import (
    _build_qpe_circuit,
    _fraction_with_bounded_denominator_from_counts,
)


# def _measure_counting_register_once(
#     tqc,
#     backend,
#     *,
#     t: int,
#     shots: int,
#     seed: int
# ) -> int:
#     """
#     Run shots on a pre-transpiled measured-QPE circuit.
#     Return the most frequent observed integer c in [0, 2^t).
#     """
#     job = backend.run(tqc, shots=shots, seed_simulator=seed)
#     result = job.result()
#     counts = result.get_counts()

#     bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
#     c = int(bitstring, 2)
#     return c

def _measure_counting_register_counts(
    tqc,
    backend,
    *,
    shots: int,
    seed: int
) -> dict:
    """Run shots on pre-transpiled circuit and return counts dict."""
    job = backend.run(tqc, shots=shots, seed_simulator=seed)
    return job.result().get_counts()

def FindOrderCandidate_measured(
    a: int,
    N: int,
    *,
    t: Optional[int] = None,
    shots: int = 256,
    retries: int = 16,
    seed: int = 0,
) -> int:
    """
    Measured order finding:
      - run QPE, measure counting register
      - convert measured c to fraction c/2^t, continued fraction denom <= N
      - validate pow(a, r, N) == 1
    If it fails, reconstruct+measure again (retries times).

    Returns r or 0 if it gives up.
    """
    print("order of ", a)
    if N <= 1:
        raise ValueError("N must be > 1")
    if gcd(a, N) != 1:
        return 0

    if t is None:
        t = max(8, 2 * int(ceil(log2(N))) + 2)

    # Build the QPE circuit once
    qc, m = _build_qpe_circuit(a, N, t)

    meas = QuantumCircuit(qc.num_qubits, t)
    meas.compose(qc, inplace=True)
    meas.measure(range(t), range(t))

    backend = Aer.get_backend("aer_simulator")

    # Transpile once
    tqc = transpile(meas, backend, optimization_level=0)
    print("transpile finished for ", a)

    # for j in range(retries):
    #     c = _measure_counting_register_once(
    #         tqc,
    #         backend,
    #         t=t,
    #         shots=shots,
    #         seed=seed + j
    #     )

    #     q = _fraction_with_bounded_denominator_from_counts(c, t, N)
    #     r = q.denominator

    #     if r > 0 and pow(a, r, N) == 1:
    #         return r

        # Run ONCE with aggregated shots instead of many retries.
    total_shots = shots * max(1, retries)
    counts = _measure_counting_register_counts(
        tqc, backend, shots=total_shots, seed=seed
    )

    # Try several most-likely outcomes, not just the single mode.
    # This is both faster and more robust than retrying.
    topk = 8
    for bitstring, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topk]:
        c = int(bitstring, 2)
        q = _fraction_with_bounded_denominator_from_counts(c, t, N)
        r = q.denominator
        if r > 0 and pow(a, r, N) == 1:
            print("order for ", a, " is ", r)
            return r

    return 0


def _try_shor_with_base_a_measured(
    N: int,
    a: int,
    *,
    t: Optional[int],
    shots: int,
    retries: int,
    seed: int,
) -> int:
    """
    One measured Shor attempt with fixed base a. Returns factor or 0.
    """
    print("a:",a)
    d = gcd(a, N)
    if 1 < d < N:
        return d

    r = FindOrderCandidate_measured(a, N, t=t, shots=shots, retries=retries, seed=seed)
    if r <= 0 or (r % 2) != 0:
        return 0

    ar2 = pow(a, r // 2, N)

    d1 = gcd((ar2 - 1) % N, N)
    if 1 < d1 < N:
        return d1

    d2 = gcd((ar2 + 1) % N, N)
    if 1 < d2 < N:
        return d2

    return 0


def Shor_value(
    N: int,
    *,
    a: Optional[int] = None,
    t: Optional[int] = None,
    max_tries: int = 25,
    seed: int = 0,
    allow_random_a: bool = True,
    shots: int = 256,
    retries: int = 16,
) -> int:
    """
    Returns a nontrivial factor of N (measured-style) or raises RuntimeError.
    """
    if N <= 1:
        raise ValueError("N must be > 1")
    if N % 2 == 0:
        return 2
    if N <= 3:
        raise ValueError("N must be composite and > 3")

    if t is None:
        t = max(8, 2 * int(ceil(log2(N))) + 2)

    rng = np.random.default_rng(seed)

    attempts: List[int] = []
    if a is not None:
        a0 = int(a) % N
        if a0 < 2:
            a0 = 2
        attempts.append(a0)

    if allow_random_a:
        while len(attempts) < max_tries:
            attempts.append(int(rng.integers(2, N-2)))
    else:
        if not attempts:
            raise ValueError("allow_random_a=False requires providing a valid 'a' in config.")

    for i, base in enumerate(attempts):
        factor = _try_shor_with_base_a_measured(
            N,
            base,
            t=t,
            shots=shots,
            retries=retries,
            seed=seed + 1000 * i,
        )
        if 1 < factor < N:
            return np.array(factor)

    raise RuntimeError("Shor failed to find a factor with the given configuration.")


def run_simulation(config: Dict[str, Any]) -> int:
    """
    Harness entry point: returns a nontrivial factor of N.
    """
    if "N" not in config:
        raise ValueError("config must include key 'N'")

    N = int(config["N"])
    a = config.get("a", None)
    a = int(a) if a is not None else None
    t = config.get("t", None)
    t = int(t) if t is not None else None

    max_tries = int(config.get("max_tries", 10))
    seed = int(config.get("seed", np.random.SeedSequence().entropy))
    allow_random_a = bool(config.get("allow_random_a", True))

    shots = int(config.get("shots", 256))       # measurement shots per attempt
    retries = int(config.get("retries", 16))    # reconstruct+measure attempts for order finding

    return Shor_value(
        N,
        a=a, # none
        t=t,
        max_tries=max_tries,
        seed=seed,
        allow_random_a=allow_random_a,
        shots=shots,
        retries=retries,
    )


# for the benchmarking 
def build_circuit(config: Dict[str, Any]) -> QuantumCircuit:
    N = int(config["N"])
    a = int(config.get("a", 2))
    t = int(config.get("t", 8))

    qc, _ = _build_qpe_circuit(a, N, t)

    meas = QuantumCircuit(qc.num_qubits, t)
    meas.compose(qc, inplace=True)
    meas.measure(range(t), range(t))

    return meas

if __name__ == "__main__":
    
    cfg = {"N": 45, "t": 6, "shots": 512, "retries": 10, "max_tries": 1}
    
    print("here")
    print("Factor:", run_simulation(cfg))