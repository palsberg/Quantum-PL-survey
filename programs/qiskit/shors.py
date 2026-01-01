"""
Hybrid Shor factoring with FindOrderCandidate(a, N) as center piece:
- Quantum part: Phase estimation via QPE circuit (no measurement gates).
- Classical part: Continued fractions + gcd logic to extract factor.

Exposes run_simulation(config) that returns the full statevector as convention.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, log2
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.quantum_info import Statevector

import sys

def _mul_mod_N_gate(a: int, N: int, m: int) -> UnitaryGate:
    """
    Build the unitary |x> -> |(a*x) mod N> for x < N, and |x> -> |x> for x >= N,
    on m qubits (dimension 2^m). 
    This is a permutation matrix, wrapped as a UnitaryGate.
    """
    dim = 1 << m
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (a * x) % N if x < N else x # calculate correct permutation result
        U[y, x] = 1.0
    return UnitaryGate(U, label=f"mul_{a}_mod_{N}")


def _build_qpe_circuit(a:int, N:int, t:int)-> Tuple[QuantumCircuit,int]:
    """
    Build QPE circuit for the Ma: |x>->|a*x mod N> with NO measurements
    returns (circuit, m) where m is the number of target qubits.
    """
    if N<=1:
        raise ValueError("N must be >1")
    
    if gcd(a,N)!=1:
        raise ValueError("QPE requires gcd(a,N)==1 for order finding")
    
    m = int(ceil(log2(N))) 
    qc = QuantumCircuit(t + m, name=f"qpe_a{a}_N{N}")

    # Counting register in uniform superposition
    qc.h(range(t))

    # Target register initialized to |1>
    qc.x(t+m-1) # least-significant target qubit

    targets=range(t,t+m)

    # Controlled-U^(2^k). Instead of powering U, directly use a^(2^k) mod N to save unnecessary gates
    for idx in range(t):
        a_k=pow(a,1<<idx,N)
        U_k=_mul_mod_N_gate(a_k,N,m)
        cU_k=U_k.control(1)
        qc.append(cU_k,[idx]+list(targets)) # effect on target quits

    # Inverse QFT on reversed counting register to match the reference codes
    qc.append(QFT(t, do_swaps=False).inverse(), list(range(t)))
    
    return qc, m


def _statevector_from_circuit(qc:QuantumCircuit)->np.ndarray:
    """
    Extract full statevector with no measurement.
    See https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector#qiskit.quantum_info.Statevector
    """
    sv=Statevector.from_instruction(qc).data #np.darray type
    expected=2**qc.num_qubits
    if sv.size != expected:
        raise ValueError(f"Statevector dim {sv.size} != 2^{qc.num_qubits}")
    
    return np.asarray(sv,dtype=np.complex128)

def qiskit_to_reference_order(sv: np.ndarray, n: int) -> np.ndarray:
    # Qiskit: index bit k corresponds to qubit k (qubit 0 is LSB).
    # Reference: opposite qubit significance.
    # Convert by reversing qubit axes.
    return sv.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)

def run_simulation(config: Dict[str, Any]):
    """
    Build Shor QPE circuit for factoring 21 with a=2, without measurement.
    Returns the full final statevector.
    Expected config keys: t, N, a
    """
    t=int(config.get("t",8))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    if N != 21 or a != 2:
        # using the wrong code, raise error
        raise ValueError(
            f"This implementation is specialized for N=21, a=2 (got N={N}, a={a})."
        )

    qc, m = _build_qpe_circuit(a, N, t)
    sv=_statevector_from_circuit(qc)
    sv = qiskit_to_reference_order(sv, qc.num_qubits)


    print("qiskit code:")
    np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
    p = np.abs(sv)**2
    print(np.sum(p > 1e-12), np.max(p[p <= 1e-12]) if np.any(p <= 1e-12) else 0.0)

    return sv



#############################################################################
# Below are lecture note style find order candidate function implementations#
#############################################################################

def _counting_marginal_probs(state: np.ndarray, t: int, m: int) -> np.ndarray:
    """
    Circuit qubits are [0..t-1]=counting, [t..t+m-1]=target.
    Qiskit basis index uses qubit k as bit k (little-endian), so index = c + (y << t).
    Marginal over target: prob[c] = sum_over_y |amp[c + (y<<t)]|^2
    """
    probs = np.zeros(1 << t, dtype=np.float64)
    for c in range(1 << t):
        s = 0.0
        base = c
        for y in range(1 << m):
            idx = base + (y << t)
            amp = state[idx]
            s += (amp.real * amp.real + amp.imag * amp.imag)
        probs[c] = s
    # Numerical safety
    total = probs.sum()
    if total <= 0:
        raise ValueError("Zero total probability (unexpected).")
    return probs / total

def _fraction_with_bounded_denominator_from_counts(c: int, t: int, N: int) -> Fraction:
    """
    Continued fraction helper:
    Given measured counting value c (0..2^t-1), interpret f = c/2^t and return best rational approximation with denominator <= N.
    """
    f = Fraction(c, 1 << t)  # exact
    return f.limit_denominator(N)

def FindOrderCandidate(a: int, N: int, *, t: Optional[int] = None, top_k: int = 32) -> int:
    """
    Hybrid routine (quantum + classical)
      f = PhaseEstimation(Ma, |1>)
      q = FractionWithBoundedDenominator(f, N)
      return Denominator(q)
    but we have NO measurement gates: we compute the counting-register distribution
    from the final statevector and try the most likely outcomes first.

    Returns an integer r > 0 (may be wrong), but we try to validate pow(a, r, N)==1.
    """

    if N <= 1:
        raise ValueError("N must be > 1")
    if gcd(a, N) != 1:
        return 0
    
    if t is None:
        t = max(8, 2 * int(ceil(log2(N))) + 2)

    qc, m = _build_qpe_circuit(a, N, t)
    state = _statevector_from_circuit(qc)
    probs = _counting_marginal_probs(state, t, m)

    candidates = np.argsort(-probs)[: min(top_k, probs.size)]

    best_r = 0
    for c in candidates:
        q = _fraction_with_bounded_denominator_from_counts(int(c), t, N)
        r = q.denominator
        if r <= 0:
            continue
        # Validate the order candidate when possible
        if pow(a, r, N) == 1:
            return r
        # Keep some fallback
        if best_r == 0:
            best_r = r

    return best_r if best_r != 0 else 0

def Shor(N: int, *, max_tries: int = 25, t: Optional[int] = None) -> int:
    """
    Classical Shor loop (hybrid): pick random a, check gcd, call FindOrderCandidate, etc.
    Returns a nontrivial factor d of N, or raises if it gives up.
    Matches the lecture structure: gcd(a,N), then FindOrderCandidate, then gcd(a^(r/2)-1, N). 
    """
    if N % 2 == 0:
        return 2
    if N <= 3:
        raise ValueError("N must be composite and > 3")

    rng = np.random.default_rng(0)  # deterministic; change/seed if you want true randomness

    for _ in range(max_tries):
        a = int(rng.integers(2, N))  # 2..N-1
        d = gcd(a, N)
        if d > 1:
            return d

        r = FindOrderCandidate(a, N, t=t)
        if r <= 0 or (r % 2) != 0:
            continue

        # x = a^(r/2) - 1 (mod N), then gcd(x, N)
        x = (pow(a, r // 2, N) - 1) % N
        d2 = gcd(x, N)
        if 1 < d2 < N:
            return d2

    raise RuntimeError("Shor gave up without finding a factor.")


def factor_integer(N: int) -> List[int]:
    """Simple driver: recursively factor N using Shor until prime-ish (naive)."""
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        p = 3
        while p * p <= n:
            if n % p == 0:
                return False
            p += 2
        return True

    factors: List[int] = []

    def rec(n: int):
        if n == 1:
            return
        if is_prime(n):
            factors.append(n)
            return
        d = Shor(n)
        rec(d)
        rec(n // d)

    rec(N)
    factors.sort()
    return factors

if __name__ == "__main__":
    N=21
    print(f"Order candidate r for a=2, N={N}:", FindOrderCandidate(2, N, t=12))
    print(f"One factor from Shor({N}):", Shor(N, max_tries=5, t=12))
    print(f"Full factorization of {N}:", factor_integer(N))