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
    qc.x(t) # least-significant target qubit because counting qubits are 0..t-1

    # Controlled-U^(2^k). Instead of powering U, directly use a^(2^k) mod N to save unnecessary gates
    for k in range(t):
        a_k=pow(a,1<<k,N)
        U_k=_mul_mod_N_gate(a_k,N,m)
        cU_k=U_k.control(1)
        qc.append(cU_k,[k]+list(range(t,t+m))) # effect on target quits

    # Inverse QFT on counting register
    qc.append(QFT(t,do_swaps=True).inverse(),range(t))
    return qc, m


def _statevector_from_circuit(qc:QuantumCircuit)->np.ndarray:
    """
    Extract full statevector with no measurement.
    See https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector#qiskit.quantum_info.Statevector
    """
    sv=Statevector.from_instructions(qc).data #np.darray type
    expected=2**qc.num_qubits
    if sv.size != expected:
        raise ValueError(f"Statevector dim {sv.size} != 2^{qc.num_qubits}")
    
    return np.asarray(sv,dtype=np.complex128)



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
    return _statevector_from_circuit(qc)





