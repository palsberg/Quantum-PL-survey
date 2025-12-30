"""
Shor's algorithm order finding for factoring 21 using a=2 in qiskit. 
Returns statevector.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector


def _simulate_statevector_like_lcu(qc: QuantumCircuit) -> np.ndarray:
    """Extract full statevector like the LCU helper: Statevector.from_instruction(qc).data with checks."""
    state = Statevector.from_instruction(qc).data
    n = qc.num_qubits
    expected_dim = 2**n
    if state.size != expected_dim:
        raise ValueError(
            f"Statevector dimension {state.size} incompatible with circuit qubits {n}."
        )
    # Ensure consistent dtype
    return np.asarray(state, dtype=np.complex128)

def _build_U_mul2_mod21() -> QuantumCircuit: # just for 21;2 case
    N = 21
    m = int(math.ceil(math.log2(N)))  # 5
    if m != 5:
        raise ValueError("This helper is hardcoded for N=21 (m=5).")

    U_qc = QuantumCircuit(m, name="2Mod21")
    # Make 11 into 21 so the rest would convert it to 1 (for this case).
    U_qc.cswap(0, 3, 4)
    U_qc.cswap(0, 1, 2)

    # After 16 comes 11, not 1: adjust.
    U_qc.cx(4, 2)
    U_qc.cx(4, 0)

    # Left rotation of bits for 1,2,4,8,16
    U_qc.swap(3, 4)
    U_qc.swap(0, 3)
    U_qc.swap(3, 2)
    U_qc.swap(2, 1)

    return U_qc


def _apply_control_power(qc: QuantumCircuit, control_qubit: int, U_gate, target_qubits: Sequence[int], power:int) -> None:
    """
    Apply given U_gate power of times on correspondign qubits for QPE
    """
    cU = U_gate.control(1)
    for _ in range(power):
        qc.append(cU, [control_qubit] + list(target_qubits))





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

    m=int(math.ceil(math.log2(N)))
    U_qc=_build_U_mul2_mod21()
    U_gate=U_qc.to_gate()
    U_gate.name="2Mod21"

    