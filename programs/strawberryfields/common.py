"""Helper utilities for dual-rail encodings in Strawberry Fields."""

from __future__ import annotations

from typing import List

import numpy as np
import strawberryfields as sf
from strawberryfields import ops


def dual_rail_program(num_qubits: int):
    """Return (program, mode register) for a dual-rail encoded circuit."""
    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        for i in range(num_qubits):
            ops.Fock(1) | q[2 * i]     # |1,0> encodes logical |0>
            ops.Fock(0) | q[2 * i + 1]
    return prog


def apply_logical_x(q, qubit_idx: int, theta: float):
    """Approximate logical X via beam splitter between rails."""
    ops.BSgate(theta, 0.0) | (q[2 * qubit_idx], q[2 * qubit_idx + 1])


def apply_logical_z(q, qubit_idx: int, phi: float):
    """Logical Z as differential phase between rails."""
    ops.Rgate(phi) | q[2 * qubit_idx]
    ops.Rgate(-phi) | q[2 * qubit_idx + 1]


def apply_logical_zz(q, i: int, j: int, phi: float):
    """Implement ZZ via controlled-phase between matching rails."""
    ops.CZgate(phi) | (q[2 * i], q[2 * j])
    ops.CZgate(-phi) | (q[2 * i + 1], q[2 * j + 1])


def run_program(prog: sf.Program, cutoff: int = 3):
    backend_options = {"cutoff_dim": cutoff, "pure": True}
    eng = sf.Engine(backend="fock", backend_options=backend_options)
    return eng.run(prog)


def logical_state_from_result(result, num_qubits: int, cutoff: int) -> np.ndarray:
    """Project a dual-rail CV state onto the logical qubit subspace."""

    dm = result.state.dm()
    num_modes = 2 * num_qubits
    dim = cutoff**num_modes
    rho = dm.reshape((dim, dim))

    logical_dim = 2**num_qubits
    index_map: List[int] = []
    for basis in range(logical_dim):
        occ = []
        for qubit in range(num_qubits):
            bit = (basis >> (num_qubits - 1 - qubit)) & 1
            occ.extend([1 - bit, bit])
        idx = 0
        for photons in occ:
            idx = idx * cutoff + photons
        index_map.append(idx)

    rho_logic = rho[np.ix_(index_map, index_map)]
    eigvals, eigvecs = np.linalg.eigh(rho_logic)
    leading = eigvecs[:, np.argmax(eigvals)]
    norm = np.linalg.norm(leading)
    if norm == 0:
        raise ValueError("State leaked out of the logical dual-rail subspace.")
    return leading / norm
