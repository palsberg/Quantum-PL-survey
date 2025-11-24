"""Shared helpers for PyQuil-based Hamiltonian simulation programs."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from pyquil import Program
from pyquil.gates import CNOT, H, PHASE, RX, RY, RZ
from pyquil.simulation import NumpyWavefunctionSimulator
from pyquil.quilbase import DefGate, Gate


def _basis_change(prog: Program, qubit: int, axis: str, inverse: bool) -> None:
    axis = axis.upper()
    if axis == "X":
        prog += H(qubit)
    elif axis == "Y":
        if inverse:
            prog += PHASE(+np.pi / 2, qubit)
            prog += H(qubit)
        else:
            prog += H(qubit)
            prog += PHASE(-np.pi / 2, qubit)


def _apply_zz_rotation(prog: Program, qa: int, qb: int, theta: float) -> None:
    prog += CNOT(qa, qb)
    prog += RZ(2 * theta, qb)
    prog += CNOT(qa, qb)


def apply_two_qubit_rotation(prog: Program, qa: int, qb: int, theta: float, axis: str) -> None:
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported axis {axis}")
    _basis_change(prog, qa, axis, inverse=False)
    _basis_change(prog, qb, axis, inverse=False)
    _apply_zz_rotation(prog, qa, qb, theta)
    _basis_change(prog, qa, axis, inverse=True)
    _basis_change(prog, qb, axis, inverse=True)


def apply_single_qubit_rotation(prog: Program, qubit: int, theta: float, axis: str) -> None:
    axis = axis.upper()
    if axis == "Z":
        prog += RZ(2 * theta, qubit)
    elif axis == "X":
        prog += RX(2 * theta, qubit)
    elif axis == "Y":
        prog += RY(2 * theta, qubit)
    else:
        raise ValueError(f"Unsupported axis {axis}")


def _qubit_indices(inst: Gate) -> List[int]:
    return [int(q.index) for q in inst.qubits]


def simulate_statevector(prog: Program, num_qubits: int) -> np.ndarray:
    simulator = NumpyWavefunctionSimulator(n_qubits=num_qubits)
    custom_defs = {gate.name: gate for gate in prog.defined_gates}
    for inst in prog:
        if not isinstance(inst, Gate):
            continue

        qubits = _qubit_indices(inst)

        if inst.name in custom_defs:
            matrix = np.asarray(custom_defs[inst.name].matrix, dtype=np.complex128)
            simulator.do_gate_matrix(matrix, qubits)
            continue

        if inst.modifiers:
            matrix = np.asarray(inst.to_unitary_mut(len(inst.qubits)), dtype=np.complex128)
            simulator.do_gate_matrix(matrix, qubits)
            continue

        simulator.do_gate(inst)

    wf = simulator.wf.reshape(-1)
    return np.asarray(wf, dtype=np.complex128)


def trotterize_tfim(
    num_sites: int, J: float, h: float, total_time: float, steps: int
) -> Tuple[Program, Sequence[int]]:
    prog = Program()
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            apply_two_qubit_rotation(prog, i, i + 1, J * dt, axis="Z")
        for i in range(num_sites):
            apply_single_qubit_rotation(prog, i, h * dt, axis="X")
    return prog, list(range(num_sites))


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, total_time: float, steps: int
) -> Tuple[Program, Sequence[int]]:
    prog = Program()
    dt = total_time / steps
    for _ in range(steps):
        for i in range(num_sites - 1):
            for axis in ("X", "Y", "Z"):
                apply_two_qubit_rotation(prog, i, i + 1, J * dt, axis=axis)
        for i in range(num_sites):
            apply_single_qubit_rotation(prog, i, field * dt, axis="Z")
    return prog, list(range(num_sites))
