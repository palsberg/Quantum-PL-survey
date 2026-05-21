"""
Shared helpers for the Cirq Hamiltonian-simulation programs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cirq
import numpy as np


def _apply_basis_change(circuit: cirq.Circuit, qubit: cirq.Qid, axis: str, inverse: bool) -> None:
    """Apply basis change that maps Z to the requested Pauli axis (or inverse)."""
    if axis == "X":
        circuit.append(cirq.H(qubit))
    elif axis == "Y":
        if inverse:
            circuit.append(cirq.S(qubit))
            circuit.append(cirq.H(qubit))
        else:
            circuit.append(cirq.H(qubit))
            circuit.append(cirq.S(qubit) ** -1)


def _apply_zz_rotation(circuit: cirq.Circuit, qa: cirq.Qid, qb: cirq.Qid, theta: float) -> None:
    """Implement exp(-i * theta * Z⊗Z)."""
    circuit.append(cirq.CNOT(qa, qb))
    circuit.append(cirq.rz(2 * theta).on(qb))
    circuit.append(cirq.CNOT(qa, qb))


def apply_two_qubit_rotation(
    circuit: cirq.Circuit, qa: cirq.Qid, qb: cirq.Qid, theta: float, axis: str
) -> None:
    """Apply exp(-i * theta * σ_axis⊗σ_axis)."""
    axis = axis.upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported axis {axis}")
    _apply_basis_change(circuit, qa, axis, inverse=False)
    _apply_basis_change(circuit, qb, axis, inverse=False)
    _apply_zz_rotation(circuit, qa, qb, theta)
    _apply_basis_change(circuit, qa, axis, inverse=True)
    _apply_basis_change(circuit, qb, axis, inverse=True)


def apply_single_qubit_rotation(
    circuit: cirq.Circuit, qubit: cirq.Qid, theta: float, axis: str
) -> None:
    """Apply exp(-i * theta * σ_axis) on a single qubit."""
    axis = axis.upper()
    if axis == "Z":
        circuit.append(cirq.rz(2 * theta).on(qubit))
    elif axis == "X":
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(2 * theta).on(qubit))
        circuit.append(cirq.H(qubit))
    elif axis == "Y":
        circuit.append(cirq.S(qubit) ** -1)
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(2 * theta).on(qubit))
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.S(qubit))
    else:
        raise ValueError(f"Unsupported axis {axis}")


def simulate_statevector(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid]) -> np.ndarray:
    """Simulate the given circuit initialized to |0...0> and return the statevector."""
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def _sorted_qubits_from_sums(*sums: "cirq.PauliSum") -> Sequence[cirq.Qid]:
    """Collect and sort the qubits appearing in one or more PauliSums."""
    qubit_set: set[cirq.Qid] = set()
    for ps in sums:
        qubit_set.update(ps.qubits)
    return tuple(sorted(qubit_set))


def _tfim_pauli_sums(num_sites: int, J: float, h: float) -> tuple["cirq.PauliSum", "cirq.PauliSum"]:
    """Build TFIM H_ZZ and H_X as Cirq PauliSums."""
    sites = cirq.LineQubit.range(num_sites)

    ps_zz = cirq.PauliSum()
    for i in range(num_sites - 1):
        ps_zz += J * cirq.Z(sites[i]) * cirq.Z(sites[i+1])

    ps_x = cirq.PauliSum()
    for i in range(num_sites):
        ps_x += h * cirq.X(sites[i])

    return ps_zz, ps_x


def _heis_pauli_sums(
    num_sites: int, J: float, field: float
) -> tuple["cirq.PauliSum", "cirq.PauliSum", "cirq.PauliSum", "cirq.PauliSum"]:
    """Build Heisenberg XXX interaction and field terms as Cirq PauliSums."""
    sites = cirq.LineQubit.range(num_sites)

    ps_xx = cirq.PauliSum()
    ps_yy = cirq.PauliSum()
    ps_zz = cirq.PauliSum()
    for i in range(num_sites - 1):
        ps_xx += J * cirq.X(sites[i]) * cirq.X(sites[i+1])
        ps_yy += J * cirq.Y(sites[i]) * cirq.Y(sites[i+1])
        ps_zz += J * cirq.Z(sites[i]) * cirq.Z(sites[i+1])

    ps_field = cirq.PauliSum()
    for i in range(num_sites):
        ps_field += field * cirq.Z(sites[i])

    return ps_xx, ps_yy, ps_zz, ps_field


def trotterize_tfim(
    num_sites: int, J: float, h: float, time: float, steps: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Construct a Lie–Trotter circuit for the TFIM Hamiltonian."""
    ps_zz, ps_x = _tfim_pauli_sums(num_sites, J, h)
    qubits = _sorted_qubits_from_sums(ps_zz, ps_x)
    circuit = cirq.Circuit()
    dt = time / steps

    exp_zz = cirq.PauliSumExponential(ps_zz, exponent=-dt)
    exp_x = cirq.PauliSumExponential(ps_x, exponent=-dt)
    for _ in range(steps):
        circuit.append(exp_zz)
        circuit.append(exp_x)
    return circuit, qubits


def trotterize_heisenberg_xxx(
    num_sites: int, J: float, field: float, time: float, steps: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Construct a Lie–Trotter circuit for the Heisenberg XXX Hamiltonian with a field."""
    ps_xx, ps_yy, ps_zz, ps_field = _heis_pauli_sums(num_sites, J, field)
    qubits = _sorted_qubits_from_sums(ps_xx, ps_yy, ps_zz, ps_field)
    circuit = cirq.Circuit()
    dt = time / steps

    exp_xx = cirq.PauliSumExponential(ps_xx, exponent=-dt)
    exp_yy = cirq.PauliSumExponential(ps_yy, exponent=-dt)
    exp_zz = cirq.PauliSumExponential(ps_zz, exponent=-dt)
    exp_field = cirq.PauliSumExponential(ps_field, exponent=-dt)
    for _ in range(steps):
        circuit.append(exp_xx)
        circuit.append(exp_yy)
        circuit.append(exp_zz)
        circuit.append(exp_field)
    return circuit, qubits

