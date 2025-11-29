"""Helpers for emitting HML (SimuQ) sketches and delegating execution to backends."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List, Tuple

import numpy as np

from simuq.qsystem import QSystem
from simuq.environment import Qubit
from simuq.hamiltonian import TIHamiltonian
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _resolve_output_path(raw_path: str | None, default_name: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_dir():
        path = path / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_spec(spec: str, params: Dict, default_name: str) -> None:
    path = _resolve_output_path(params.get("hml_spec_path"), default_name)
    if path is None:
        return
    path.write_text(spec)


def render_tfim_trotter_spec(num_sites: int, J: float, h: float, total_time: float, steps: int) -> str:
    dt = total_time / steps
    return dedent(
        f"""\
        # Auto-generated HML / SimuQ sketch for TFIM Trotterization
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        h = {h}
        dt = {dt}
        steps = {steps}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]

        H_ZZ = 0
        for i in range(n - 1):
            H_ZZ += J * (q[i].Z * q[i+1].Z)

        H_X = sum(h * q[i].X for i in range(n))

        for _ in range(steps):
            qs.add_evolution(H_ZZ, dt)
            qs.add_evolution(H_X, dt)

        # Compile qs with your desired AAIS/backend (e.g., QuEra Rydberg or IBM).
        """
    )


def render_heis_trotter_spec(num_sites: int, J: float, field: float, total_time: float, steps: int) -> str:
    dt = total_time / steps
    return dedent(
        f"""\
        # Auto-generated HML / SimuQ sketch for Heisenberg XXX + field
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        field = {field}
        dt = {dt}
        steps = {steps}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]

        H_xx = 0
        H_yy = 0
        H_zz = 0
        H_field = 0
        for i in range(n - 1):
            H_xx += J * (q[i].X * q[i+1].X)
            H_yy += J * (q[i].Y * q[i+1].Y)
            H_zz += J * (q[i].Z * q[i+1].Z)
        for i in range(n):
            H_field += field * q[i].Z

        for _ in range(steps):
            qs.add_evolution(H_xx, dt)
            qs.add_evolution(H_yy, dt)
            qs.add_evolution(H_zz, dt)
            qs.add_evolution(H_field, dt)

        # Compile qs with the appropriate AAIS/backend.
        """
    )


def render_tfim_lcu_spec(num_sites: int, J: float, h: float, total_time: float) -> str:
    return dedent(
        f"""\
        # HML sketch for routing TFIM into an LCU backend
        # The Hamiltonian definition mirrors the SimuQ model; execution uses
        # an external gate-model LCU implementation.
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        h = {h}
        T = {total_time}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]
        H = 0
        for i in range(n - 1):
            H += J * (q[i].Z * q[i+1].Z)
        for i in range(n):
            H += h * q[i].X

        qs.add_evolution(H, T)
        # Forward qs to a gate-model LCU engine for block-encoding.
        """
    )


def render_heis_lcu_spec(num_sites: int, J: float, field: float, total_time: float) -> str:
    return dedent(
        f"""\
        # HML sketch for Heisenberg XXX + field routed to an external LCU backend
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        field = {field}
        T = {total_time}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]
        H = 0
        for i in range(n - 1):
            H += J * (q[i].X * q[i+1].X + q[i].Y * q[i+1].Y + q[i].Z * q[i+1].Z)
        for i in range(n):
            H += field * q[i].Z

        qs.add_evolution(H, T)
        # Use a gate-model LCU implementation to simulate the block-encoding.
        """
    )


def emit_spec(name: str, spec: str, params: Dict) -> None:
    default_file = f"{name}.hml.py"
    _write_spec(spec, params, default_file)


def _ti_to_pauli_terms(h: TIHamiltonian) -> List[Tuple[str, complex]]:
    """Convert a SimuQ TIHamiltonian into Pauli-string terms.

    Returns a list of (pauli_string, coefficient) where pauli_string is over
    the alphabet {I, X, Y, Z} and has length equal to the number of sites.
    """
    n = len(h.sites_type)
    acc: Dict[str, complex] = {}
    for prod, coeff in h.ham:
        if abs(coeff) < 1e-12:
            continue
        # Start with all identity.
        chars = ["I"] * n
        # productHamiltonian maps site index -> single-qubit operator label.
        # For the qubit sites we care about, labels are "", "X", "Y", or "Z".
        for idx, op in prod.to_list():
            if op == "":
                continue
            if op not in ("X", "Y", "Z"):
                # For non-qubit or unsupported ops, skip; not used in this artifact.
                continue
            chars[idx] = op
        pauli = "".join(chars)
        acc[pauli] = acc.get(pauli, 0.0 + 0j) + coeff
    return [(p, c) for p, c in acc.items() if abs(c) >= 1e-12]


def _apply_pauli_evolution(qc: QuantumCircuit, pauli: str, theta: float) -> None:
    """Apply exp(-i * theta * P) for a given Pauli string P."""
    n = len(pauli)
    # Identify non-identity positions.
    targets: List[int] = [i for i, ch in enumerate(pauli) if ch != "I"]
    if not targets:
        # Global phase only; no effect on statevector.
        return

    # Basis change: map X/Y to Z.
    def _basis_change() -> None:
        for i, ch in enumerate(pauli):
            if ch == "X":
                qc.h(i)
            elif ch == "Y":
                qc.sdg(i)
                qc.h(i)

    def _basis_undo() -> None:
        for i, ch in enumerate(pauli):
            if ch == "X":
                qc.h(i)
            elif ch == "Y":
                qc.h(i)
                qc.s(i)

    _basis_change()
    if len(targets) == 1:
        # Single-qubit Z rotation: exp(-i theta Z) = RZ(2 theta) up to global phase.
        q = targets[0]
        qc.rz(2.0 * theta, q)
    else:
        # Multi-qubit ZZ...Z via CNOT chain and a single RZ on the last qubit.
        last = targets[-1]
        for q in targets[:-1]:
            qc.cx(q, last)
        qc.rz(2.0 * theta, last)
        for q in reversed(targets[:-1]):
            qc.cx(q, last)
    _basis_undo()


def simulate_qsystem_with_qiskit(qs: QSystem, num_sites: int) -> np.ndarray:
    """Compile a SimuQ QSystem to a Qiskit circuit and return a statevector.

    We interpret each evolution segment (H, t) in `qs.evos` as exp(-i H t) and
    approximate it using a first-order product formula over the Pauli terms
    of H. The resulting QuantumCircuit is simulated with Qiskit's statevector
    simulator, and we return the amplitudes over the logical spin-chain
    qubits only (no ancillas are introduced in this construction).
    """
    qc = qsystem_to_qiskit_circuit(qs, num_sites)
    state = Statevector.from_instruction(qc).data
    return np.asarray(state, dtype=np.complex128)


def qsystem_to_qiskit_circuit(qs: QSystem, num_sites: int) -> QuantumCircuit:
    """Build a Qiskit QuantumCircuit approximating the QSystem evolution."""
    qc = QuantumCircuit(num_sites)
    # Start in |0...0>, matching the harness reference initial state.
    for h_seg, t_seg in qs.evos:
        if not isinstance(h_seg, TIHamiltonian):
            continue
        for pauli, coeff in _ti_to_pauli_terms(h_seg):
            theta = float(coeff.real) * float(t_seg)
            if abs(theta) < 1e-12:
                continue
            _apply_pauli_evolution(qc, pauli, theta)
    return qc
