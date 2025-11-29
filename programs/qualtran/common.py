"""Shared helpers for Qualtran-based simulations."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Optional

import attrs
from functools import cached_property
import numpy as np
from numpy.typing import NDArray

from ..common import pauli_models

# Qualtran imports must occur after MPLCONFIGDIR is set to avoid permission warnings.
_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
_MPL_DIR.mkdir(exist_ok=True)

import cirq
from qualtran import AddControlledT, Bloq, BQUInt, CtrlSpec, QAny, QBit, Register, Signature
from qualtran._infra.gate_with_registers import get_named_qubits, merge_qubits
from qualtran.bloqs.block_encoding import LCUBlockEncoding
from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Rx, Rz
from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
from qualtran.bloqs.multiplexers.select_base import SelectOracle
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
    StatePreparationAliasSampling,
)
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager

SelectionData = Tuple[List[cirq.DensePauliString], List[float], float]


def _dense_pauli(pauli: str, coeff: complex) -> cirq.DensePauliString:
    magnitude = abs(coeff)
    if magnitude == 0.0:
        raise ValueError("Coefficient magnitude must be positive.")
    phase = coeff / magnitude
    return cirq.DensePauliString(pauli, coefficient=phase)


def _to_tuple_dense(xs: Iterable[cirq.DensePauliString]) -> Tuple[cirq.DensePauliString, ...]:
    """attrs converter for TaylorSelectPauliLCU.select_unitaries."""
    return tuple(xs)


@attrs.frozen
class TaylorSelectPauliLCU(SelectOracle, UnaryIterationGate):  # type: ignore[misc]
    """Project-local SELECT oracle for LCU over DensePauliStrings.

    This mirrors qualtran.bloqs.multiplexers.select_pauli_lcu.SelectPauliLCU but
    keeps the full complex phase of each DensePauliString instead of truncating
    to sign(real(coeff)). This avoids introducing 0·Pauli channels for purely
    imaginary coefficients that arise in the 2nd-order Taylor LCU construction.
    """

    selection_bitsize: int
    target_bitsize: int
    select_unitaries: Tuple[cirq.DensePauliString, ...] = attrs.field(converter=_to_tuple_dense)
    control_val: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        if any(len(dps) != self.target_bitsize for dps in self.select_unitaries):
            raise ValueError(
                f"Each dense pauli string in {self.select_unitaries} should contain "
                f"{self.target_bitsize} terms."
            )
        min_bitsize = (len(self.select_unitaries) - 1).bit_length()
        if self.selection_bitsize < min_bitsize:
            raise ValueError(
                f"selection_bitsize={self.selection_bitsize} should be at-least {min_bitsize}"
            )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register("control", QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register("selection", BQUInt(self.selection_bitsize, len(self.select_unitaries))),)

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("target", QAny(self.target_bitsize)),)

    def decompose_from_registers(
        self, context, **quregs: "NDArray[cirq.Qid]"  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        if self.control_val == 0:
            yield cirq.X(*quregs["control"])
        # Delegate unary-iteration-based decomposition to the parent class.
        yield from super(TaylorSelectPauliLCU, self).decompose_from_registers(
            context=context, **quregs
        )
        if self.control_val == 0:
            yield cirq.X(*quregs["control"])

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        selection: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        """Applies select_unitaries[selection] with its full complex phase.

        Keeping the original DensePauliString coefficient ensures the gate
        remains unitary even when the phase is ±i, and avoids 0·Pauli channels.
        """
        ps = self.select_unitaries[selection].on(*target)
        return ps.controlled_by(control)

    def get_ctrl_system(self, ctrl_spec: "CtrlSpec") -> Tuple["Bloq", "AddControlledT"]:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv

        return get_ctrl_system_1bit_cv(
            self,
            ctrl_spec=ctrl_spec,
            current_ctrl_bit=self.control_val,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (
                attrs.evolve(self, control_val=cv),
                "control",
            ),
        )

    def adjoint(self) -> "Bloq":
        return self

    def _has_unitary_(self) -> bool:
        return True


def taylor_terms_to_paulis(gamma: Dict[str, complex]) -> SelectionData:
    """Combine Taylor coefficients into unique Pauli terms with nonzero magnitude."""
    # Combine duplicate Pauli strings before building DensePauliStrings to avoid
    # branches corresponding to a net zero operator.
    combined: Dict[str, complex] = {}
    for pauli, coeff in gamma.items():
        combined[pauli] = combined.get(pauli, 0.0 + 0j) + coeff

    paulis: List[cirq.DensePauliString] = []
    weights: List[float] = []
    for pauli, coeff in combined.items():
        magnitude = abs(coeff)
        if magnitude < 1e-12:
            continue
        paulis.append(_dense_pauli(pauli, coeff))
        weights.append(magnitude)
    if not paulis:
        raise ValueError("No non-zero coefficients produced for LCU block.")
    total_weight = float(sum(weights))
    return paulis, weights, total_weight


def build_lcu_block(
    paulis: List[cirq.DensePauliString],
    weights: List[float],
    *,
    precision: float,
) -> LCUBlockEncoding:
    """Build an LCU block encoding using a project-local SELECT oracle.

    This mirrors Qualtran's standard LCUBlockEncoding construction but uses
    TaylorSelectPauliLCU instead of SelectPauliLCU, so we can safely handle
    complex Pauli coefficients from the 2nd-order Taylor expansion.
    """
    target_bitsize = len(paulis[0])
    selection_bitsize = max(1, math.ceil(math.log2(len(paulis))))
    select = TaylorSelectPauliLCU(
        selection_bitsize=selection_bitsize,
        target_bitsize=target_bitsize,
        select_unitaries=paulis,
    )
    prepare = StatePreparationAliasSampling.from_probabilities(weights, precision=precision)
    return LCUBlockEncoding(select=BlackBoxSelect(select), prepare=BlackBoxPrepare(prepare))


def _simulate_block(block: LCUBlockEncoding) -> Tuple[np.ndarray, List[int]]:
    cbloq = block.decompose_bloq()
    init_quregs = get_named_qubits(block.signature)
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    circuit, quregs_out = cbloq.to_cirq_circuit_and_quregs(qubit_manager=qm, **init_quregs)

    # Qubits corresponding to the named registers in the block signature.
    sig_qubits = merge_qubits(block.signature, **quregs_out)
    sig_set = set(sig_qubits)

    # Include all additional ancillas introduced during decomposition.
    extra_qubits = sorted([q for q in circuit.all_qubits() if q not in sig_set], key=lambda q: str(q))
    qubit_order = list(sig_qubits) + extra_qubits

    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    bits_per_register = [reg.total_bits() for reg in block.signature]
    if extra_qubits:
        # Treat remaining qubits as a single extra ancilla register projected to |0...0⟩.
        bits_per_register.append(len(extra_qubits))
    return np.asarray(result.final_state_vector, dtype=np.complex128), bits_per_register


def _extract_system_state(state: np.ndarray, bits_per_register: List[int], system_index: int) -> np.ndarray:
    shape = [1 << bits for bits in bits_per_register]
    reshaped = state.reshape(shape)
    indexer = [0] * len(bits_per_register)
    indexer[system_index] = slice(None)
    vec = reshaped[tuple(indexer)].reshape(-1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("LCU block returned zero amplitude on the |0...0> ancilla subspace.")
    return vec / norm


def simulate_lcu_state(block: LCUBlockEncoding, num_system: int) -> np.ndarray:
    state, bits_per_register = _simulate_block(block)
    reg_names = [reg.name for reg in block.signature]
    system_index = reg_names.index("system")
    vec = _extract_system_state(state, bits_per_register, system_index)
    if vec.size != 2**num_system:
        raise ValueError("Unexpected system register size in Qualtran block.")
    return vec


def tfim_lcu_state(num_sites: int, J: float, h: float, time: float, precision: float) -> np.ndarray:
    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, time)
    paulis, weights, alpha = taylor_terms_to_paulis(gamma)
    block = build_lcu_block(paulis, weights, precision=precision)
    state = simulate_lcu_state(block, num_sites)
    return state * alpha / np.linalg.norm(state * alpha)


def heis_lcu_state(num_sites: int, J: float, field: float, time: float, precision: float) -> np.ndarray:
    H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
    gamma = pauli_models.taylor_coefficients(H, time)
    paulis, weights, alpha = taylor_terms_to_paulis(gamma)
    block = build_lcu_block(paulis, weights, precision=precision)
    state = simulate_lcu_state(block, num_sites)
    return state * alpha / np.linalg.norm(state * alpha)


@attrs.frozen
class HeisenbergPairUnitary(Bloq):
    """Two-qubit Heisenberg XXX + field unitary for a single Trotter slice.

    Implements, on a pair (i,i+1),
        U = e^{-i J dt X_i X_{i+1}}
            e^{-i J dt Y_i Y_{i+1}}
            e^{-i J dt Z_i Z_{i+1}}
            e^{-i B dt (Z_i + Z_{i+1})},
    using basis changes around ZZ rotations and single-qubit Rz gates.
    """

    angle_j: float  # J * dt
    angle_field: float  # B * dt

    @cached_property
    def signature(self) -> Signature:
        # Two-qubit system register.
        return Signature.build(system=2)

    def build_composite_bloq(self, bb: "BloqBuilder", system: "Soquet") -> Dict[str, "Soquet"]:
        system = bb.split(system)
        q0, q1 = system

        j = self.angle_j
        b = self.angle_field

        # Helper: ZZ rotation via CNOT–Rz–CNOT.
        def zz_rot(q0_in, q1_in, theta: float):
            q0_out, q1_out = bb.add(CNOT(), ctrl=q0_in, target=q1_in)
            q1_out = bb.add(Rz(2.0 * theta), q=q1_out)
            q0_out, q1_out = bb.add(CNOT(), ctrl=q0_out, target=q1_out)
            return q0_out, q1_out

        # e^{-i J dt X X}  via H⊗H • ZZ( J dt ) • H⊗H
        q0 = bb.add(Hadamard(), q=q0)
        q1 = bb.add(Hadamard(), q=q1)
        q0, q1 = zz_rot(q0, q1, j)
        q0 = bb.add(Hadamard(), q=q0)
        q1 = bb.add(Hadamard(), q=q1)

        # e^{-i J dt Y Y} via Rx(-π/2)⊗Rx(-π/2) • ZZ( J dt ) • Rx(π/2)⊗Rx(π/2)
        half_pi = math.pi / 2.0
        q0 = bb.add(Rx(-half_pi), q=q0)
        q1 = bb.add(Rx(-half_pi), q=q1)
        q0, q1 = zz_rot(q0, q1, j)
        q0 = bb.add(Rx(half_pi), q=q0)
        q1 = bb.add(Rx(half_pi), q=q1)

        # e^{-i J dt Z Z}
        q0, q1 = zz_rot(q0, q1, j)

        # Field term e^{-i B dt (Z_i + Z_{i+1})} via single-qubit Rz rotations.
        q0 = bb.add(Rz(2.0 * b), q=q0)
        q1 = bb.add(Rz(2.0 * b), q=q1)

        system = [q0, q1]
        return {"system": bb.join(system)}


def apply_ising_step(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
    bloq: Bloq,
) -> None:
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    op, _ = bloq.as_cirq_op(qm, system=np.array(qubits, dtype=object))
    circuit.append(op)


def apply_heisenberg_pair_step(
    circuit: cirq.Circuit,
    q_left: cirq.Qid,
    q_right: cirq.Qid,
    bloq: Bloq,
) -> None:
    """Apply a two-qubit HeisenbergPairUnitary bloq to a given pair in a Cirq circuit."""
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    op, _ = bloq.as_cirq_op(qm, system=np.array([q_left, q_right], dtype=object))
    circuit.append(op)


def tfim_trotter_state(
    num_sites: int,
    J: float,
    h: float,
    total_time: float,
    steps: int,
    order: int,
    init_angle: float,
) -> np.ndarray:
    if order != 1:
        raise ValueError("Qualtran TFIM Trotter currently supports only first-order evolution.")
    qubits = cirq.LineQubit.range(num_sites)
    circuit = cirq.Circuit()
    dt = total_time / steps
    # Small tilt away from |0...0>
    for q in qubits:
        circuit.append(cirq.ry(init_angle).on(q))
    for _ in range(steps):
        apply_ising_step(circuit, qubits, IsingZZUnitary(nsites=num_sites, angle=2 * J * dt))
        apply_ising_step(circuit, qubits, IsingXUnitary(nsites=num_sites, angle=2 * h * dt))
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def heis_trotter_state(
    num_sites: int,
    J: float,
    field: float,
    total_time: float,
    steps: int,
    init_angle: float,
) -> np.ndarray:
    """First-order Trotterization of the Heisenberg XXX chain using HeisenbergPairUnitary."""
    qubits = cirq.LineQubit.range(num_sites)
    circuit = cirq.Circuit()
    dt = total_time / steps
    # Small tilt away from |0...0> to seed dynamics.
    for q in qubits:
        circuit.append(cirq.ry(init_angle).on(q))
    for _ in range(steps):
        pair_bloq = HeisenbergPairUnitary(angle_j=J * dt, angle_field=field * dt)
        for i in range(num_sites - 1):
            apply_heisenberg_pair_step(circuit, qubits[i], qubits[i + 1], pair_bloq)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)
