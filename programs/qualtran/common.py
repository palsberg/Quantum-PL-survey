"""Shared helpers for Qualtran-based simulations."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..common import pauli_models

# Qualtran imports must occur after MPLCONFIGDIR is set to avoid permission warnings.
_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
_MPL_DIR.mkdir(exist_ok=True)

import cirq
from qualtran._infra.gate_with_registers import get_named_qubits, merge_qubits
from qualtran.bloqs.block_encoding import LCUBlockEncoding
from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary
from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
    StatePreparationAliasSampling,
)
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager
from qualtran import Bloq

SelectionData = Tuple[List[cirq.DensePauliString], List[float], float]


def _dense_pauli(pauli: str, coeff: complex) -> cirq.DensePauliString:
    magnitude = abs(coeff)
    if magnitude == 0.0:
        raise ValueError("Coefficient magnitude must be positive.")
    phase = coeff / magnitude
    return cirq.DensePauliString(pauli, coefficient=phase)


def taylor_terms_to_paulis(gamma: Dict[str, complex]) -> SelectionData:
    paulis: List[cirq.DensePauliString] = []
    weights: List[float] = []
    for pauli, coeff in gamma.items():
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
    target_bitsize = len(paulis[0])
    selection_bitsize = max(1, math.ceil(math.log2(len(paulis))))
    select = SelectPauliLCU(
        selection_bitsize=selection_bitsize,
        target_bitsize=target_bitsize,
        select_unitaries=paulis,
    )
    prepare = StatePreparationAliasSampling.from_probabilities(weights, precision=precision)
    return LCUBlockEncoding(select=BlackBoxSelect(select), prepare=BlackBoxPrepare(prepare))


def _simulate_block(block: LCUBlockEncoding) -> Tuple[np.ndarray, List[int]]:
    cbloq = block.decompose_bloq()
    quregs = get_named_qubits(block.signature)
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    circuit, _ = cbloq.to_cirq_circuit_and_quregs(qubit_manager=qm, **quregs)
    circuit = cirq.Circuit(cirq.decompose(circuit))
    qubit_order = merge_qubits(block.signature, **quregs)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    bits_per_register = [reg.total_bits() for reg in block.signature]
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


def apply_ising_step(
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
    bloq: Bloq,
) -> None:
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    op, _ = bloq.as_cirq_op(qm, system=np.array(qubits, dtype=object))
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
