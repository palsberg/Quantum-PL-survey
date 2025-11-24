"""TFIM via a second-order LCU block in Cirq."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import cirq

from . import lcu_common
from ..common import pauli_models


def _build_lcu_circuit(
    num_sites: int,
    weights: List[float],
    paulis: List[str],
    phases: List[str],
) -> Tuple[cirq.Circuit, Sequence[cirq.Qid], Sequence[cirq.Qid], cirq.Qid]:
    """Return the PREPARE†·SELECT·PREPARE circuit and qubits."""
    L = len(weights)
    if L == 0:
        raise ValueError("No LCU terms were generated.")
    m = max(1, int(math.ceil(math.log2(L))))
    target_len = 2**m
    identity = "I" * num_sites

    # Pad to power of two with zero-weight identity placeholders.
    if target_len > L:
        pad = target_len - L
        weights.extend([0.0] * pad)
        paulis.extend([identity] * pad)
        phases.extend(["1"] * pad)

    amps = lcu_common.amps_from_weights(weights)

    system = cirq.LineQubit.range(num_sites)
    index = cirq.LineQubit.range(num_sites, num_sites + m)
    phase = cirq.LineQubit(num_sites + m)

    circuit = cirq.Circuit()
    circuit.append(cirq.X(phase))

    prepare_gate = lcu_common.prepare_gate_from_amplitudes(amps)
    circuit.append(prepare_gate.on(*index))
    controls = list(index)

    for idx, weight in enumerate(weights):
        circuit.append(lcu_common.select_mask_ops(index, idx))
        if weight > 0:
            lcu_common.apply_phase_tag(circuit, controls, phase, phases[idx])
            if paulis[idx] != identity:
                lcu_common.apply_controlled_pauli_string(
                    circuit, controls, system, paulis[idx]
                )
        circuit.append(lcu_common.select_mask_ops(index, idx))

    circuit.append(prepare_gate.on(*index))
    ordered_qubits = list(system) + list(index) + [phase]
    return circuit, ordered_qubits, index, phase


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    """Approximate TFIM evolution via a 2nd-order LCU block."""
    num_sites = int(config["num_sites"])
    time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))

    H = pauli_models.tfim_pauli_terms(num_sites, J, h)
    gamma = pauli_models.taylor_coefficients(H, time)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)

    circuit, ordered_qubits, index_qubits, _ = _build_lcu_circuit(
        num_sites, weights, paulis, phases
    )
    num_index = len(index_qubits)
    return lcu_common.simulate_lcu_block(circuit, ordered_qubits, num_sites, num_index)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "h": 0.7}}
    psi = run_simulation(cfg)
    print(psi)
