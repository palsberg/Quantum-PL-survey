from typing import Dict

from qrisp import QuantumVariable, inner_LCU, prepare, x,y,z, gphase
import numpy as np

from ..common import pauli_models
from .common import _state_from_qv


def substate(state: np.ndarray, nqubits: int) -> np.ndarray:
    # Take a subset of statevector including only the last nqubits, assuming the
    # preceding qubits are zero.
    new_state = state[::state.shape[0]//(2**nqubits)]

    # Normalize
    norm = np.sqrt(np.sum(np.abs(new_state ** 2))).item()
    return new_state / norm;


def lcu(num_sites: int, H: Dict[str, complex], t: float) -> np.ndarray:
    gamma = pauli_models.taylor_coefficients(H, t)
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)

    unitaries = []
    for pauli_string, phase in zip(paulis, phases):
        def U(qubits, pauli_string=pauli_string, phase=phase):
            match phase:
                case "1":  pass
                case "-1": gphase(np.pi, qubits[0])
                case "i":  gphase(np.pi/2, qubits[0])
                case "-i": gphase(-np.pi/2, qubits[0])
            for i, P in enumerate(pauli_string):
                match P:
                    case "I": pass
                    case "X": x(qubits[i])
                    case "Y": y(qubits[i])
                    case "Z": z(qubits[i])
        unitaries.append(U)

    def operand_prep():
        operand = QuantumVariable(num_sites)
        return operand

    def state_prep(case):
        total = float(np.sum(weights))
        amps = np.sqrt(np.array(weights) / total)
        prepare(case, amps)


    ancilla, qv = inner_LCU(operand_prep, state_prep, unitaries)
    state = _state_from_qv(qv)

    state = substate(state, qv.size)
    return state
