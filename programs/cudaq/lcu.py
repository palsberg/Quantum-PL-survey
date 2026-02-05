import cudaq
from cudaq import spin
import numpy as np
import time
import sys
import math
from typing import List, Dict, Tuple

# Paulis: I = 0, X = 1, Y = 2, Z = 3
# Phases: 1 = 0, i = 1, -1 = 2, -i = 3

def create_hamiltonian_tfim(n_sites: int, coupling: float, field: float) -> cudaq.SpinOperator:
    """Create the TFIM Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.x(i)
    return H

def create_hamiltonian_heis(n_sites: int, coupling: float, field: float) -> cudaq.SpinOperator:
    """Create the Heisenberg Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.x(i) * spin.x(i + 1)
        H += coupling * spin.y(i) * spin.y(i + 1)
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.z(i)
    return H

def get_taylor_coeffs(H: cudaq.SpinOperator) -> Dict[str, complex]:
    taylor_H = spin.i(0) - (1j * H * t) - (H * H * t**2 / 2)
    coeffs = dict()
    for x in taylor_H:
        x = x.canonicalize(set(range(H.qubit_count)))
        coeff = x.evaluate_coefficient()
        word = x.get_pauli_word()
        coeffs[word] = coeffs.get(word, 0) + coeff
    return coeffs

def get_lcu_weights(coeffs: Dict[str, complex]) -> Tuple[List[float], List[str], List[str]]:
    weights: List[float] = []
    paulis: List[str] = []
    phases: List[str] = []

    for P, coeff in coeffs.items():
        if abs(coeff) < 1e-12:
            continue
        real = coeff.real
        imag = coeff.imag

        if real > 0:
            weights.append(real)
            paulis.append(P)
            phases.append("1")
        elif real < 0:
            weights.append(-real)
            paulis.append(P)
            phases.append("-1")

        if imag > 0:
            weights.append(imag)
            paulis.append(P)
            phases.append("i")
        elif imag < 0:
            weights.append(-imag)
            paulis.append(P)
            phases.append("-i")

    return weights, paulis, phases


def get_rotation_angles(amps: List[float]) -> List[float]:
    amps = amps / np.linalg.norm(amps)
    n = math.ceil(np.log2(len(amps)))
    assert len(amps) == 2**n

    angles = []

    for k in range(n):
        s = 2 ** (n - k - 1)
        for c in range(2 ** k):
            offset = c * 2*s
            amps0    = amps[offset     : offset + s]
            amps1    = amps[offset + s : offset + 2*s]
            amps_tot = amps[offset     : offset + 2*s]

            norm_tot = np.linalg.norm(amps_tot)
            if norm_tot == 0:
                angles.append(0.0)
                continue

            amps0_normalized = amps0 / norm_tot
            amps1_normalized = amps1 / norm_tot

            p0 = np.linalg.norm(amps0_normalized) ** 2
            p1 = np.linalg.norm(amps1_normalized) ** 2
            assert np.isclose(p0 + p1, 1.0)

            theta = 2 * np.arccos(np.sqrt(p0))
            angles.append(float(theta))

    assert len(angles) == 2**n - 1
    return angles


@cudaq.kernel
def apply_controlled_pauli_string(
        operand: cudaq.qview, control: cudaq.qview,
        pauli: List[int], phase: int, cond: int):
    n_ctrl = control.size()

    # Flip controlled-on-zero qubits
    for i in range(n_ctrl):
        if ((cond >> i) & 1) == 0:
            x(control[i])

    # Apply phase
    # phase_qubit = operand[operand.size() - 1]
    phase_qubit = operand.back()
    if phase == 0:
        pass
    elif phase == 1:
        s.ctrl(control, phase_qubit)
    elif phase == 2:
        z.ctrl(control, phase_qubit)
    elif phase == 3:
        s.ctrl(control, phase_qubit)
        s.ctrl(control, phase_qubit)
        s.ctrl(control, phase_qubit)

    # Apply Pauli string
    for i in range(len(pauli)):
        if pauli[i] == 0:
            pass
        elif pauli[i] == 1:
            x.ctrl(control, operand[i])
        elif pauli[i] == 2:
            y.ctrl(control, operand[i])
        elif pauli[i] == 3:
            z.ctrl(control, operand[i])

    # Unflip controlled-on-zero qubits
    for i in range(n_ctrl):
        if ((cond >> i) & 1) == 0:
            x(control[i])


@cudaq.kernel
def prepare(qs: cudaq.qview, angles: List[float]):
    # Get angles from get_rotation_angles()
    n = qs.size()
    i = 0
    for k in range(n):
        for c in range(2 ** k):
            # Flip controlled-on-zero qubits
            for j in range(k):
                if ((c >> j) & 1) == 0:
                    x(qs[n - k + j])

            if k == 0:
                ry(angles[i], qs[n-1])
            else:
                ry.ctrl(angles[i], qs[n-k:n], qs[n-k-1])

            # Unflip controlled-on-zero qubits
            for j in range(k):
                if ((c >> j) & 1) == 0:
                    x(qs[n - k + j])

            i += 1

### Using cudaq.adjoint gives us lower fidelity than just manually defining unprepare()
# @cudaq.kernel
# def unprepare(qs: cudaq.qview, angles: List[float]):
#     cudaq.adjoint(prepare, qs, angles)

@cudaq.kernel
def unprepare(qs: cudaq.qview, angles: List[float]):
    # Use the same angles as in prepare(); they should come from get_rotation_angles()
    n = qs.size()
    i = len(angles) - 1
    for k in range(n-1, -1, -1):
        for c in range(2**k - 1, -1, -1):
            # Flip controlled-on-zero qubits
            for j in range(k):
                if ((c >> j) & 1) == 0:
                    x(qs[n - k + j])

            if k == 0:
                ry(-1 * angles[i], qs[n-1])
            else:
                ry.ctrl(-1 * angles[i], qs[n-k:n], qs[n-k-1])

            # Unflip controlled-on-zero qubits
            for j in range(k):
                if ((c >> j) & 1) == 0:
                    x(qs[n - k + j])

            i -= 1

@cudaq.kernel
def select_unitary(operand: cudaq.qview, ancilla: cudaq.qview,
                  paulis: List[List[int]], phases: List[int]):
    for i in range(len(paulis)):
        apply_controlled_pauli_string(operand, ancilla, paulis[i], phases[i], i)


@cudaq.kernel
def lcu_circuit(n_operand: int, n_anc: int, angles: List[float],
                paulis: List[List[int]], phases: List[int]):
    operand = cudaq.qvector(n_operand + 1)
    ancilla = cudaq.qvector(n_anc)

    while True:
        reset(operand)
        reset(ancilla)

        x(operand[n_operand])  # phase qubit

        ### Ideally, compute_action() undoes prepare() automatically. But it does not
        ### compile, so we use a custom unprepare() kernel.
        # cudaq.compute_action(lambda: prepare(ancilla, angles),
        #                      lambda: lcu_selection(operand, ancilla, paulis, phases))
        prepare(ancilla, angles)
        select_unitary(operand, ancilla, paulis, phases)
        unprepare(ancilla, angles)

        x(operand[n_operand])

        # Measure and repeat unless ancilla == 0
        ancilla_measurement = mz(ancilla)
        success = True
        for i in range(len(ancilla_measurement)):
            if ancilla_measurement[i]:
                success = False
                break
        if success:
            break


def lcu(n_sites: int, hamiltonian: cudaq.SpinOperator):
    taylor_coeffs = get_taylor_coeffs(hamiltonian)
    weights, paulis, phases = get_lcu_weights(taylor_coeffs)

    # weights -> amplitudes for lcu ancilla state preparation
    amps = [math.sqrt(weight / sum(weights)) for weight in weights]
    n_ancilla = math.ceil(math.log2(len(amps)))
    amps += [0.0] * (2**n_ancilla - len(amps))

    # cudaq kernels only supports numeric types, so we map strings to ints
    pauli_map = {'I':0, 'X':1, 'Y':2, 'Z':3}
    phase_map = {'1':0, 'i':1, '-1':2, '-i':3}
    paulis = [[pauli_map[p] for p in pauli] for pauli in paulis]
    phases = [phase_map[p] for p in phases]

    angles = get_rotation_angles(amps)

    state = cudaq.get_state(lcu_circuit, n_sites, n_ancilla, angles, paulis, phases)
    return np.array(state)




# Parameters
n_sites = 3
ham_type = "tfim"
coupling = 2.0
field = 0.5
t = 1.0

# Create Hamiltonian
if ham_type == "tfim":
    H = create_hamiltonian_tfim(n_sites, coupling, field)
elif ham_type == "heis":
    H = create_hamiltonian_heis(n_sites, coupling, field)
else:
    assert False

state = lcu(n_sites, H)
state = state[:2**n_sites]
state = state / np.linalg.norm(state)



# Ground truth
from reference_hamiltonians import *
if ham_type == 'tfim':
    ref_H = tfim_hamiltonian(n_sites, coupling, field)
elif ham_type == 'heis':
    ref_H = heis_xxx_hamiltonian(n_sites, coupling, field)
ref_state = time_evolve(ref_H, zero_state(n_sites), t)

def compute_fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    state = normalize(state)
    reference = normalize(reference)
    return float(np.abs(np.vdot(reference, state)) ** 2)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128).flatten()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return vec / norm

fidelity = compute_fidelity(state, ref_state)
print(f'{fidelity=}')
