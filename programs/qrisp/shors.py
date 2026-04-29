import qrisp
from qrisp import *
from typing import Any, Dict, Sequence
import math
import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational
try:
    # package import (benchmark / python -m ...)
    from .reference_shors import shor_qpe_statevector_small, make_shors
except ImportError:
    # script mode (python programs/qrisp/shors.py)
    from reference_shors import shor_qpe_statevector_small, make_shors
    
from qrisp.simulator import statevector_sim
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def find_order(a, N, t):
    qg = QuantumModulus(N)
    qg[:] = 1
    qpe_res = QuantumFloat(t, exponent=-(t))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a * a) % N
    QFT(qpe_res, inv=True)
    return qpe_res,qg

def get_r_candidates(approx):
        rationals = continued_fraction_convergents(
            continued_fraction_iterator(Rational(approx))
        )
        return [rat.q for rat in rationals]

def shor(N, a):
    meas_res = find_order(a, N).get_measurement()
    # print(meas_res)
    # print(f"number of outcomes: {len(meas_res)}")
    
    r_candidates = sum([get_r_candidates(approx) for approx in meas_res.keys()], [])

    for cand in r_candidates:
        if (a**cand) % N == 1:
            r = cand
            break
    else:
        raise Exception("Please sample again")

    if r % 2:
        raise Exception("Please choose another a")
    
    g = np.gcd(a ** (r // 2) + 1, N)
    return g

def bit_reversal_permutation(arr, bit_range=None):
    """
    Perform a bit-reversal permutation on a 1D numpy array.

    Parameters:
        arr: 1D numpy array (length must be a power of 2)
        bit_range: tuple (low, high) specifying which bits to reverse (inclusive).
                   Defaults to all bits.

    Example:
        arr of length 16 (4 bits: b3 b2 b1 b0)
        bit_range=(0, 1) reverses bits 0-1:  b3 b2 b1 b0 -> b3 b2 b0 b1
        bit_range=(1, 3) reverses bits 1-3:  b3 b2 b1 b0 -> b0 b1 b2 b3 (shifts into position)
    """
    n = len(arr)
    assert n & (n - 1) == 0, "Array length must be a power of 2"
    num_bits = n.bit_length() - 1

    if bit_range is None:
        bit_range = (0, num_bits - 1)

    low, high = bit_range
    assert 0 <= low <= high < num_bits, f"bit_range must be within [0, {num_bits - 1}]"

    indices = np.arange(n, dtype=np.int32)

    # Extract all bits as a (n, num_bits) array, bit 0 = LSB
    bits = (indices[:, None] >> np.arange(num_bits)) & 1

    # Reverse only the bits in [low, high]
    bits_to_reverse = bits[:, low:high+1].copy()
    bits[:, low:high+1] = bits_to_reverse[:, ::-1]

    # Reconstruct indices from modified bits
    reversed_indices = (bits * (1 << np.arange(num_bits))).sum(axis=1)

    return arr[reversed_indices]

def run_simulation(config: Dict[str, Any]):
    """
    Run Shor's algorithm given N and a
    and returns a factor of N
    """
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    m = int(np.ceil(np.log2(N))) 

    qpe_res,qg = find_order(a, N, t) # This returns the QuantumFloat
    qs = qpe_res.qs            # Get the session
    qc = qs.compile()
    full_state = qc.statevector_array()
    dim_target = 1 << m   # Qubits 0-4  (LSBs -> Last Axis)
    dim_count  = 1 << t   # Qubits 5-10 (Middle -> Middle Axis)
    psi_tensor = full_state.reshape((dim_count, dim_target, -1))
    psi_clean_tensor = psi_tensor[:, :, 0]
    psi_final = psi_clean_tensor.reshape(-1)
    return bit_reversal_permutation(psi_final)

if __name__ == "__main__":
    # 1. Setup and Run
    N, a = 21, 2
    t=6
    qpe_res,qg = find_order(a, N,t) # This returns the QuantumFloat
    qs = qpe_res.qs            # Get the session

    qc = qs.compile().to_qiskit()
    print(qc)
    full_state = Statevector(qc).data
    
    # 3. Define Dimensions based on your circuit layout
    dim_target = 1 << 5   # Qubits 0-4  (LSBs -> Last Axis)
    dim_count  = 1 << t   # Qubits 5-10 (Middle -> Middle Axis)
    dim_work   = 1 << 10  # Qubits 11-20 (MSBs -> First Axis)

    # 4. Reshape: (MSB Axis, Middle Axis, LSB Axis)
    #    Layout becomes: (Workspace, Count, Target)
    psi_tensor = full_state.reshape((-1, dim_count, dim_target))

    # 5. Remove Ancilla
    #    We select index 0 of the Workspace axis (Axis 0)
    #    This leaves us with a tensor of shape (dim_count, dim_target)
    psi_clean_tensor = psi_tensor[0, :, :]

    # 6. Flatten to get |Count> (x) |Target>
    #    NumPy flattens row-by-row, so Axis 0 (Count) becomes the "upper" bits
    #    and Axis 1 (Target) becomes the "lower" bits.
    psi_final = psi_clean_tensor.reshape(-1)
    # print(f"psi_final: {psi_final}")
    ref=make_shors(t=6, N=21, a=2)
    # print(f"ref: {ref}")

    psi_tensor = ref.reshape(2**t, -1)
    probs_array = np.sum(np.abs(psi_tensor)**2, axis=1)
    print("Printing ref probs")
    results = {}
    normalization = 1 << t
    for k, p in enumerate(probs_array):
        if p > 1e-9:  # Filter out numerical noise
            phase_val = k / normalization
            results[phase_val] = float(p)
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    print(sorted_results)
    psi_tensor = full_state.reshape((dim_work, dim_count,dim_target))
    probs_array = np.sum(np.abs(psi_tensor)**2, axis=(0,2))
    print("Printing qrisp probs")
    results = {}
    normalization = 1 << t
    for k, p in enumerate(probs_array):
        if p > 1e-9:  # Filter out numerical noise
            phase_val = k / normalization
            results[phase_val] = float(p)
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    print(sorted_results)

    # print(np.sum(np.abs(full_state.reshape((dim_work, dim_count,dim_target)))**2,axis=(0,2)))
    # print(np.sum(np.abs(ref.reshape(dim_count, dim_target))**2,axis=1))

    print(f"Fidelity: {np.abs(np.vdot(psi_final, ref))**2}")

