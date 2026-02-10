from qrisp import *
from typing import Any, Dict, Sequence
import math
import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational
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

def shor(N, a, t):
    meas_res = find_order(a, N, t)[0].get_measurement()
    print(meas_res)
    print(f"number of outcomes: {len(meas_res)}")
    
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

def run_simulation(config: Dict[str, Any]):
    """
    Run Shor's algorithm given N and a
    and returns a factor of N
    """
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    m = int(np.ceil(np.log2(N))) 
    g=shor(N, a, t)
    return np.array([g])

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

