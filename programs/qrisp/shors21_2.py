from qrisp import *
from typing import Any, Dict, Sequence
import math
import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational
from reference_shors import shor_qpe_statevector_small, make_shors

def find_order(a, N):
    qg = QuantumModulus(N)
    qg[:] = 1
    qpe_res = QuantumFloat(6, exponent=-(6))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a * a) % N
    QFT(qpe_res, inv=True)
    return qpe_res

def get_r_candidates(approx):
        rationals = continued_fraction_convergents(
            continued_fraction_iterator(Rational(approx))
        )
        return [rat.q for rat in rationals]

def shor(N, a):
    meas_res = find_order(a, N).get_measurement()
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

    statevector=find_order(a, N).qs.statevector(return_type="array")
    psi = statevector.reshape(2**(t+m), -1)
    state = np.sum(psi, axis=1)
    return state

if __name__ == "__main__":
    print(shor(21,2))
    qs=find_order(2, 21).qs
    # print(qs)
    ref=make_shors(t=6, N=21, a=2)
    mine=qs.statevector(return_type="array")
    # total_dim = ref.size
    # dim_prefix = 1 << 5      # Dimension of qubits BEFORE the window
    # dim_target = 1 << 6       # Dimension of the window itself
    # dim_suffix = total_dim // (dim_prefix * dim_target) # Dimension of qubits AFTER
    
    # 1. Reshape to (Prefix, Target, Suffix)
    #    Axis 0: Qubits 0 to start-1
    #    Axis 1: Qubits start to start+num-1 (The ones we want)
    #    Axis 2: Qubits start+num to end
    psi_tensor = ref.reshape(2**6, -1)
    psi_tens_act = np.sum(psi_tensor, axis=1)
    
    # 2. Marginalize (Trace out Prefix and Suffix)
    #    Sum |amplitude|^2 over axis 0 and axis 2
    probs_array = np.sum(np.abs(psi_tensor)**2, axis=1)
    
    psi=mine.reshape(2**6, -1)
    psi_act=np.sum(psi, axis=1)

    probs_mine = np.sum(np.abs(psi)**2, axis=1)
    print(f"Fidelity: {np.abs(np.vdot(psi_tens_act, psi_act))**2}")
    # 3. Create dictionary mapping phase (float) -> probability
    #    Phase = integer_outcome / 2^t
    #    We filter out near-zero probabilities for cleaner output
    results = {}
    normalization = 1 << 6
    
    for k, p in enumerate(probs_array):
        if p > 1e-9:  # Filter out numerical noise
            phase_val = k / normalization
            results[phase_val] = float(p)
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    print(f"make_shor state results: {sorted_results}")