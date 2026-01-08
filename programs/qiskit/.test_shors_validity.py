import numpy as np
import shors
import harness.reference_shors as reference_shors

def verify():
    # Parameters used in reference test
    t, N, a = 6, 21, 2
    
    # 1. Get Reference State
    ref_sv = reference_shors.shor_qpe_statevector_small(t, N, a)
    
    # 2. Get Qiskit State
    qiskit_sv = shors.run_simulation({"t": t, "N": N, "a": a})
    
    # 3. Calculate Fidelity
    fidelity = np.abs(np.vdot(ref_sv, qiskit_sv))**2
    print(f"Fidelity: {fidelity:.10f}") # Should be 1.0000000000

if __name__ == "__main__":
    verify()