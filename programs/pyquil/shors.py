from pyquil import Program, get_qc
from pyquil.gates import H, CNOT, SWAP, X, MEASURE, Gate
from pyquil.quilbase import DefGate
from pyquil.simulation import NumpyWavefunctionSimulator
import numpy as np
from .reference_shors import shor_qpe_statevector_small, make_shors
from math import gcd
from fractions import Fraction
import random
from typing import Any, Dict, Sequence

def _qubit_indices(inst):
    """Extract qubit indices from an instruction."""
    return [q.index for q in inst.qubits]

def simulate_statevector(prog: Program, num_qubits: int) -> np.ndarray:
    simulator = NumpyWavefunctionSimulator(n_qubits=num_qubits)
    custom_defs = {gate.name: gate for gate in prog.defined_gates}
    for inst in prog:
        if not isinstance(inst, Gate):
            continue

        qubits = _qubit_indices(inst)

        if inst.name in custom_defs:
            matrix = np.asarray(custom_defs[inst.name].matrix, dtype=np.complex128)
            simulator.do_gate_matrix(matrix, qubits)
            continue

        if inst.modifiers:
            matrix = np.asarray(inst.to_unitary_mut(len(inst.qubits)), dtype=np.complex128)
            simulator.do_gate_matrix(matrix, qubits)
            continue

        simulator.do_gate(inst)

    wf = simulator.wf.reshape(-1)
    return np.asarray(wf, dtype=np.complex128)

def measure_statevector(statevector: np.ndarray, qubits_to_measure: list, num_shots: int = 1) -> list:
    """
    Classically measure specific qubits from a statevector.
    
    Args:
        statevector: The quantum statevector
        qubits_to_measure: List of qubit indices to measure
        num_shots: Number of measurement shots
    
    Returns:
        List of measurement results (bitstrings)
    """
    n_qubits = int(np.log2(len(statevector)))
    probabilities = np.abs(statevector) ** 2
    
    results = []
    for _ in range(num_shots):
        # Sample a basis state according to probabilities
        outcome = np.random.choice(len(statevector), p=probabilities)
        
        # Convert outcome to binary and extract measured qubits
        bitstring = format(outcome, f'0{n_qubits}b')
        measured_bits = ''.join([bitstring[n_qubits - 1 - q] for q in qubits_to_measure])
        results.append(int(measured_bits, 2))
    
    return results

def qft(qubits):
    """Quantum Fourier Transform on a list of qubits."""
    p = Program()
    n = len(qubits)
    
    for i in range(n):
        p += H(qubits[i])
        for j in range(i + 1, n):
            # Controlled rotation
            angle = np.pi / (2 ** (j - i))
            crot = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * angle)]
            ])
            crot_def = DefGate("CROT{}".format(j-i), crot)
            p += crot_def
            p += ("CROT{}".format(j-i), qubits[j], qubits[i])
    
    # Swap qubits for bit reversal
    for i in range(n // 2):
        p += SWAP(qubits[i], qubits[n - i - 1])
    
    return p

def inverse_qft(qubits):
    """Inverse Quantum Fourier Transform."""
    p = Program()
    n = len(qubits)
    
    # Reverse swaps
    for i in range(n // 2):
        p += SWAP(qubits[i], qubits[n - i - 1])
    
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            # Controlled rotation (negative angle for inverse)
            angle = -np.pi / (2 ** (j - i))
            crot = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * angle)]
            ])
            crot_def = DefGate("ICROT{}".format(j-i), crot)
            p += crot_def
            p += ("ICROT{}".format(j-i), qubits[j], qubits[i])
        p += H(qubits[i])
    
    return p

def mul_mod_N_gate(a: int, N: int, m: int):
    """
    Build the unitary |x> -> |(a*x) mod N> for x < N, and |x> -> |x> for x >= N,
    on m qubits (dimension 2^m).
    This is a permutation matrix.
    
    Args:
        a: multiplier
        N: modulus
        m: number of qubits
    
    Returns:
        DefGate object and gate name
    """
    dim = 1 << m  # 2^m
    U = np.zeros((dim, dim), dtype=complex)
    
    for x in range(dim):
        y = (a * x) % N if x < N else x
        U[y, x] = 1.0
    
    gate_name = f"MUL_{a}_MOD_{N}"
    gate_def = DefGate(gate_name, U)
    
    return gate_def, gate_name

def controlled_mul_mod_N(p, control, target_qubits, a, N):
    """
    Apply controlled modular multiplication gate.
    |control⟩|x⟩ -> |control⟩|(a*x mod N)⟩ if control=1, else |control⟩|x⟩
    
    Args:
        p: Program to add gates to
        control: control qubit
        target_qubits: list of target qubits
        a: multiplier
        N: modulus
    """
    m = len(target_qubits)
    
    # Create the multiplication gate
    gate_def, gate_name = mul_mod_N_gate(a, N, m)
    p += gate_def
    
    # Apply as controlled gate
    # In PyQuil, we need to create a controlled version
    # For simplicity, we create the full controlled unitary
    dim_target = 1 << m
    dim_total = 2 * dim_target
    
    # Build controlled version: control ⊗ target
    U_controlled = np.zeros((dim_total, dim_total), dtype=complex)
    
    # When control = 0: identity on target
    for i in range(dim_target):
        U_controlled[i, i] = 1.0
    
    # When control = 1: apply multiplication gate
    U_mult = np.zeros((dim_target, dim_target), dtype=complex)
    for x in range(dim_target):
        y = (a * x) % N if x < N else x
        U_mult[y, x] = 1.0
    
    for i in range(dim_target):
        for j in range(dim_target):
            U_controlled[dim_target + i, dim_target + j] = U_mult[i, j]
    
    controlled_gate_name = f"C{gate_name}"
    controlled_gate_def = DefGate(controlled_gate_name, U_controlled)
    p += controlled_gate_def
    
    # Apply to control + target qubits
    all_qubits = [control] + target_qubits
    p += (controlled_gate_name, *all_qubits)
    
    return p

def modular_exponentiation(a, N, count_qubits, target_qubits):
    """
    Create a quantum circuit for modular exponentiation: |x⟩|1⟩ -> |x⟩|a^x mod N⟩
    Implements controlled-U^(2^j) gates where U|y⟩ = |(a*y) mod N⟩
    
    For each counting qubit j, applies controlled multiplication by a^(2^j) mod N.
    """
    p = Program()
    n_count = len(count_qubits)
    
    # Apply controlled modular multiplication gates
    # For each qubit j in the counting register, apply controlled-U^(2^j)
    for j in range(n_count):
        # Calculate a^(2^j) mod N classically
        a_power = pow(a, 2**j, N)
        control = count_qubits[j]
        
        # Apply controlled modular multiplication by a^(2^j) mod N
        controlled_mul_mod_N(p, control, target_qubits, a_power, N)
    
    return p

def shors_algorithm(N, a=None):
    """
    Shor's algorithm for factoring N.
    
    Args:
        N: The number to factor
        a: Base for modular exponentiation (chosen randomly if not provided)
    
    Returns:
        A quantum program (without measurements) for Shor's algorithm
    """
    # Classical pre-processing
    if N % 2 == 0:
        return None, 2
    
    # Choose random a if not provided
    if a is None:
        a = random.randint(2, N - 1)
    
    # Check if a and N are coprime
    g = gcd(a, N)
    if g != 1:
        return None, g
    
    # Determine number of qubits needed
    n_count = int(np.ceil(np.log2(N))) * 2  # Counting qubits
    n_target = int(np.ceil(np.log2(N)))      # Target qubits
    
    # Create quantum program
    p = Program()
    
    # Counting register qubits
    count_qubits = list(range(n_count))
    # Target register qubits
    target_qubits = list(range(n_count, n_count + n_target))
    
    # Initialize counting register to superposition
    for q in count_qubits:
        p += H(q)
    
    # Initialize target register to |1⟩
    p += X(target_qubits[0])
    
    # Apply modular exponentiation
    # Implements controlled-U^(2^j) gates where U|y⟩ = |ay mod N⟩
    p += modular_exponentiation(a, N, count_qubits, target_qubits)
    
    # Apply inverse QFT to counting register
    p += inverse_qft(count_qubits)
    
    return p, (n_count, n_target, count_qubits, target_qubits, a)

def classical_postprocessing(measured_value, a, N, n_count):
    """
    Classical post-processing to extract factors from measurement.
    
    Args:
        measured_value: The measured value from the quantum circuit
        a: The base used in modular exponentiation
        N: The number to factor
        n_count: Number of counting qubits
    
    Returns:
        A factor of N, or None if unsuccessful
    """
    if measured_value == 0:
        return None
    
    # Use continued fractions to find period r
    phase = measured_value / (2 ** n_count)
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    
    # Check if r is even and a^(r/2) != -1 mod N
    if r % 2 == 0:
        x = pow(a, r // 2, N)
        if x != N - 1:
            factor1 = gcd(x + 1, N)
            factor2 = gcd(x - 1, N)
            
            if factor1 != 1 and factor1 != N:
                return factor1
            if factor2 != 1 and factor2 != N:
                return factor2
    
    return None

# def test_controlled_mul_mod():
#     N = 15  # Number to factor
#     a = 7   # Base (coprime to N)
#     control=0
#     n_count = 1  # Counting qubits
#     n_target = int(np.ceil(np.log2(N)))      # Target qubits
#     target_qubits = list(range(n_count, n_count + n_target))
#     a_power=a
#     p=Program()
#     p += X(target_qubits[0])
#     controlled_mul_mod_N(p, control, target_qubits, a_power, N)
#     statevector = simulate_statevector(p, n_count+n_target)
#     print



def run_simulation(config: Dict[str, Any]):
    """
    Run Shor's algorithm given N and a
    and returns a factor of N
    """
    n_count=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))
    n_target = int(np.ceil(np.log2(N))) 
    count_qubits = list(range(n_count))

    count_dim=2**n_count
    target_dim=2**n_target
    # Target register qubits
    target_qubits = list(range(n_count, n_count + n_target))
    p=Program()
    # Initialize counting register to superposition
    for q in count_qubits:
        p += H(q)
    p += X(target_qubits[n_target-1])

    #Apply controlled multiplication
    for j in range(n_count):
        # Calculate a^(2^j) mod N classically
        a_power = pow(a, 2**j, N)
        control = count_qubits[n_count-j-1]
        
        # Apply controlled modular multiplication by a^(2^j) mod N
        controlled_mul_mod_N(p, control, target_qubits, a_power, N)
    
    # Apply inverse QFT to counting register
    p += inverse_qft(count_qubits)
    statevector = simulate_statevector(p, n_count+n_target)
    return statevector
    
    
# for benchmark metrics collection.
def build_program(config: Dict[str, Any]):
    """
    Return the PyQuil program for the Shor QPE/statevector benchmark.
    This is the PyQuil analogue of qiskit's build_circuit(config).
    """
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    prog, _ = shors_algorithm(N, a=a)
    return prog


# Example usage
if __name__ == "__main__":
    N=21
    n_count=6
    n_target=int(np.ceil(np.log2(N)))
    count_dim=2**n_count
    target_dim=2**n_target
    statevector=run_simulation({"t":6,"N":21,"a":2})
    print(np.sum(np.abs(statevector.reshape(2**n_count,2**n_target))**2,axis=0))
    ref=make_shors(t=6, N=21, a=2)
    print(np.real(np.sum(np.abs(ref.reshape(2**n_count,2**n_target))**2,axis=0)))
    print(f"Fidelity: {np.abs(np.vdot(statevector, ref))**2}")