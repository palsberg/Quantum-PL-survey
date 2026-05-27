'''
Findings:
-PennyLane has a built in subroutine called QuantumPhaseEstimation which is easy to call
-There is also a function called iterative_qpe which applies the circuit a set number of times and returns 
a list of measurements. I didn't use this but it could also be done.
-There is no built in way to find the modular exponentiation gate/matrix. Borrowed Elliot's method from Qiskit.


Reference: https://pennylane.ai/codebook/shors-algorithm
*loosely used as a guide. needed a lot of editing to work*
'''

import math
import pennylane as qml
import numpy as np
from typing import Dict, Any

#using Elliot's function from the qiskit implementation
def get_Ma(a: int, N: int, m: int):
    """
    Build the unitary |x> -> |(a*x) mod N> for x < N, and |x> -> |x> for x >= N,
    on m qubits (dimension 2^m). 
    This is a permutation matrix, wrapped as a UnitaryGate.
    """
    dim = 1 << m
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (a * x) % N if x < N else x # calculate correct permutation result
        U[y, x] = 1.0
    return U

def qpe_circuit(a, N, t):
    '''
    Function for the quantum phase estimation circuit of Shor's algorithm.
    
    :param a: a random choice in {2,...,N-2}
    :param N: the integer you want to factor
    :param t: number of qubits in the estimation register
    '''

    n_operand = int(math.ceil(math.log2(N)))
    n_ancilla = n_operand + 2

    ancilla = list(range(0, n_ancilla))
    counting = list(range(n_ancilla, n_ancilla + t))
    operand = list(range(n_ancilla + t, n_ancilla + t + n_operand))

    #set up quantum device
    dev = qml.device("default.qubit", wires=n_ancilla+t+n_operand)

    @qml.qnode(dev)
    def circuit():
        """Return the entire statevector after running the QPE circuit.

        Args:
            matrix (array[complex]): matrix representation of Ma.

        Returns:
            statevector(numpy tensor): final statevector
        """

        # Prepare ancillas and operand
        for i in counting:
            qml.Hadamard(i)
        qml.BasisEmbedding(1, wires=operand)

        # Modular exponentiation. We don't use qml.ModExp because it takes over
        # 10x as long.
        base_power = a
        for ctrl_wire in counting[::-1]:
            qml.ctrl(qml.Multiplier(base_power, operand, N, ancilla), control=ctrl_wire)
            base_power = (base_power * base_power) % N

        # iQFT
        qml.adjoint(qml.QFT)(counting)

        return qml.state()

    state = circuit()[:2**(t + n_operand)]  # Remove the ancilla qubits
    return state

# Additional helper for metric evaluation
def build_qpe_qnode(a: int, N: int, t: int):
    """
    Return a QNode for the PennyLane Shor QPE subroutine, without executing it.
    This is the canonical circuit builder that both the harness and runtime path
    should use.
    """
    n_pe_qubits = t
    n_comp_qubits = int(np.ceil(np.log2(N)))
    total_wires = n_pe_qubits + n_comp_qubits

    pe_wires = list(range(n_pe_qubits))
    comp_wires = list(range(n_pe_qubits, n_pe_qubits + n_comp_qubits))
    matrix = get_Ma(a, N, n_comp_qubits)

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=comp_wires[-1])
        qml.QuantumPhaseEstimation(matrix, target_wires=comp_wires, estimation_wires=pe_wires)
        return qml.state()

    return circuit, total_wires

def build_circuit(config: Dict[str, Any]):
    """
    Harness-facing helper: return an unexecuted QNode plus wire count.
    """
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    return build_qpe_qnode(a=a, N=N, t=t)

def run_simulation(config: Dict[str, Any]):
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))

    sv = qpe_circuit(a, N, t)

    return np.asarray(sv, dtype=np.complex128)
    
if __name__ == "__main__":
    N = 21
    a = 2
    t = 6
    #print(qpe_circuit(2,21,6))
