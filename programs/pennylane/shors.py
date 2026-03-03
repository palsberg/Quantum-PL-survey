'''
Findings:
-PennyLane has a built in subroutine called QuantumPhaseEstimation which is easy to call
-There is also a function called iterative_qpe which applies the circuit a set number of times and returns 
a list of measurements. I didn't use this but it could also be done.
-There is no built in way to find the modular exponentiation gate/matrix. Borrowed Elliot's method from Qiskit.


Reference: https://pennylane.ai/codebook/shors-algorithm
*loosely used as a guide. needed a lot of editing to work*
'''

import pennylane as qml
from pennylane import numpy as np
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

    n_pe_qubits = t #size of estimation register
    pe_wires = range(n_pe_qubits)
    n_comp_qubits = int(np.ceil(np.log2(N))) #size of computational register
    comp_wires = range(n_pe_qubits, n_pe_qubits+n_comp_qubits)

    #define the matrix Ma for later
    matrix = get_Ma(a, N, n_comp_qubits)

    #set up quantum device
    dev = qml.device("default.qubit", wires=n_comp_qubits + n_pe_qubits)

    @qml.qnode(dev)
    def circuit(matrix):
        """Return the entire statevector after running the QPE circuit.

        Args:
            matrix (array[complex]): matrix representation of Ma.

        Returns:
            statevector(numpy tensor): final statevector
        """

        # CREATE THE INITIAL STATE |0...1> ON TARGET WIRES
        qml.PauliX(wires = comp_wires[-1])

        #Built in Quantum Phase Estimation subroutine
        qml.QuantumPhaseEstimation(matrix,comp_wires,pe_wires)

        return qml.state()
    
    return circuit(matrix)

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