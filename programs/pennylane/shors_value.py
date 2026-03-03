import pennylane as qml
from pennylane import numpy as np
from fractions import Fraction
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

def FindOrderCandidate(a, N, t):
    '''
    Lecture note-style function for the order finding subroutine of Shor's algorithm.
    
    :param a: a random choice in {2,...,N-2}
    :param N: the integer you want to factor
    :param t: number of qubits in the estimation register
    '''

    n_comp_qubits = int(np.ceil(np.log2(N)))
    n_pe_qubits = t
    pe_wires = range(n_pe_qubits)
    comp_wires = range(n_pe_qubits, n_pe_qubits + n_comp_qubits)

    matrix = get_Ma(a, N, n_comp_qubits)

    dev = qml.device("default.qubit", wires=n_comp_qubits + n_pe_qubits)

    @qml.set_shots(10)
    @qml.qnode(dev)
    def qpe_circuit(matrix):
        qml.PauliX(wires=comp_wires[-1])
        qml.QuantumPhaseEstimation(matrix, comp_wires, pe_wires)
        return qml.sample(wires=pe_wires)

    samples = qpe_circuit(matrix)

    aux = 2 ** n_pe_qubits
    r = 0
    for sample in samples:
        binary = "".join([str(int(b)) for b in sample])
        phase = Fraction(int(binary, 2), 2 ** n_pe_qubits)
        r = max(r, phase.limit_denominator(aux).denominator)

    return r

def is_coprime(a, N):
    """
    Checks if a and N are coprime.

    """

    if np.gcd(a, N) == 1:
        return True
    return False

def is_odd(r):
    '''
    Checks if r is odd.'''

    if r % 2 == 1:
        return True
    return False

def is_not_one(x, N):
    """
    Checks if x is not 1 or N-1.

    """

    if x != 1 and x != N - 1:
        return True
    return False

def shor(N, t):
    """
    Return the factorization of a given integer.

    Args:
       N (int): integer we want to factorize.
       t (int): number of qubits in the estimation register.

    Returns:
        array[(int)]: [p,q] Prime factors of N.

    """
    
    period = 1
    while is_odd(period):
        a = np.random.randint(2, N - 2)

        if not is_coprime(a, N):
            p = np.gcd(a, N)
            return [p]

        period = FindOrderCandidate(a, N, t)

        x = (a ** (period // 2)) % N

        if is_not_one(x, N) and not is_odd(period):
            p = np.gcd(x - 1, N)
            return p

        period = 1

def run_simulation(config: Dict[str, Any]):
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))

    return np.array(shor(N,t))

if __name__ == "__main__":
    N = 64
    print(shor(N,6))