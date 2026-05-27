import pennylane as qml
import numpy as np
from fractions import Fraction
from typing import Dict, Any
import math

def FindOrderCandidate(a, N, t):
    '''
    Lecture note-style function for the order finding subroutine of Shor's algorithm.
    
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

    @qml.set_shots(10)
    @qml.qnode(dev)
    def qpe_circuit():
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

        return qml.sample(wires=counting)

    samples = qpe_circuit()

    aux = 2 ** t
    r = 0
    for sample in samples:
        binary = "".join([str(int(b)) for b in sample])
        phase = Fraction(int(binary, 2), 2 ** t)
        r = max(r, phase.limit_denominator(aux).denominator)

    return r

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
    while period % 2 == 1:
        a = np.random.randint(2, N - 2)

        if np.gcd(a, N) != 1:
            return [np.gcd(a, N)]

        period = FindOrderCandidate(a, N, t)

        x = (a ** (period // 2)) % N

        if (x != 1 and x != N-1) and period % 2 == 0:
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
