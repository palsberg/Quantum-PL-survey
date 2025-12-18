from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np

from reference_hamiltonians import kron_many
from reference_hamiltonians import PAULI_I, PAULI_X, PAULI_Z

HADAMARD = (1/np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=np.complex128)

P0 = 0.5 * (PAULI_I + PAULI_Z)  # |0><0|
P1 = 0.5 * (PAULI_I - PAULI_Z)  # |1><1|


def one_qubit_U(n:int, k:int, U:np.ndarray)->np.ndarray:
    """Apply U on kth (0 indexed) qubit in a n qubit circuit"""
    ops=[PAULI_I]*n
    ops[k]=U
    return kron_many(ops) # one single expanded tensored matrix


def control_on(n:int, control:int, target: int, U:np.ndarray)->np.ndarray:
    """Apply U on target qubit iff control qubit is 1"""
    ops0=[PAULI_I]*n
    ops1=[PAULI_I]*n

    ops0[control]=P0 # stay the same if control qubit is 0
    ops1[control]=P1 

    ops1[target]=U # use U if control qubit is 1
    return kron_many(ops0) + kron_many(ops1) # since only one set (btw I and U) is selected, this is automatically normalized



def cnot_on(n:int, control: int, target: int) -> np.ndarray:
    """Apply CNOT(control, target)"""
    return control_on(n=n, control=control, target=target, U=PAULI_X)

def ccnot_on(n:int, c1:int, c2:int, target: int) ->np.ndarray:
    """Apply CCNOT(c1, c2, target)"""
    base=np.eye(2**n, dtype=np.complex128)
    ops=[PAULI_I]*n
    # CCX = I + (|11><11| on controls) ⊗ (X - I on target)
    ops[c1]=P1
    ops[c2]=P1
    ops[target]=PAULI_X-PAULI_I

    return base+kron_many(ops)


def Rk(k: int)-> np.ndarray:
    """phase shift by exp(2i*pi/(2**k))"""
    return np.array([[1,0],[0,np.exp(2j*np.pi/(2**k))]], dtype=np.complex128)

def swap(n:int, t1:int, t2:int)->np.ndarray:
    """swap qubits at t1 and t2"""
    if t1==t2:
        return np.eye(2**n, dtype=np.complex128) # identity
    
    op1=cnot_on(n,t1,t2)
    op2=cnot_on(n,t2,t1)
    return op1 @ op2 @ op1 # 3 CNOT dotted together as a swap


def embed_on_contiguous_block(n: int, offset: int, block: np.ndarray) -> np.ndarray:
    """
    Embed block (size 2^k x 2^k) onto qubits [offset .. offset+k-1] in an n-qubit system
    """
    k = int(np.log2(block.shape[0]))
    if block.shape != (1 << k, 1 << k):
        raise ValueError("block must be 2^k x 2^k.")
    if offset < 0 or offset + k > n:
        raise ValueError("Invalid offset/k for n.")

    I_left = np.eye(1 << offset, dtype=np.complex128)
    I_right = np.eye(1 << (n - offset - k), dtype=np.complex128)
    return np.kron(np.kron(I_left, block), I_right).astype(np.complex128)

def controlled_U_on_block(n: int, control: int, target_offset: int, U_block: np.ndarray) -> np.ndarray:
    """
    Controlled-(U_block on contiguous target block), where control is a single qubit.
    """
    # Projectors on the control qubit embedded into n qubits:
    P0c = one_qubit_U(n, control, P0)  # |0><0| on control
    P1c = one_qubit_U(n, control, P1)  # |1><1| on control

    U_emb = embed_on_contiguous_block(n, target_offset, U_block)
    return P0c @ np.eye(2**n, dtype=np.complex128) + P1c @ U_emb


def qft_matrix(n:int)->np.ndarray:
    """
    QFT on n qubits as a 2^n x 2^n unitary matrix.
    Convention:
      QFT|x> = 1/sqrt(2^n) * sum_y exp(2πi x y / 2^n) |y>
    with basis ordering |0>,|1>,...,|2^n-1> in standard binary.
    """
    dim = 1<<n
    omega = np.exp(2j * np.pi / dim)
    x = np.arange(dim)
    # W[y, x] = omega^(x*y)
    W = omega ** (np.outer(x, x))
    return W/np.sqrt(2**n)

def iqft_matrix(n: int) -> np.ndarray:
    """
    Inverse QFT on n qubits (QFT dagger).
    """
    Q = qft_matrix(n)
    return Q.conj().T

def iqft_on_counting_register(n: int, t: int) -> np.ndarray:
    return embed_on_contiguous_block(n, offset=0, block=iqft_matrix(t))

def Ma_matrix(N:int,a:int): 
    """
    Our manually implemented modular matrix. m is set to 5 as our register qubit.
    We factor 21 with a=2 in Ma.
    """
    m = int(np.ceil(np.log2(N))) 
    dim = 2**m
    M = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        if x < N:
            y = (a * x) % N
        else:
            y = x
        M[y, x] = 1.0  # column x maps to row y
    return M



def make_shors(t:int, N:int, a:int)->np.ndarray:
    """
    Given counting qubits length t, generate a (t+m)*(t+m) np array that represents the shor's algorithm of factoring 21 with a=2. 
    """
    m = int(np.ceil(np.log2(N)))  # 5
    t = 9 # can't hold matrix too large, or we could run on a server
    n = t + m
    Ma = Ma_matrix(N=N,a=a)  # 32x32

    # Initialize state
    # start in |00...0> then apply X on qubit t
    U = np.eye(2**n, dtype=np.complex128)
    
    for q in range(t):
        U = one_qubit_U(n, q, HADAMARD) @ U
    
    # X on qubit t 
    U = one_qubit_U(n, t, PAULI_X) @ U

    # treat counting qubtis as binary, apply corresponding number of Ma
    for idx in range(t):
        U_pow = np.linalg.matrix_power(Ma, 1 << idx)
        CU = controlled_U_on_block(n, control=idx, target_offset=t, U_block=U_pow)
        U = CU @ U

    # inverse QFT on counting register
    U = iqft_on_counting_register(n, t) @ U

    return U 












        











