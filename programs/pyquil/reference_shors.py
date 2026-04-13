from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np
import sys

try:
    from .reference_hamiltonians import kron_many, PAULI_I, PAULI_X, PAULI_Z
except ImportError:
    from reference_hamiltonians import kron_many, PAULI_I, PAULI_X, PAULI_Z

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
    # working qubit at the bottom
    m = int(np.ceil(np.log2(N)))  # 5
    n = t + m
    Ma = Ma_matrix(N=N,a=a)  # 32x32

    # Initialize state
    # start in |00...0> then apply X on qubit t + m - 1
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    # U = np.eye(2**n, dtype=np.complex128)
    
    for q in range(t):
        state = one_qubit_U(n, q, HADAMARD) @ state
    
    # X on qubit t + m - 1
    # We are using |00..01> as 1
    state = one_qubit_U(n, t + m - 1, PAULI_X) @ state

    # treat counting qubtis as binary, apply corresponding number of Ma
    for idx in range(t):
        # U_pow = np.linalg.matrix_power(Ma, 1 << idx)
        U_pow = np.linalg.matrix_power(Ma, 1 << (t - 1 - idx)) # Big Endian! first bit gives largest exponent!
        CU = controlled_U_on_block(n, control=idx, target_offset=t, U_block=U_pow)
        state = CU @ state

    # inverse QFT on counting register
    state = iqft_on_counting_register(n, t) @ state

    np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
    #print("reference state:")
    p = np.abs(state)**2
    #print(np.sum(p > 1e-12), np.max(p[p <= 1e-12]) if np.any(p <= 1e-12) else 0.0)

    return state 




#################################################################
# The following are only for testing validity of the tester.
#################################################################
def test_Ma_is_unitary(N=21, a=2):
    U = Ma_matrix(N, a)
    I = np.eye(U.shape[0], dtype=np.complex128)
    assert np.allclose(U.conj().T @ U, I, atol=1e-12, rtol=0)
    assert np.allclose(U @ U.conj().T, I, atol=1e-12, rtol=0)

def test_Ma_action_on_basis(N=21, a=2):
    m = int(np.ceil(np.log2(N)))
    dim = 1 << m
    U = Ma_matrix(N, a)

    for x in range(dim):
        e = np.zeros(dim, dtype=np.complex128)
        e[x] = 1.0
        y_state = U @ e
        y = int(np.argmax(np.abs(y_state)))   # permutation => single 1

        expected = (a * x) % N if x < N else x
        assert y == expected

def shor_qpe_statevector_small(t: int, N: int, a: int) -> np.ndarray:
    """
    Exact final statevector of QPE for modular multiplication oracle,
    using only:
      - Ma_matrix(N,a)
      - iqft_matrix(t)
    No huge (2^(t+m) x 2^(t+m)) matrix is built.
    """
    m = int(np.ceil(np.log2(N)))
    dim_c = 1 << t
    dim_w = 1 << m

    U = Ma_matrix(N, a)

    # state as (counting, work)
    psi = np.zeros((dim_c, dim_w), dtype=np.complex128)
    psi[0, 1] = 1.0  # |0...0> ⊗ |1>

    # H on counting (do it as a dense QFT-free row transform)
    inv_sqrt2 = 1 / np.sqrt(2)
    for q in range(t):
        mask = 1 << (t - 1 - q)
        for r in range(dim_c):
            if (r & mask) == 0:
                r0 = r
                r1 = r | mask
                a0 = psi[r0, :].copy()
                a1 = psi[r1, :].copy()
                psi[r0, :] = (a0 + a1) * inv_sqrt2
                psi[r1, :] = (a0 - a1) * inv_sqrt2

    # controlled-U^(2^idx)
    rows = np.arange(dim_c)
    for idx in range(t):
        # U_pow = np.linalg.matrix_power(U, 1 << idx)
        U_pow = np.linalg.matrix_power(U, 1 << (t - 1 - idx))
        sel = (rows & (1 << (t - 1 - idx))) != 0
        psi[sel, :] = psi[sel, :] @ U_pow.T

    # IQFT on counting
    psi = iqft_matrix(t) @ psi

    return psi.reshape(-1)


def counting_marginal_probs(state: np.ndarray, t: int, N: int) -> np.ndarray:
    """
    Marginalize probabilities onto counting register.
    state has length 2^(t+m), with m=ceil(log2 N).
    """
    m = int(np.ceil(np.log2(N)))
    dim_c = 1 << t
    dim_w = 1 << m
    psi = state.reshape(dim_c, dim_w)
    probs = np.sum(np.abs(psi)**2, axis=1)
    return probs / probs.sum()


def test_qpe_peaks_small():
    N, a = 21, 2
    t = 7  # small enough for numpy
    state = shor_qpe_statevector_small(t, N, a)
    probs = counting_marginal_probs(state, t, N)

    top = np.argsort(probs)[-8:][::-1]
    print("Top outcomes:", [(format(i, f"0{t}b"), probs[i]) for i in top])

    # For N=21, a=2 the order r is 6.
    # QPE peaks near k/2^t ≈ s/r for s=0..r-1.
    r = 6
    expected_peaks = [round((s/r) * (1<<t)) % (1<<t) for s in range(r)]
    # just check at least a few expected peaks appear in top results
    assert len(set(top) & set(expected_peaks)) >= 6

def test_make_shors_matches_reference():
    N, a = 21, 2
    t = 6  # keep small; your make_shors builds big matrices internally

    psi_ref = shor_qpe_statevector_small(t, N, a)
    psi_me  = make_shors(t, N, a)

    # 1) same length
    assert psi_me.shape == psi_ref.shape

    # 2) normalized (numerical tolerance)
    assert np.allclose(np.vdot(psi_me, psi_me), 1.0, atol=1e-10)

    # 3) global phase invariant comparison
    # align phases using the largest-magnitude amplitude
    k = int(np.argmax(np.abs(psi_ref)))
    if np.abs(psi_ref[k]) > 1e-14:
        phase = psi_me[k] / psi_ref[k]
        psi_me_aligned = psi_me / phase
    else:
        psi_me_aligned = psi_me

    assert np.allclose(psi_me_aligned, psi_ref, atol=1e-10, rtol=0)

def main():
    test_Ma_is_unitary()
    test_Ma_action_on_basis()
    shor_qpe_statevector_small(6,21,2)
    test_make_shors_matches_reference()

    print(np.array(one_qubit_U(3,0,PAULI_X))@np.array([1,0,0,0,0,0,0,0]))
    print("All tests passed")

if __name__ == "__main__":
    main()













        











