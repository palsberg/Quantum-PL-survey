import cudaq
import numpy as np
from fractions import Fraction

def get_oracle_matrix(N: int, a: int):
    m = int(np.ceil(np.log2(N)))
    oracle_matrix = np.zeros((2**m, 2**m))
    for x in range(N):
        y = (a * x) % N
        oracle_matrix[y, x] = 1
    for x in range(N, 2**m):
        oracle_matrix[x, x] = 1
    return oracle_matrix

# To increase MAX_ORACLE_QUBITS, modify apply_oracle() to handle more cases of m.
# This is necessary because cuda-q does not support calling custom operations
# with qvectors---only individual qubits.
MAX_ORACLE_QUBITS = 9
def register_oracle(N: int, a: int):
    m = int(np.ceil(np.log2(N)))
    for i in range(1, MAX_ORACLE_QUBITS+1):
        if i == m:
            cudaq.register_operation(f'oracle{i}', get_oracle_matrix(N, a).reshape((-1,)))
        else:
            cudaq.register_operation(f'oracle{i}', np.eye(2 ** i))


@cudaq.kernel
def qft(qs: cudaq.qview):
    n = qs.size()
    for i in range(n // 2):
        swap(qs[i], qs[n - i - 1])
    for i in range(n):
        h(qs[i])
        for j in range(i + 1, n):
            angle = (2 * np.pi) / (2**(j - i + 1))
            cr1(angle, [qs[j]], qs[i])

@cudaq.kernel
def apply_oracle(q: cudaq.qview):
    m = q.size()
    if m == 1:
        oracle1(q[0])
    elif m == 2:
        oracle2(q[1], q[0])
    elif m == 3:
        oracle3(q[2], q[1], q[0])
    elif m == 4:
        oracle4(q[3], q[2], q[1], q[0])
    elif m == 5:
        oracle5(q[4], q[3], q[2], q[1], q[0])
    elif m == 6:
        oracle6(q[5], q[4], q[3], q[2], q[1], q[0])
    elif m == 7:
        oracle7(q[6], q[5], q[4], q[3], q[2], q[1], q[0])
    elif m == 8:
        oracle8(q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0])
    elif m == 9:
        oracle9(q[8], q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0])


@cudaq.kernel
def shors(N: int, a: int, t: int, measure: bool):
    # m := log2(N)
    m = 0
    tmp = N
    while tmp > 0:
        m += 1
        tmp = tmp // 2

    ancilla = cudaq.qvector(t)
    operand = cudaq.qvector(m)

    h(ancilla)
    x(operand[0])

    for k in range(t):
        for _ in range(2**k):
            cudaq.control(apply_oracle, ancilla[k], operand)

    cudaq.adjoint(qft, ancilla)
    if measure:
        mz(ancilla)



def main():
    N = 21
    a = 2
    t = 3

    register_oracle(N, a)

    res = cudaq.sample(shors, N, a, t, True)
    print(res)


if __name__ == '__main__':
    main()
