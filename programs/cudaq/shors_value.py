import cudaq
import numpy as np
from fractions import Fraction
import random
from math import gcd
from typing import Any, Dict, Optional, List



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

    operand = cudaq.qvector(m)
    ancilla = cudaq.qvector(t)

    h(ancilla)
    x(operand[0])

    for k in range(t):
        for _ in range(2**k):
            cudaq.control(apply_oracle, ancilla[k], operand)

    cudaq.adjoint(qft, ancilla)
    if measure:
        mz(ancilla)




# Classical routine

# n is the integer that is measured
# t is the number of measurement qubits
def find_order(n, t, a, N) -> Optional[int]:
    x = n / (2**t)
    frac = Fraction(x).limit_denominator(N)

    candidates = []
    for q in range(1, frac.denominator + 1):
        if abs(x - round(x*q)/q) < 2**-(t+1):
            candidates.append(q)

    for r in candidates:
        if pow(a, r, N) == 1:
            return r
    return None


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    if "N" not in config:
        raise ValueError("config must include key 'N'")

    N = int(config["N"])
    a = config.get("a", None)
    a = int(a) if a is not None else None
    t = config.get("t", None)
    t = int(t) if t is not None else None

    max_tries = int(config.get("max_tries", 25))
    seed = int(config.get("seed", 0))
    allow_random_a = bool(config.get("allow_random_a", True))

    shots = int(config.get("shots", 256))
    retries = int(config.get("retries", 16))



    for _ in range(retries):
        a = random.randint(2, N-1)
        K = gcd(a, N)
        if K != 1:
            return np.array(K)

        register_oracle(N, a)
        for _ in range(shots):
            res = cudaq.sample(shors, N, a, t, True, shots_count=10)
            bitstring = max(res, key=lambda b:res[b])
            n = int(bitstring[::-1], 2)

            r = find_order(n, t, a, N)
            if r == None or r % 2 == 1:
                continue

            g = gcd(N, int(a**(r//2) + 1))
            if g == 1 or g == N:
                continue

            return np.array(g)

    raise RuntimeError('Failed to find a factor')




def main():
    N = 21
    a = 2
    t = 6

    register_oracle(N, a)

    for i in range(100):
        res = cudaq.sample(shors, N, a, t, True, shots_count=10)
        bitstring = max(res, key=lambda b:res[b])
        n = int(bitstring[::-1], 2)
        print(n)


if __name__ == '__main__':
    main()
