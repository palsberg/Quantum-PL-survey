from qrisp import *
from typing import Any, Dict, Sequence
import math
import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational

def find_order(a, N):
    qg = QuantumModulus(N)
    qg[:] = 1
    qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a * a) % N
    QFT(qpe_res, inv=True)
    return qpe_res.get_measurement()

def get_r_candidates(approx):
        rationals = continued_fraction_convergents(
            continued_fraction_iterator(Rational(approx))
        )
        return [rat.q for rat in rationals]

def shor(N, a):
    qg = QuantumModulus(N)
    qg[:] = 1
    n = qg.size
    qpe_res = QuantumFloat(2 * n + 1, exponent=-(2 * n + 1))
    h(qpe_res)
    x = a
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= x
            x = (x * x) % N
    QFT(qpe_res, inv=True)
    meas_res = qpe_res.get_measurement()
    
    r_candidates = sum([get_r_candidates(approx) for approx in meas_res.keys()], [])

    for cand in r_candidates:
        if (a**cand) % N == 1:
            r = cand
            break
    else:
        raise Exception("Please sample again")

    if r % 2:
        raise Exception("Please choose another a")
    
    g = np.gcd(a ** (r // 2) + 1, N)
    return g

def run_simulation(config: Dict[str, Any]):
    """
    Run Shor's algorithm given N and a
    and returns a factor of N
    """
    t=int(config.get("t",6))
    N=int(config.get("N",21))
    a=int(config.get("a",2))

    return shor(N,a)

if __name__ == "__main__":
    print(shor(21,2))