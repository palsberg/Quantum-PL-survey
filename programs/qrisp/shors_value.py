from qrisp import *
from typing import Any, Dict, Sequence
import numpy as np
from sympy import continued_fraction_convergents, continued_fraction_iterator, Rational
import random
from math import gcd

def find_order(a, N, t):
    qg = QuantumModulus(N)
    qg[:] = 1
    qpe_res = QuantumFloat(t, exponent=-(t))
    h(qpe_res)
    for i in range(len(qpe_res)):
        with control(qpe_res[i]):
            qg *= a
            a = (a * a) % N
    QFT(qpe_res, inv=True)
    return qpe_res,qg

def get_r_candidates(approx):
        rationals = continued_fraction_convergents(
            continued_fraction_iterator(Rational(approx))
        )
        return [rat.q for rat in rationals]

def shor(N, a, t):
    meas_res = find_order(a, N, t)[0].get_measurement()
    # print(meas_res)
    # print(f"number of outcomes: {len(meas_res)}")
    
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

def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    if "N" not in config:
        raise ValueError("config must include key 'N'")

    N = int(config["N"])
    t = config.get("t", None)
    t = int(t) if t is not None else None

    retries = int(config.get("retries", 16))

    for _ in range(retries):
        a = random.randint(2, N-1)
        K = gcd(a, N)
        if K != 1:
            return np.array(K)

        g = shor(N, a, t)
        if g == 1 or g == N:
            continue

        return np.array(g)

    raise RuntimeError('Failed to find a factor')
