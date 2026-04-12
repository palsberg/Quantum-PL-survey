from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qsharp

from . import ensure_compiled




def run_simulation(config: Dict[str, Any]) -> int:
    ensure_compiled()
    from fractions import Fraction
    from math import gcd
    from collections import Counter

    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    bitsize = N.bit_length()
    t = int(config.get("t", 2 * bitsize + 1))   # 11 for N=21

    # Run full-register QPE 2048 times; each shot returns one frequency integer
    freqs = qsharp.run(
        qsharp.code.HamiltonianSimulation.Shors.MeasureFrequency,
        2048,   # shots
        N, a, t
    )

    counts = Counter(freqs)
    for freq, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if freq == 0:
            continue
        phase = Fraction(freq, 1 << t).limit_denominator(N)
        r = phase.denominator
        if r > 0 and pow(a, r, N) == 1 and r % 2 == 0:
            ar2 = pow(a, r // 2, N)
            d = gcd(ar2 - 1, N)
            if 1 < d < N:
                return np.array(int(d))
            d = gcd(ar2 + 1, N)
            if 1 < d < N:
                return np.array(int(d))

    raise RuntimeError(f"Shor (Q#) failed to find a factor for N={N}")



# def run_simulation(config: Dict[str, Any]) -> int:
#     ensure_compiled()

#     N = int(config.get("N", 21))

#     # Call the Q# operation — returns a tuple (p, q)
#     results = qsharp.run(
#         qsharp.code.HamiltonianSimulation.Shors.FactorSemiprimeInteger,
#         1,  # shots
#         N
#     )

#     # results is a list of shot results; take the first
#     p, q = results[0]

#     # Return a single factor
#     return np.array(int(p))

