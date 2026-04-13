from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qsharp

from . import ensure_compiled


def run_simulation(config: Dict[str, Any]) -> int:
    ensure_compiled()

    N = int(config.get("N", 21))

    # Call the Q# operation — returns a tuple (p, q)
    results = qsharp.run(
        qsharp.code.HamiltonianSimulation.Shors.FactorSemiprimeInteger,
        1,  # shots
        N
    )

    # results is a list of shot results; take the first
    p, q = results[0]

    # Return a single factor
    return np.array(int(p))