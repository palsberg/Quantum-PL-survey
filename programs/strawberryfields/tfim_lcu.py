"""Sequential dual-rail CV circuit over TFIM terms (not a full Taylor LCU).

This module applies TFIM terms sequentially in a dual-rail CV encoding
using Strawberry Fields. It is *not* a full 2nd-order Taylor LCU block
encoding: there are no explicit selection ancillas or PREPARE/SELECT
oracles. In the cross-language comparison, this implementation is kept
N/A for LCU correctness and gate-count metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import strawberryfields as sf
from strawberryfields import ops

from . import common

LcuTerm = Tuple[str, int, int, float]


def tfim_terms(num_qubits: int, coupling: float, field: float) -> List[LcuTerm]:
    terms: List[LcuTerm] = []
    for i in range(num_qubits):
        terms.append(("X", i, -1, field))
    for i in range(num_qubits - 1):
        terms.append(("ZZ", i, i + 1, coupling))
    return terms


def run_simulation(config: Dict[str, Any]):
    num_qubits = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    coupling = float(params.get("J", 1.0))
    field = float(params.get("h", 1.0))
    cutoff = int(params.get("cutoff", 3))

    terms = tfim_terms(num_qubits, coupling * total_time, field * total_time)

    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        for i in range(num_qubits):
            ops.Fock(1) | q[2 * i]
            ops.Fock(0) | q[2 * i + 1]

        for kind, idx, jdx, angle in terms:
            if kind == "X":
                common.apply_logical_x(q, idx, angle)
            elif kind == "ZZ" and jdx >= 0:
                common.apply_logical_zz(q, idx, jdx, angle)
    result = common.run_program(prog, cutoff=cutoff)
    return common.logical_state_from_result(result, num_qubits, cutoff)


if __name__ == "__main__":
    state = run_simulation({"num_sites": 2, "time": 0.1, "params": {"J": 0.5, "h": 0.8}})
    print(state)
