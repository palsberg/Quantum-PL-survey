"""Sequential Heisenberg XXX 'LCU' circuit using Strawberry Fields."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import strawberryfields as sf
from strawberryfields import ops

from . import common

LcuTerm = Tuple[str, int, int, float]


def heis_terms(num_qubits: int, coupling: float, field: float) -> List[LcuTerm]:
    terms: List[LcuTerm] = []
    for i in range(num_qubits - 1):
        terms.append(("XX", i, i + 1, coupling))
        terms.append(("ZZ", i, i + 1, coupling))
    for i in range(num_qubits):
        terms.append(("Z", i, -1, field))
    return terms


def run_simulation(config: Dict[str, Any]):
    num_qubits = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    coupling = float(params.get("J", 1.0))
    field = float(params.get("field", 0.1))
    cutoff = int(params.get("cutoff", 3))

    terms = heis_terms(num_qubits, coupling * total_time, field * total_time)

    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        for i in range(num_qubits):
            ops.Fock(1) | q[2 * i]
            ops.Fock(0) | q[2 * i + 1]

        for kind, idx, jdx, angle in terms:
            if kind == "XX" and jdx >= 0:
                common.apply_logical_x(q, idx, angle)
                common.apply_logical_x(q, jdx, angle)
            elif kind == "ZZ" and jdx >= 0:
                common.apply_logical_zz(q, idx, jdx, angle)
            elif kind == "Z":
                common.apply_logical_z(q, idx, angle)
    result = common.run_program(prog, cutoff=cutoff)
    return common.logical_state_from_result(result, num_qubits, cutoff)


if __name__ == "__main__":
    state = run_simulation({"num_sites": 2, "time": 0.1, "params": {"J": 0.5, "field": 0.3}})
    print(state)
