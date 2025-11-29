"""Dual-rail CV Trotterization of the Heisenberg XXX model in Strawberry Fields.

This module implements a dual-rail continuous-variable encoding of a
Trotterized Heisenberg XXX chain with longitudinal field using
Strawberry Fields. The resulting state is projected onto the logical
qubit subspace and used as an approximate qubit implementation in the
cross-language harness; fidelity is evaluated against the exact qubit
Heisenberg Hamiltonian, and can be reduced by leakage and finite Fock
cutoff effects.
"""

from __future__ import annotations

from typing import Any, Dict

import strawberryfields as sf
from strawberryfields import ops

from . import common


def run_simulation(config: Dict[str, Any]):
    num_qubits = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    coupling = float(params.get("J", 1.0))
    field = float(params.get("field", 0.1))
    steps = int(params.get("trotter_steps", 10))
    cutoff = int(params.get("cutoff", 3))

    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        for i in range(num_qubits):
            ops.Fock(1) | q[2 * i]
            ops.Fock(0) | q[2 * i + 1]

        dt = total_time / steps
        theta = coupling * dt
        phi = field * dt

        for _ in range(steps):
            for i in range(num_qubits - 1):
                common.apply_logical_x(q, i, theta)
                common.apply_logical_zz(q, i, i + 1, theta)
            for i in range(num_qubits):
                common.apply_logical_z(q, i, phi)
    result = common.run_program(prog, cutoff=cutoff)
    return common.logical_state_from_result(result, num_qubits, cutoff)


if __name__ == "__main__":
    state = run_simulation({"num_sites": 2, "time": 0.1, "params": {"J": 0.5, "field": 0.3, "trotter_steps": 5}})
    print(state)
