"""TFIM dual-rail Trotter circuit using Strawberry Fields."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import strawberryfields as sf
from strawberryfields import ops

from . import common


def run_simulation(config: Dict[str, Any]):
    num_qubits = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    coupling = float(params.get("J", 1.0))
    field = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 10))
    cutoff = int(params.get("cutoff", 3))

    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        # initialize logical |0>^n
        for i in range(num_qubits):
            ops.Fock(1) | q[2 * i]
            ops.Fock(0) | q[2 * i + 1]

        dt = total_time / steps
        theta_x = field * dt
        theta_zz = coupling * dt

        for _ in range(steps):
            for i in range(num_qubits - 1):
                common.apply_logical_zz(q, i, i + 1, theta_zz)
            for i in range(num_qubits):
                common.apply_logical_x(q, i, theta_x)
    result = common.run_program(prog, cutoff=cutoff)
    return common.logical_state_from_result(result, num_qubits, cutoff)


if __name__ == "__main__":
    state = run_simulation({"num_sites": 2, "time": 0.1, "params": {"J": 0.5, "h": 0.8, "trotter_steps": 5}})
    print(state)
