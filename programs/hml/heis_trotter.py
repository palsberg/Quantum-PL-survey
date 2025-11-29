"""Heisenberg XXX via HML / SimuQ sketch + Qiskit backend."""

from __future__ import annotations

from typing import Any, Dict

from simuq.qsystem import QSystem
from simuq.environment import Qubit

from . import common as hml_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 32))

    spec = hml_common.render_heis_trotter_spec(num_sites, J, field, total_time, steps)
    hml_common.emit_spec("heis_trotter", spec, params)

    dt = total_time / steps
    qs = QSystem()
    q = [Qubit(qs) for _ in range(num_sites)]

    # Match the HML spec: Heisenberg XXX chain plus field.
    H_xx = 0
    H_yy = 0
    H_zz = 0
    H_field = 0
    for i in range(num_sites - 1):
        H_xx += J * (q[i].X * q[i + 1].X)
        H_yy += J * (q[i].Y * q[i + 1].Y)
        H_zz += J * (q[i].Z * q[i + 1].Z)
    for i in range(num_sites):
        H_field += field * q[i].Z

    for _ in range(steps):
        qs.add_evolution(H_xx, dt)
        qs.add_evolution(H_yy, dt)
        qs.add_evolution(H_zz, dt)
        qs.add_evolution(H_field, dt)

    return hml_common.simulate_qsystem_with_qiskit(qs, num_sites)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "field": 0.25, "trotter_steps": 48}}
    print(run_simulation(cfg))

