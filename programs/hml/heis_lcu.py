"""Heisenberg XXX via HML / SimuQ sketch + Qiskit backend (single-step evolution)."""

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

    spec = hml_common.render_heis_lcu_spec(num_sites, J, field, total_time)
    hml_common.emit_spec("heis_lcu", spec, params)

    qs = QSystem()
    q = [Qubit(qs) for _ in range(num_sites)]

    # Single Hamiltonian H = J Σ (XX+YY+ZZ) + field Σ Z, evolved for total_time.
    H = 0
    for i in range(num_sites - 1):
        H += J * (q[i].X * q[i + 1].X)
        H += J * (q[i].Y * q[i + 1].Y)
        H += J * (q[i].Z * q[i + 1].Z)
    for i in range(num_sites):
        H += field * q[i].Z

    qs.add_evolution(H, total_time)
    return hml_common.simulate_qsystem_with_qiskit(qs, num_sites)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "field": 0.3}}
    print(run_simulation(cfg))

