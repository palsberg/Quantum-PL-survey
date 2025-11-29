"""TFIM via HML / SimuQ sketch + Qiskit backend (single-step evolution)."""

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
    h = float(params.get("h", 1.0))

    spec = hml_common.render_tfim_lcu_spec(num_sites, J, h, total_time)
    hml_common.emit_spec("tfim_lcu", spec, params)

    qs = QSystem()
    q = [Qubit(qs) for _ in range(num_sites)]

    # Single Hamiltonian H = J Σ ZZ + h Σ X, evolved for total_time.
    H = 0
    for i in range(num_sites - 1):
        H += J * (q[i].Z * q[i + 1].Z)
    for i in range(num_sites):
        H += h * q[i].X

    qs.add_evolution(H, total_time)
    return hml_common.simulate_qsystem_with_qiskit(qs, num_sites)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "h": 0.7}}
    print(run_simulation(cfg))

