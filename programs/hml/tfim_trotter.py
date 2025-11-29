"""TFIM via HML / SimuQ sketch + Qiskit backend."""

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
    steps = int(params.get("trotter_steps", 32))

    spec = hml_common.render_tfim_trotter_spec(num_sites, J, h, total_time, steps)
    hml_common.emit_spec("tfim_trotter", spec, params)

    dt = total_time / steps
    qs = QSystem()
    q = [Qubit(qs) for _ in range(num_sites)]

    # Match the HML spec: H_ZZ and H_X terms on a chain of qubits.
    H_ZZ = 0
    for i in range(num_sites - 1):
        H_ZZ += J * (q[i].Z * q[i + 1].Z)
    H_X = 0
    for i in range(num_sites):
        H_X += h * q[i].X

    for _ in range(steps):
        qs.add_evolution(H_ZZ, dt)
        qs.add_evolution(H_X, dt)

    return hml_common.simulate_qsystem_with_qiskit(qs, num_sites)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "h": 0.7, "trotter_steps": 40}}
    print(run_simulation(cfg))

