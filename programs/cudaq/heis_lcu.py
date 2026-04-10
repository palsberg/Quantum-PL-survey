from __future__ import annotations

from typing import Any, Dict

from cudaq import spin
import numpy as np

from . import lcu_common


def create_hamiltonian_heisenberg(n_sites: int, coupling: float, field: float):
    """Create the Heisenberg Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.x(i) * spin.x(i + 1)
        H += coupling * spin.y(i) * spin.y(i + 1)
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.z(i)
    return H

def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))

    H = create_hamiltonian_heisenberg(num_sites, J, field)
    state = lcu_common.lcu(num_sites, H, total_time, no_loop=True)
    return state


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.3,
        "params": {"J": 1.0, "field": 0.25},
    }
    state = run_simulation(cfg)
    print(state)
