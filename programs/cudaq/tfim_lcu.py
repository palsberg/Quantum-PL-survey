from __future__ import annotations

from typing import Any, Dict

from cudaq import spin
import numpy as np

from . import lcu_common


def create_hamiltonian_tfim(n_sites: int, coupling: float, field: float):
    """Create the TFIM Hamiltonian operator"""
    H = 0
    for i in range(0, n_sites-1):
        H += coupling * spin.z(i) * spin.z(i + 1)
    for i in range(0, n_sites):
        H += field * spin.x(i)
    return H

def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))

    H = create_hamiltonian_tfim(num_sites, J, h)
    state = lcu_common.lcu(num_sites, H, total_time)
    return state


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.3,
        "params": {"J": 1.0, "h": 0.7},
    }
    state = run_simulation(cfg)
    print(state)
