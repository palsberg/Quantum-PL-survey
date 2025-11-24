"""TFIM via Lie–Trotterization in Cirq."""

from __future__ import annotations

from typing import Any, Dict

from . import common


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    """Simulate TFIM time evolution using Lie–Trotterization."""
    num_sites = int(config["num_sites"])
    time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 2))

    circuit, qubits = common.trotterize_tfim(num_sites, J, h, time, steps)
    return common.simulate_statevector(circuit, qubits)


if __name__ == "__main__":
    example_config = {
        "num_sites": 3,
        "time": 0.4,
        "params": {"J": 1.0, "h": 0.7, "trotter_steps": 4},
    }
    psi = run_simulation(example_config)
    print(psi)
