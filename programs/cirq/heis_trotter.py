"""Heisenberg XXX with field via Lie–Trotterization in Cirq."""

from __future__ import annotations

from typing import Any, Dict

from . import common


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    """Simulate Heisenberg XXX evolution with a Lie–Trotter circuit."""
    num_sites = int(config["num_sites"])
    time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 2))

    circuit, qubits = common.trotterize_heisenberg_xxx(num_sites, J, field, time, steps)
    return common.simulate_statevector(circuit, qubits)


if __name__ == "__main__":
    example_config = {
        "num_sites": 3,
        "time": 0.4,
        "params": {"J": 1.0, "field": 0.3, "trotter_steps": 4},
    }
    psi = run_simulation(example_config)
    print(psi)
