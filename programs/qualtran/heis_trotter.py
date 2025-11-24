"""Heisenberg XXX via a Cirq fallback (Qualtran lacks a dedicated bloq)."""

from __future__ import annotations

from typing import Any, Dict

from ..cirq import common as cirq_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 32))
    circuit, qubits = cirq_common.trotterize_heisenberg_xxx(num_sites, J, field, total_time, steps)
    return cirq_common.simulate_statevector(circuit, qubits)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.3,
        "params": {"J": 1.0, "field": 0.2, "trotter_steps": 40},
    }
    print(run_simulation(cfg))

