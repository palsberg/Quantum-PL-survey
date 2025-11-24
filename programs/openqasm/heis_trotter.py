"""Heisenberg XXX via native OpenQASM 3 plus Cirq replay."""

from __future__ import annotations

from typing import Any, Dict

from ..cirq import common as cirq_common
from . import common as oq_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 32))

    qasm, operations = oq_common.render_heis_trotter_qasm(num_sites, J, field, total_time, steps)
    oq_common._write_qasm("heis_trotter", qasm, params)

    circuit, qubits = oq_common.build_circuit_from_operations(num_sites, operations)
    return cirq_common.simulate_statevector(circuit, qubits)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "field": 0.3, "trotter_steps": 48}}
    print(run_simulation(cfg))
