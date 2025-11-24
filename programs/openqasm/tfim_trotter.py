"""TFIM via native OpenQASM 3 plus Cirq replay."""

from __future__ import annotations

from typing import Any, Dict

from ..cirq import common as cirq_common
from . import common as oq_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 32))

    qasm, operations = oq_common.render_tfim_trotter_qasm(num_sites, J, h, total_time, steps)
    oq_common._write_qasm("tfim_trotter", qasm, params)

    circuit, qubits = oq_common.build_circuit_from_operations(num_sites, operations)
    return cirq_common.simulate_statevector(circuit, qubits)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "h": 0.7, "trotter_steps": 40}}
    print(run_simulation(cfg))
