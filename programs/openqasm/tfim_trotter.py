"""TFIM via native OpenQASM 3 executed on a Qiskit backend."""

from __future__ import annotations

from typing import Any, Dict

from . import common as oq_common


def run_simulation(config: Dict[str, Any]):
    # Extract benchmark parameters.
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 32))

    # Emit OpenQASM 3 program and persist it for artifacts.
    qasm = oq_common.render_tfim_trotter_qasm(num_sites, J, h, total_time, steps)
    oq_common._write_qasm("tfim_trotter", qasm, params)

    # Execute the QASM program using Qiskit's OpenQASM 3 importer and
    # statevector simulator.
    from qiskit.qasm3 import loads as qasm3_loads  # type: ignore
    from qiskit.quantum_info import Statevector  # type: ignore

    qc = qasm3_loads(qasm)
    state = Statevector.from_instruction(qc).data
    return state


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "h": 0.7, "trotter_steps": 40}}
    print(run_simulation(cfg))
