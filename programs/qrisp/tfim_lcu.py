"""TFIM LCU bridge for Qrisp.

Qrisp currently lacks native multi-ancilla PREPARE/SELECT support, so we reuse the
Qiskit implementation to obtain the statevector while keeping Qrisp as the
Hamiltonian source of truth.
"""

from __future__ import annotations

from typing import Any, Dict

from . import common
from ..qiskit import tfim_lcu as qiskit_tfim_lcu


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))

    # Build the operator to document the Hamiltonian in Qrisp form (for notes/benchmarks).
    common.build_tfim_operator(num_sites, J, h)

    # Delegate the actual LCU block-encoding to the Qiskit backend for now.
    return qiskit_tfim_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "h": 0.7}}
    print(run_simulation(cfg))

