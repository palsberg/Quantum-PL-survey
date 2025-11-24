"""Heisenberg XXX LCU bridge for Qrisp."""

from __future__ import annotations

from typing import Any, Dict

from . import common
from ..qiskit import heis_lcu as qiskit_heis_lcu


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))

    common.build_heisenberg_operator(num_sites, J, field)
    return qiskit_heis_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "field": 0.3}}
    print(run_simulation(cfg))

