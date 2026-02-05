"""TFIM LCU bridge for Qrisp."""

from __future__ import annotations

from typing import Any, Dict

from . import lcu_common
from ..common import pauli_models


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    t = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))

    H = pauli_models.tfim_pauli_terms(num_sites, J, h)

    return lcu_common.lcu(num_sites, H, t)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "h": 0.7}}
    print(run_simulation(cfg))

