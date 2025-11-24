"""Heisenberg XXX via pytket LCU."""

from __future__ import annotations

from typing import Any, Dict

from . import lcu_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    return lcu_common.heis_lcu_state(num_sites, J, field, total_time)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "field": 0.3}}
    print(run_simulation(cfg))
