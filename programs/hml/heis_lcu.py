"""Heisenberg XXX via HML metadata + Cirq LCU backend."""

from __future__ import annotations

from typing import Any, Dict

from ..cirq import heis_lcu as cirq_heis_lcu
from . import common as hml_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))

    spec = hml_common.render_heis_lcu_spec(num_sites, J, field, total_time)
    hml_common.emit_spec("heis_lcu", spec, params)
    return cirq_heis_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "field": 0.3}}
    print(run_simulation(cfg))

