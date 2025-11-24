"""TFIM via HML metadata + Cirq LCU backend."""

from __future__ import annotations

from typing import Any, Dict

from ..cirq import tfim_lcu as cirq_tfim_lcu
from . import common as hml_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))

    spec = hml_common.render_tfim_lcu_spec(num_sites, J, h, total_time)
    hml_common.emit_spec("tfim_lcu", spec, params)
    return cirq_tfim_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "h": 0.7}}
    print(run_simulation(cfg))

