from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qsharp

from . import run_state, ensure_compiled


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    ensure_compiled()
    params = config.get("params", {})
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    steps = int(params.get("trotter_steps", 1))
    coupling = float(params.get("J", 1.0))
    field = float(params.get("h", 0.0))
    op = qsharp.code.HamiltonianSimulation.TFIMTrotter.Run
    return run_state(op, num_sites, steps, coupling, field, total_time)
