from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qsharp

from programs.common import pauli_models

from . import run_state, ensure_compiled, project_lcu_branch


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    ensure_compiled()
    params = config.get("params", {})
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    coupling = float(params.get("J", 1.0))
    field = float(params.get("field", 0.0))
    op = qsharp.code.HamiltonianSimulation.HeisenbergLCU.Run
    H = pauli_models.heisenberg_pauli_terms(num_sites, coupling, field)
    gamma = pauli_models.taylor_coefficients(H, total_time)
    weights, _, _ = pauli_models.lcu_weights_from_gamma(gamma)
    selector_bits = max(1, int(np.ceil(np.log2(len(weights))))) if weights else 1
    state = run_state(op, num_sites, coupling, field, total_time)
    return project_lcu_branch(state, num_sites, selector_bits)
