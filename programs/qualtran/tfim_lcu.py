"""TFIM via Qualtran metadata + Cirq LCU fallback.

Qualtran tracks the Hamiltonian/source of truth while the executable LCU circuit
reuses the Cirq implementation to keep the harness runnable without heavy
alias-sampling ancillas.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..cirq import tfim_lcu as cirq_tfim_lcu


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    return cirq_tfim_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.2,
        "params": {"J": 1.0, "h": 0.7, "lcu_precision": 5e-3},
    }
    print(run_simulation(cfg))
