"""Heisenberg XXX via Qualtran metadata + Cirq LCU fallback."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..cirq import heis_lcu as cirq_heis_lcu


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    return cirq_heis_lcu.run_simulation(config)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.2,
        "params": {"J": 1.0, "field": 0.3, "lcu_precision": 1e-2},
    }
    print(run_simulation(cfg))
