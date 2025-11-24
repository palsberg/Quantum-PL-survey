#!/usr/bin/env python3
"""
Temporary CLI shim for Silq cases.

Reads the harness config on stdin, delegates to an existing Python implementation
to obtain a reference statevector, and prints the amplitudes as JSON.

Once the Silq toolchain can emit statevectors directly, replace the dispatcher
below with an invocation of the actual Silq programs.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = {
    "tfim_trotter": "programs.cirq.tfim_trotter",
    "tfim_lcu": "programs.cirq.tfim_lcu",
    "heis_trotter": "programs.cirq.heis_trotter",
    "heis_lcu": "programs.cirq.heis_lcu",
}


def _run_case(case: str, config: Dict[str, Any]) -> np.ndarray:
    module_name = MODULES.get(case)
    if module_name is None:
        raise SystemExit(f"unknown Silq case '{case}'")
    module = importlib.import_module(module_name)
    return module.run_simulation(config)


def _json_state(vec: np.ndarray):
    return [{"re": float(np.real(v)), "im": float(np.imag(v))} for v in vec.ravel()]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_cli.py <case>")
    case = sys.argv[1]
    config = json.load(sys.stdin)
    state = _run_case(case, config)
    json.dump({"statevector": _json_state(state)}, sys.stdout)


if __name__ == "__main__":
    main()
