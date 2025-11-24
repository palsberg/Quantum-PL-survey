"""Helpers for running Q# programs via the modern qdk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import qsharp

_LOADED = False


def ensure_compiled() -> None:
    global _LOADED
    if _LOADED:
        return
    root = Path(__file__).resolve().parent
    chunks: list[str] = []
    for path in sorted(root.glob("*.qs")):
        chunks.append(path.read_text())
    if not chunks:
        raise RuntimeError("No Q# source files found in programs/qsharp.")
    qsharp.eval("\n".join(chunks))
    _LOADED = True


def run_state(operation: Any, *args: Any) -> np.ndarray:
    """Execute a Q# operation once and return the dumped statevector."""
    ensure_compiled()
    shots = qsharp.run(operation, 1, *args, save_events=True)
    shot = shots[0]
    dumps = shot.get("dumps", [])
    if not dumps:
        raise RuntimeError("Q# operation did not emit a DumpMachine output.")
    state = dumps[-1].as_dense_state()
    return np.array(state, dtype=np.complex128)
