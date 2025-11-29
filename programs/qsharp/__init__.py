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


def project_lcu_branch(state: np.ndarray, num_system_qubits: int, num_selector_bits: int) -> np.ndarray:
    """Project a full statevector onto the |selector=0…0, phase=1> branch."""
    total_qubits = num_system_qubits + num_selector_bits + 1
    expected_dim = 2**total_qubits
    if state.size != expected_dim:
        raise ValueError(
            f"Statevector dim {state.size} incompatible with {total_qubits} qubits."
        )
    dim_sys = 2**num_system_qubits
    success = np.zeros(dim_sys, dtype=np.complex128)
    system_shift = 1 + num_selector_bits
    for basis in range(dim_sys):
        index = 1  # phase qubit (LSB) set to |1>
        index |= basis << system_shift
        success[basis] = state[index]
    norm = np.linalg.norm(success)
    if norm == 0:
        raise ValueError("LCU block produced zero amplitude on |0^m,1> branch.")
    return success / norm
