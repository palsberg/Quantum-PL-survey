"""Shared helpers for PennyLane-based implementations."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pennylane as qml


def _run_qnode(num_wires: int, builder: Callable[[float], None]) -> np.ndarray:
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def circuit():
        builder()
        return qml.state()

    return circuit()


def tfim_trotter_state(num_sites: int, J: float, h: float, total_time: float, steps: int) -> np.ndarray:
    dt = total_time / steps

    def build():
        for _ in range(steps):
            for i in range(num_sites - 1):
                qml.PauliRot(2 * J * dt, wires=[i, i + 1], pauli_word="ZZ")
            for i in range(num_sites):
                qml.PauliRot(2 * h * dt, wires=[i], pauli_word="X")

    return _run_qnode(num_sites, build)


def heis_trotter_state(num_sites: int, J: float, field: float, total_time: float, steps: int) -> np.ndarray:
    dt = total_time / steps

    def build():
        for _ in range(steps):
            for i in range(num_sites - 1):
                for axis in ("X", "Y", "Z"):
                    qml.PauliRot(2 * J * dt, wires=[i, i + 1], pauli_word=axis * 2)
            for i in range(num_sites):
                qml.PauliRot(2 * field * dt, wires=[i], pauli_word="Z")

    return _run_qnode(num_sites, build)

