"""Shared helpers for Qrisp-based programs."""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import numpy as np

_QRISP_IMPORT_ERROR: Optional[Exception] = None

try:
    from qrisp import QuantumVariable, ry
    from qrisp.operators.qubit import QubitOperator, X, Y, Z
except Exception as exc:  # pragma: no cover - best effort when qrisp unavailable
    QuantumVariable = None  # type: ignore[assignment]
    ry = None  # type: ignore[assignment]
    QubitOperator = None  # type: ignore[assignment]
    X = Y = Z = None  # type: ignore[assignment]
    _QRISP_IMPORT_ERROR = exc


def _require_qrisp() -> None:
    if _QRISP_IMPORT_ERROR is not None:
        raise ImportError(
            "Qrisp is not installed. Please follow the instructions at https://qrisp.eu/ "
            "to install the library before running the Qrisp programs."
        ) from _QRISP_IMPORT_ERROR


def _state_from_qv(qv: "QuantumVariable") -> np.ndarray:
    for attr in ("state_vector", "statevector", "state"):
        fn = getattr(qv, attr, None)
        if callable(fn):
            state = fn()
            return np.asarray(state, dtype=np.complex128)
    qs = getattr(qv, "qs", None)
    if qs is not None:
        compiler = getattr(qs, "compile", None)
        if callable(compiler):
            circuit = compiler()
            sv_fn = getattr(circuit, "statevector_array", None)
            if callable(sv_fn):
                return np.asarray(sv_fn(), dtype=np.complex128)
    raise RuntimeError(
        "QuantumVariable does not expose a statevector API and session compilation "
        "statevector_array() is unavailable."
    )


def build_tfim_operator(num_sites: int, J: float, h: float) -> "QubitOperator":
    _require_qrisp()
    H = QubitOperator()
    for i in range(num_sites - 1):
        H += J * (Z(i) * Z(i + 1))
    for i in range(num_sites):
        H += h * X(i)
    return H


def build_heisenberg_operator(num_sites: int, J: float, field: float) -> "QubitOperator":
    _require_qrisp()
    H = QubitOperator()
    for i in range(num_sites - 1):
        H += J * (X(i) * X(i + 1))
        H += J * (Y(i) * Y(i + 1))
        H += J * (Z(i) * Z(i + 1))
    for i in range(num_sites):
        H += field * Z(i)
    return H


def _prepare_initial_state(num_sites: int, angle: float) -> "QuantumVariable":
    _require_qrisp()
    qv = QuantumVariable(num_sites)
    if angle != 0.0:
        ry(angle, qv)
    return qv


def trotter_state(
    operator_builder: Callable[[], "QubitOperator"],
    num_sites: int,
    total_time: float,
    steps: int,
    order: int = 1,
    init_angle: float = np.pi / 8,
) -> np.ndarray:
    H = operator_builder()
    trot = getattr(H, "trotterization")
    params = inspect.signature(trot).parameters
    kwargs = {"order": order} if "order" in params else {}
    evolution = trot(**kwargs)
    qv = _prepare_initial_state(num_sites, init_angle)
    evolution(qv, t=total_time, steps=steps)
    return _state_from_qv(qv)
