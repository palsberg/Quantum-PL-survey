"""Helpers for emitting HML (SimuQ) sketches and delegating execution to backends."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Dict


def _resolve_output_path(raw_path: str | None, default_name: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_dir():
        path = path / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_spec(spec: str, params: Dict, default_name: str) -> None:
    path = _resolve_output_path(params.get("hml_spec_path"), default_name)
    if path is None:
        return
    path.write_text(spec)


def render_tfim_trotter_spec(num_sites: int, J: float, h: float, total_time: float, steps: int) -> str:
    dt = total_time / steps
    return dedent(
        f"""\
        # Auto-generated HML / SimuQ sketch for TFIM Trotterization
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        h = {h}
        dt = {dt}
        steps = {steps}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]

        H_ZZ = 0
        for i in range(n - 1):
            H_ZZ += J * (q[i].Z * q[i+1].Z)

        H_X = sum(h * q[i].X for i in range(n))

        for _ in range(steps):
            qs.add_evolution(H_ZZ, dt)
            qs.add_evolution(H_X, dt)

        # Compile qs with your desired AAIS/backend (e.g., QuEra Rydberg or IBM).
        """
    )


def render_heis_trotter_spec(num_sites: int, J: float, field: float, total_time: float, steps: int) -> str:
    dt = total_time / steps
    return dedent(
        f"""\
        # Auto-generated HML / SimuQ sketch for Heisenberg XXX + field
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        field = {field}
        dt = {dt}
        steps = {steps}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]

        H_xx = 0
        H_yy = 0
        H_zz = 0
        H_field = 0
        for i in range(n - 1):
            H_xx += J * (q[i].X * q[i+1].X)
            H_yy += J * (q[i].Y * q[i+1].Y)
            H_zz += J * (q[i].Z * q[i+1].Z)
        for i in range(n):
            H_field += field * q[i].Z

        for _ in range(steps):
            qs.add_evolution(H_xx, dt)
            qs.add_evolution(H_yy, dt)
            qs.add_evolution(H_zz, dt)
            qs.add_evolution(H_field, dt)

        # Compile qs with the appropriate AAIS/backend.
        """
    )


def render_tfim_lcu_spec(num_sites: int, J: float, h: float, total_time: float) -> str:
    return dedent(
        f"""\
        # HML sketch for routing TFIM into an LCU backend
        # The Hamiltonian definition mirrors the SimuQ model; execution uses
        # an external gate-model LCU implementation.
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        h = {h}
        T = {total_time}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]
        H = 0
        for i in range(n - 1):
            H += J * (q[i].Z * q[i+1].Z)
        for i in range(n):
            H += h * q[i].X

        qs.add_evolution(H, T)
        # Forward qs to a gate-model LCU engine for block-encoding.
        """
    )


def render_heis_lcu_spec(num_sites: int, J: float, field: float, total_time: float) -> str:
    return dedent(
        f"""\
        # HML sketch for Heisenberg XXX + field routed to an external LCU backend
        from simuq.qsystem import QSystem
        from simuq.environment import qubit

        n = {num_sites}
        J = {J}
        field = {field}
        T = {total_time}

        qs = QSystem()
        q = [qubit(qs) for _ in range(n)]
        H = 0
        for i in range(n - 1):
            H += J * (q[i].X * q[i+1].X + q[i].Y * q[i+1].Y + q[i].Z * q[i+1].Z)
        for i in range(n):
            H += field * q[i].Z

        qs.add_evolution(H, T)
        # Use a gate-model LCU implementation to simulate the block-encoding.
        """
    )


def emit_spec(name: str, spec: str, params: Dict) -> None:
    default_file = f"{name}.hml.py"
    _write_spec(spec, params, default_file)

