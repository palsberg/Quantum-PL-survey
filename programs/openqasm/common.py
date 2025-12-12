"""Helpers for emitting OpenQASM 3 source programs and replaying them in Python."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Sequence, Tuple

from ..common import pauli_models

_DEFAULT_DIR = Path("programs/openqasm/generated")
_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_output_path(dest: str | None, default_name: str) -> Path:
    if dest:
        path = Path(dest)
        if path.is_dir():
            path = path / default_name
    else:
        path = _DEFAULT_DIR / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_qasm(name: str, qasm: str, params: dict) -> Path:
    path = _resolve_output_path(params.get("qasm_path"), f"{name}.qasm")
    path.write_text(qasm)
    return path


def render_tfim_trotter_qasm(
    num_sites: int, J: float, h: float, total_time: float, steps: int
) -> str:
    dt = total_time / steps
    body: List[str] = []
    for _ in range(steps):
        for i in range(num_sites - 1):
            theta = 2 * J * dt
            body.append(f"ZZ({theta}) q[{i}], q[{i + 1}];")
        for i in range(num_sites):
            theta = 2 * h * dt
            body.append(f"rx({theta}) q[{i}];")
    ops = "\n    ".join(body)
    qasm = dedent(
        f"""\
        OPENQASM 3;
        include "stdgates.inc";

        gate ZZ(theta) a, b {{
            cx a, b;
            rz(theta) b;
            cx a, b;
        }}

        qubit[{num_sites}] q;
        {ops}
        """
    )
    return qasm


def render_heis_trotter_qasm(
    num_sites: int, J: float, field: float, total_time: float, steps: int
) -> str:
    dt = total_time / steps
    body: List[str] = []
    for _ in range(steps):
        for i in range(num_sites - 1):
            theta = 2 * J * dt
            for gate in ("XX", "YY", "ZZ"):
                body.append(f"{gate}({theta}) q[{i}], q[{i + 1}];")
        for i in range(num_sites):
            theta = 2 * field * dt
            body.append(f"rz({theta}) q[{i}];")
    ops = "\n    ".join(body)
    qasm = dedent(
        f"""\
        OPENQASM 3;
        include "stdgates.inc";

        gate XX(theta) a, b {{
            h a;
            h b;
            cx a, b;
            rz(theta) b;
            cx a, b;
            h a;
            h b;
        }}

        gate YY(theta) a, b {{
            sdg a;
            sdg b;
            h a;
            h b;
            cx a, b;
            rz(theta) b;
            cx a, b;
            h a;
            h b;
            s a;
            s b;
        }}

        gate ZZ(theta) a, b {{
            cx a, b;
            rz(theta) b;
            cx a, b;
        }}

        qubit[{num_sites}] q;
        {ops}
        """
    )
    return qasm


def render_tfim_lcu_qasm(num_sites: int, J: float, h: float, total_time: float) -> str:
    return _render_lcu_qasm(
        num_sites=num_sites,
        gamma=pauli_models.taylor_coefficients(
            pauli_models.tfim_pauli_terms(num_sites, J, h), total_time
        ),
        header_comment="TFIM LCU block-encoding",
    )


def render_heis_lcu_qasm(num_sites: int, J: float, field: float, total_time: float) -> str:
    return _render_lcu_qasm(
        num_sites=num_sites,
        gamma=pauli_models.taylor_coefficients(
            pauli_models.heisenberg_pauli_terms(num_sites, J, field), total_time
        ),
        header_comment="Heisenberg XXX LCU block-encoding",
    )


__all__ = [
    "_write_qasm",
    "render_tfim_trotter_qasm",
    "render_heis_trotter_qasm",
    "render_tfim_lcu_qasm",
    "render_heis_lcu_qasm",
]


def _render_lcu_qasm(
    num_sites: int,
    gamma: Dict[str, complex],
    header_comment: str,
) -> str:
    weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
    if not weights:
        raise ValueError("No LCU terms generated.")
    selection_bits = max(1, math.ceil(math.log2(len(weights))))
    target_len = 1 << selection_bits
    if len(weights) < target_len:
        pad = target_len - len(weights)
        weights.extend([0.0] * pad)
        paulis.extend(["I" * num_sites] * pad)
        phases.extend(["1"] * pad)

    total_weight = sum(weights) if sum(weights) > 0 else 1.0
    amps = [math.sqrt(w / total_weight) if w > 0 else 0.0 for w in weights]

    lines: List[str] = [
        "OPENQASM 3;",
        "include \"stdgates.inc\";",
        "",
        f"// {header_comment}",
        f"qubit[{num_sites}] system;",
        f"qubit[{selection_bits}] selection;",
        "qubit phase_anc;",
        f"qubit[{max(1, selection_bits)}] junk;",
        "",
        "// --- PREPARE block ---",
    ]

    _emit_prepare(lines, amps, list(range(selection_bits)))

    lines.append("")
    lines.append("// --- SELECT block ---")

    for idx, (pauli, phase_tag, weight) in enumerate(zip(paulis, phases, weights)):
        if weight <= 0:
            continue
        lines.extend(
            _emit_select(idx, selection_bits, pauli, phase_tag, weight, num_sites)
        )

    return "\n".join(lines)


def _emit_prepare(lines: List[str], amps: Sequence[float], sel_indices: List[int], controls=None):
    if controls is None:
        controls = []
    if not sel_indices:
        return
    target = sel_indices[0]
    half = len(amps) // 2
    left = amps[:half]
    right = amps[half:]
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(a * a for a in right))
    total = math.sqrt(sum(a * a for a in amps))
    if total == 0 or (left_norm == 0 and right_norm == 0):
        return
    if left_norm == 0:
        theta = math.pi
    elif right_norm == 0:
        theta = 0.0
    else:
        theta = 2 * math.atan2(right_norm, left_norm)
    lines.extend(
        _emit_controlled_gate("ry", theta, f"selection[{target}]", controls)
    )

    if left_norm > 1e-12:
        new_amps = [a / left_norm for a in left]
        _emit_prepare(lines, new_amps, sel_indices[1:], controls + [(target, 0)])
    if right_norm > 1e-12:
        new_amps = [a / right_norm for a in right]
        _emit_prepare(lines, new_amps, sel_indices[1:], controls + [(target, 1)])


def _emit_select(
    idx: int,
    selection_bits: int,
    pauli: str,
    phase_tag: str,
    weight: float,
    num_sites: int,
) -> List[str]:
    lines: List[str] = []
    lines.append(f"// Term {idx}: weight={weight:.6e}, pauli={pauli}, phase={phase_tag}")
    bits = [(idx >> b) & 1 for b in range(selection_bits)]
    mask = [b for b, val in enumerate(bits) if val == 0]
    for b in mask:
        lines.append(f"x selection[{b}];")

    controls = [(bit, 1) for bit in range(selection_bits)]
    phase_map = {"-1": math.pi, "i": math.pi / 2, "-i": -math.pi / 2}
    if phase_tag in phase_map:
        lines.extend(_emit_controlled_gate("rz", phase_map[phase_tag], "phase_anc", controls))

    for q, axis in enumerate(pauli):
        if axis == "I":
            continue
        target = f"system[{q}]"
        if axis == "X":
            lines.extend(_emit_controlled_gate("x", math.nan, target, controls))
        elif axis == "Y":
            # Y = S.X.Sdg in standard gate sets; use 'sdg' as the inverse S.
            lines.extend(_emit_controlled_gate("sdg", math.nan, target, controls))
            lines.extend(_emit_controlled_gate("x", math.nan, target, controls))
            lines.extend(_emit_controlled_gate("s", math.nan, target, controls))
        elif axis == "Z":
            lines.extend(_emit_controlled_gate("z", math.nan, target, controls))

    for b in reversed(mask):
        lines.append(f"x selection[{b}];")
    lines.append("")
    return lines


def _emit_controlled_gate(
    gate: str,
    theta: float,
    target: str,
    controls: Sequence[Tuple[int, int]],
) -> List[str]:
    if not controls:
        if math.isnan(theta):
            return [f"{gate} {target};"]
        return [f"{gate}({theta}) {target};"]
    mask_lines: List[str] = []
    unmask_lines: List[str] = []
    ctrl_names = []
    for idx, val in controls:
        ctrl_names.append(f"selection[{idx}]")
        if val == 0:
            mask_lines.append(f"x selection[{idx}];")
            unmask_lines.insert(0, f"x selection[{idx}];")
    prefix = " ".join(["ctrl @" for _ in controls])
    operands = ", ".join(ctrl_names + [target])
    if math.isnan(theta):
        gate_expr = gate
    else:
        gate_expr = f"{gate}({theta})"
    line = f"{prefix} {gate_expr} {operands};"
    return mask_lines + [line] + unmask_lines
