#!/usr/bin/env python3
"""
CLI adapter for Silq cases.

Reads the harness config on stdin (used only for num_sites), invokes the
corresponding Silq program via the `silq` CLI, parses the printed statevector
in Dirac notation, and prints amplitudes as JSON in the harness format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import subprocess
import ast
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE_FILES = {
    "tfim_trotter": "tfim_trotter.slq",
    "tfim_lcu": "tfim_lcu.slq",
    "heis_trotter": "heis_trotter.slq",
    "heis_lcu": "heis_lcu.slq",
    "shors21_2":"shors.slq"
}


def _json_state(vec: np.ndarray):
    return [{"re": float(np.real(v)), "im": float(np.imag(v))} for v in vec.ravel()]


def _parse_complex(s: str) -> complex:
    """Parse strings like 'a+bi', 'a-bi', or 'a' into a complex number."""
    s = s.strip()
    if s.endswith("i"):
        core = s[:-1]
        # find the first '+' or '-' (after position 0) that is not part of an exponent
        split = -1
        for idx in range(1, len(core)):
            ch = core[idx]
            if ch in "+-" and core[idx - 1] not in "eE":
                split = idx
                break
        if split == -1:
            # purely imaginary: "bi"
            re_part = 0.0
            im_part = float(core)
        else:
            re_str = core[:split].strip()
            im_str = core[split:].strip()
            re_part = float(re_str) if re_str else 0.0
            im_part = float(im_str) if im_str else 0.0
        return complex(re_part, im_part)
    # purely real
    return complex(float(s), 0.0)


def _bits_to_index(bits: Any) -> int:
    """Map a sequence of 0/1 bits [b0,b1,...] (b0 = most significant) to an index."""
    idx = 0
    for b in bits:
        idx = 2 * idx + int(b)
    return idx


def _parse_silq_output(output: str, case: str, num_sites: int) -> np.ndarray:
    """Parse Silq's Dirac-notation output into a statevector over system qubits."""
    # Take the last non-empty line; Silq typically prints a single line of amplitudes.
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Silq produced no output.")
    line = lines[-1]

    # Manually scan for segments of the form "(amp)·|basis⟩".
    terms = []
    i = 0
    n = len(line)
    while i < n:
        start = line.find("(", i)
        if start == -1:
            break
        end_amp = line.find(")", start + 1)
        if end_amp == -1:
            break
        amp_str = line[start + 1 : end_amp]
        # Find the '|' that starts the basis.
        bar_pos = line.find("|", end_amp)
        if bar_pos == -1:
            break
        end_basis = line.find("⟩", bar_pos)
        if end_basis == -1:
            break
        basis_str = line[bar_pos + 1 : end_basis]
        amp = _parse_complex(amp_str)
        try:
            basis = ast.literal_eval(basis_str)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Silq basis label '{basis_str}'") from exc
        terms.append((amp, basis))
        i = end_basis + 1

    if not terms:
        raise RuntimeError(f"Could not parse Silq output line: {line}")

    dim = 2**num_sites
    vec = np.zeros(dim, dtype=np.complex128)

    if case in ("tfim_trotter", "heis_trotter"):
        # Basis labels are tuples like (q0,q1) for two-system-qubit states.
        for amp, basis in terms:
            if not isinstance(basis, tuple):
                continue
            system_bits = list(basis)
            if len(system_bits) != num_sites:
                continue
            idx = _bits_to_index(system_bits)
            vec[idx] = amp
    else:
        # LCU cases: basis labels look like (anc,(q0,q1)).
        for amp, basis in terms:
            if not isinstance(basis, tuple) or len(basis) != 2:
                continue
            anc, sys_part = basis
            # Only keep ancilla == 1 branch to mimic block selection.
            if int(anc) != 1:
                continue
            if isinstance(sys_part, tuple):
                system_bits = list(sys_part)
            else:
                system_bits = [sys_part]
            if len(system_bits) != num_sites:
                continue
            idx = _bits_to_index(system_bits)
            vec[idx] = amp
        # Normalize the projected state.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

    return vec


def _instantiate_trotter_template(case: str, config: Dict[str, Any]) -> Path:
    """Instantiate a Silq Trotter template with numeric parameters.

    We treat the .slq file as a template containing placeholders that are
    substituted with concrete literals derived from the JSON config.
    """
    filename = CASE_FILES[case]
    template_path = ROOT / "programs" / "silq" / filename
    template = template_path.read_text()

    num_sites = int(config.get("num_sites", 2))
    total_time = float(config.get("time", 0.1))
    params = config.get("params", {})
    steps = int(params.get("trotter_steps", 1))

    if case == "tfim_trotter":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 1.0))
        replacements = {
            "__NUM_SITES__": str(num_sites),
            "__J__": repr(J),
            "__H__": repr(h),
            "__TOTAL_TIME__": repr(total_time),
            "__STEPS__": str(steps),
        }
    elif case == "heis_trotter":
        J = float(params.get("J", 1.0))
        field = float(params.get("field", 0.0))
        replacements = {
            "__NUM_SITES__": str(num_sites),
            "__J__": repr(J),
            "__FIELD__": repr(field),
            "__TOTAL_TIME__": repr(total_time),
            "__STEPS__": str(steps),
        }
    else:
        raise RuntimeError(f"Unexpected Trotter case for templating: {case}")

    for key, value in replacements.items():
        template = template.replace(key, value)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".slq", delete=False)
    tmp.write(template)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _run_case(case: str, config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config.get("num_sites", 2))
    if case in ("tfim_trotter", "heis_trotter"):
        # Use the Trotter template mechanism to specialise to the requested
        # number of sites and parameters.
        slq_path = _instantiate_trotter_template(case, config)
        cleanup = True
    else:
        filename = CASE_FILES.get(case)
        if filename is None:
            raise SystemExit(f"unknown Silq case '{case}'")
        slq_path = ROOT / "programs" / "silq" / filename
        cleanup = False

    cmd = ["silq", "--run", str(slq_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if cleanup:
        try:
            slq_path.unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"silq exited with code {proc.returncode}: {proc.stderr.strip()}"
        )
    return _parse_silq_output(proc.stdout, case, num_sites)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_cli.py <case>")
    case = sys.argv[1]
    config = json.load(sys.stdin)
    state = _run_case(case, config)
    json.dump({"statevector": _json_state(state)}, sys.stdout)


if __name__ == "__main__":
    main()
