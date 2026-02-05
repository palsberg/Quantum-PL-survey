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
import math

import numpy as np
import subprocess
import ast
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HARNESS_DIR = ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))


CASE_FILES = {
    "tfim_trotter": "tfim_trotter.slq",
    "tfim_lcu": "tfim_lcu.slq",
    "heis_trotter": "heis_trotter.slq",
    "heis_lcu": "heis_lcu.slq",
    "shors21_2": "shors.slq",

}


def _json_state(vec: np.ndarray):
    return [{"re": float(np.real(v)), "im": float(np.imag(v))} for v in vec.ravel()]

def _parse_silq_amp(s: str) -> complex:
    """
    Parse Silq amplitudes, e.g.
      √0.9950120∠-0.0500156
      -√0.0006210∠ 1.5457963
      √1.0000000
    and also keep support for plain a+bi strings via fallback.
    """
    raw = s.strip()

    # Some Silq outputs include a dot multiplication marker; strip it if present.
    raw = raw.replace("·", " ").strip()

    sign = 1.0
    if raw.startswith("+"):
        raw = raw[1:].strip()
    elif raw.startswith("-"):
        sign = -1.0
        raw = raw[1:].strip()

    # Polar form: (magnitude) ∠ (angle)
    if "∠" in raw:
        mag_str, ang_str = raw.split("∠", 1)
        mag_str = mag_str.strip()
        ang = float(ang_str.strip())

        if mag_str.startswith("√"):
            mag = math.sqrt(float(mag_str[1:].strip()))
        else:
            mag = float(mag_str)

        mag *= sign
        return complex(mag * math.cos(ang), mag * math.sin(ang))

    # Sqrt-only form: √p
    if raw.startswith("√"):
        mag = math.sqrt(float(raw[1:].strip()))
        return complex(sign * mag, 0.0)

    # Fallback: old a+bi format (reuse your existing parser)
    prefix = "-" if sign < 0 else ""
    return _parse_complex(prefix + raw)



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

def _shors_num_qubits(config: Dict[str, Any]) -> int:
    """
    Harness config provides:
      t: number of counting qubits
      N: modulus
    Work register size m = ceil(log2(N)).
    Total qubits n = t + m.
    """


    t = int(config["t"])
    N = int(config["N"])
    m = int(np.ceil(np.log2(N)))
    return t + m


def _parse_silq_output(output: str, case: str, num_sites: int) -> np.ndarray:
    """Parse Silq's Dirac-notation output into a statevector over system qubits."""
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Silq produced no output.")

    terms = []
    for line in lines:
        # Expect lines like:
        #   √1.0000000 · |(0,0)⟩
        #   -√0.0000062∠-0.0375137 · |(1,1)⟩
        if "|" not in line or "⟩" not in line:
            continue

        bar_pos = line.find("|")
        end_basis = line.find("⟩", bar_pos)
        if end_basis == -1:
            continue

        basis_str = line[bar_pos + 1 : end_basis].strip()

        # Everything left of '|' contains the amplitude (maybe plus a '·').
        left = line[:bar_pos].strip()
        if "·" in left:
            left = left.split("·", 1)[0].strip()

        # Support old "(amp)" format if it exists; otherwise parse Silq amp format.
        if "(" in left and ")" in left:
            start = left.rfind("(")
            end = left.find(")", start + 1)
            if end != -1:
                amp_str = left[start + 1 : end].strip()
                amp = _parse_complex(amp_str)
            else:
                amp = _parse_silq_amp(left)
        else:
            amp = _parse_silq_amp(left)

        try:
            basis = ast.literal_eval(basis_str)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Silq basis label '{basis_str}'") from exc

        terms.append((amp, basis))

    if not terms:
        raise RuntimeError(f"Could not parse Silq output. Last line was: {lines[-1]}")

    dim = 2**num_sites
    vec = np.zeros(dim, dtype=np.complex128)


    if case == "shors21_2":
        for amp, basis in terms:
            if not isinstance(basis, tuple):
                continue

            if len(basis) == 2 and isinstance(basis[0], tuple) and isinstance(basis[1], tuple):
                bits = list(basis[0]) + list(basis[1])
            else:
                bits = list(basis)

            if len(bits) != num_sites:
                continue

            idx = _bits_to_index(bits)
            vec[idx] = amp
        # No projection; keep full statevector
        return vec
        
    if case in ("tfim_trotter", "heis_trotter"):
        # Basis labels are tuples like (q0,q1,...).
        for amp, basis in terms:
            if not isinstance(basis, tuple):
                continue
            system_bits = list(basis)
            if len(system_bits) != num_sites:
                continue
            idx = _bits_to_index(system_bits)
            vec[idx] = amp



    else:
        # LCU: basis labels like (anc,(q0,q1,...)) or (anc,q0) for 1-site.
        for amp, basis in terms:
            if not isinstance(basis, tuple) or len(basis) != 2:
                continue
            anc, sys_part = basis
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


    if case == "shors21_2":
        num_sites = _shors_num_qubits(config)
    else:
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


def _fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Fidelity between pure states |psi>, |phi>."""
    psi = psi.reshape(-1)
    phi = phi.reshape(-1)
    # normalize defensively
    n1 = np.linalg.norm(psi)
    n2 = np.linalg.norm(phi)
    if n1 == 0 or n2 == 0:
        return 0.0
    psi = psi / n1
    phi = phi / n2
    ov = np.vdot(phi, psi)  # <phi|psi>
    return float(np.abs(ov) ** 2)


def _selftest_shors(parsed: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Compare parsed Silq statevector against the reference implementation.
    Prints diagnostics to stderr.
    """
    # Import reference
    try:
        from harness.reference_shors import make_shors
    except Exception:
        # fallback if run from different cwd/layout
        from reference_shors import make_shors  # type: ignore

    t = int(config["t"])
    N = int(config["N"])
    a = int(config["a"])

    ref = make_shors(t=t, N=N, a=a)

    if parsed.shape != ref.shape:
        print(
            f"[shors selftest] shape mismatch: parsed={parsed.shape}, ref={ref.shape}",
            file=sys.stderr,
        )
        return

    fid = _fidelity(parsed, ref)
    print(f"[shors selftest] fidelity={fid:.12f}", file=sys.stderr)



def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: run_cli.py <case> [--selftest]")
    case = sys.argv[1]
    do_selftest = (len(sys.argv) == 3 and sys.argv[2] == "--selftest")
    config = json.load(sys.stdin)
    state = _run_case(case, config)
    if do_selftest and case == "shors21_2":
        _selftest_shors(state, config)
    json.dump({"statevector": _json_state(state)}, sys.stdout)



if __name__ == "__main__":
    main()