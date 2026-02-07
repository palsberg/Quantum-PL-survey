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
import math
import re


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import reference Shor's implementation
try:
    from harness.reference_shors import make_shors
except ImportError:
    make_shors = None

CASE_FILES = {
    "tfim_trotter": "tfim_trotter.slq",
    "tfim_lcu": "tfim_lcu.slq",
    "heis_trotter": "heis_trotter.slq",
    "heis_lcu": "heis_lcu.slq",
    "shors21_2": "shors.slq",
    "shors21_2_value": "shors_value.slq",
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


def _parse_silq_output(output: str, case: str, num_sites: int) -> np.ndarray | int:
    """Parse Silq's Dirac-notation output into a statevector over system qubits."""
    # Take the last non-empty line; Silq typically prints a single line of amplitudes.
    if case == "shors21_2_value":
        # For the value case, we expect a single integer output, not a statevector.
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("Silq produced no output.")
        line = lines[-1]
        try:
            value = int(line)
            return value
        except ValueError as exc:
            raise RuntimeError(f"Failed to parse Shor's value from Silq output: '{line}'") from exc
        
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

    # Special handling for Shor's algorithm
    if case == "shors21_2":
        # Shor's output format: |(counting_int, work_int)⟩
        # Silq already handles bit ordering to match Python reference
        # Just convert directly to flat index
        
        if not terms:
            raise RuntimeError("No terms found in Shor's output")
        
        # Infer dimensions from maximum indices
        max_counting = 0
        max_work = 0
        for amp, basis in terms:
            if isinstance(basis, tuple) and len(basis) == 2:
                c, w = basis
                if isinstance(c, int) and isinstance(w, int):
                    max_counting = max(max_counting, c)
                    max_work = max(max_work, w)
        
        # Bit widths (ensure minimum t=6, m=5)
        t = max(max_counting.bit_length(), 6)
        m = max(max_work.bit_length(), 5)
        
        dim = 2 ** (t + m)
        vec = np.zeros(dim, dtype=np.complex128)
        
        for amp, basis in terms:
            if isinstance(basis, tuple) and len(basis) == 2:
                counting_int, work_int = basis
                if isinstance(counting_int, int) and isinstance(work_int, int):
                    # Direct conversion: index = counting * 2^m + work
                    idx = counting_int * (2 ** m) + work_int
                    if idx < dim:
                        vec[idx] = amp
        
        return vec

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



def _instantiate_shor_state_template(config: Dict[str, Any]) -> Path:
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    t = int(config.get("t", 6))
    after_qft = bool(config.get("after_qft", True))

    w = max(1, math.ceil(math.log2(N)))
    after = "true" if after_qft else "false"

    base_path = ROOT / "programs" / "silq" / CASE_FILES["shors21_2"]
    src = base_path.read_text()
    m=math.ceil(math.log2(N))

    # Ensure helper exists (the file currently doesn't have it)
    if "def estimatePhaseState" not in src:
        src += """

// Returns the joint quantum state (phase register, work register).
def estimatePhaseState[n:!ℕ](
    f:uint[n] !→lifted uint[n],
    after_qft:!𝔹
) mfree : uint[n] × uint[n] {
    cand := 0:uint[n];
    for k in [0..n) { cand[k] := H(cand[k]); }
    work := f(cand);
    if after_qft { cand := reverse(QFT[n])(cand); }
    return (cand, work);
}
def estimatePhaseState2[t:!ℕ, m:!ℕ](
    f:uint[t] !→lifted uint[m],
    after_qft:!𝔹
) mfree : uint[t] × uint[m] {

    cand := 0:uint[t];
    for k in [0..t) { cand[k] := H(cand[k]); }

    work := f(cand);

    if after_qft {
        cand := reverse(QFT[t])(cand);
    }

    return (cand, work);
}
"""

    new_main = f"""
def main() {{
  N := {N};
  a := {a};

  def f(b:uint[{t}]) lifted => powm((a as uint[{m}]), b, N);
  return estimatePhaseState2[{t},{m}](f, true);
}}
""".strip()

    # Replace def main() { ... } in the file
    main_re = re.compile(r"def\s+main\s*\(\s*\)\s*\{{.*?\}}\s*", re.DOTALL)
    if main_re.search(src):
        src = main_re.sub(new_main + "\n\n", src, count=1)
    else:
        src = new_main + "\n\n" + src

    tmp = tempfile.NamedTemporaryFile("w", suffix=".slq", delete=False)
    tmp.write(src)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _instantiate_shor_value_template(config: Dict[str, Any]) -> Path:
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    t = int(config.get("t", 6))
    after_qft = bool(config.get("after_qft", True))

    w = max(1, math.ceil(math.log2(N)))
    after = "true" if after_qft else "false"

    base_path = ROOT / "programs" / "silq" / CASE_FILES["shors21_2"]
    src = base_path.read_text()
    new_main = f"""
def main(){{
	n:={N};
	r:=shor(n);
	assert(1<r&&r<n);
	assert(n%r=0);
	return r;
}}
""".strip()

    # Replace def main() { ... } in the file
    main_re = re.compile(r"def\s+main\s*\(\s*\)\s*\{{.*?\}}\s*", re.DOTALL)
    if main_re.search(src):
        src = main_re.sub(new_main + "\n\n", src, count=1)
    else:
        src = new_main + "\n\n" + src

    tmp = tempfile.NamedTemporaryFile("w", suffix=".slq", delete=False)
    tmp.write(src)
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
    elif case == "shors21_2":
        slq_path = _instantiate_shor_state_template(config)
        cleanup = True
    elif case == "shors21_2_value":
        slq_path = _instantiate_shor_value_template(config)
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
    
    if isinstance(state, int):
        json.dump({"value": state}, sys.stdout)
    else:
        json.dump({"statevector": _json_state(state)}, sys.stdout)


if __name__ == "__main__":
    main()