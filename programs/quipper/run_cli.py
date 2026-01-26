#!/usr/bin/env python3
"""CLI adapter for Quipper circuits used by the benchmarking harness."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
QUIPPER_DIR = ROOT / "programs" / "quipper"
BUILD_DIR = QUIPPER_DIR / "build"
BIN_DIR = QUIPPER_DIR / "bin"

HS_SOURCES = {
    "tfim_trotter": "tfim_trotter.hs",
    "tfim_lcu": "tfim_lcu.hs",
    "heis_trotter": "heis_trotter.hs",
    "heis_lcu": "heis_lcu.hs",
    "shors21_2":"shors.hs"
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quipper CLI shim")
    parser.add_argument(
        "--mode",
        choices=("simulate", "metrics"),
        default="simulate",
        help="Operation mode (default: simulate).",
    )
    parser.add_argument("case", choices=sorted(HS_SOURCES.keys()), help="Benchmark case.")
    args = parser.parse_args()

    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"invalid JSON from harness: {exc}") from exc

    payload = invoke_quipper(args.case, args.mode, config)

    if args.mode == "simulate":
        amplitudes = payload.get("amplitudes")
        if amplitudes is None:
            raise SystemExit("Quipper simulation payload missing 'amplitudes'.")
        state = process_amplitudes(args.case, config, amplitudes)
        json.dump({"statevector": state}, sys.stdout)
    else:
        metrics = payload.get("metrics")
        if metrics is None:
            raise SystemExit("Quipper metrics payload missing 'metrics'.")
        json.dump({"metrics": metrics}, sys.stdout)


def invoke_quipper(case: str, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    binary = ensure_binary(case)
    mode_flag = "--simulate-json" if mode == "simulate" else "--metrics-json"
    command = (
        "source ~/.zshrc-quipper"
        + f" && cd {shlex.quote(str(ROOT))}"
        + f" && {shlex.quote(str(binary))} {mode_flag}"
    )
    if case.startswith("shors") and "params" not in config:
        config = {"params": config}
    proc = subprocess.run(
        ["arch", "-x86_64", "zsh", "--login", "-c", command],
        input=json.dumps(config).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Quipper binary failed ({case}, mode={mode}): {proc.stderr.decode().strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            f"Invalid JSON from Quipper binary ({case}, mode={mode}): {proc.stdout.decode(errors='ignore')}"
        ) from exc


def process_amplitudes(case: str, config: Dict[str, Any], amplitudes: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Slice Quipper amplitudes to match harness expectations (postselect for LCU)."""

    def bits_to_index_be(bits: str) -> int:
        acc = 0
        for ch in bits:
            acc = (acc << 1) | (1 if ch == "1" else 0)
        return acc

    amps = list(amplitudes)
    if not amps:
        raise SystemExit("Empty amplitude list from Quipper.")

    bitlen = len(amps[0].get("bitstring", ""))
    if any(len(a.get("bitstring", "")) != bitlen for a in amps):
        raise SystemExit("Inconsistent bitstring lengths in amplitudes.")

    # get correct config depending on what we are testing
    if case.startswith("shors"):
        t = int(config.get("t", 6))
        N = int(config.get("N", 21))
        m = int(math.ceil(math.log2(N)))
        num_sites = t + m
    else:
        num_sites = int(config.get("num_sites", 0))

    if case in {"tfim_lcu", "heis_lcu"}:
        ancilla = bitlen - num_sites
        if ancilla > 0:
            selector_bits = ancilla - 1  # remaining qubit is the phase wire (last)
            if selector_bits < 0:
                raise SystemExit("Invalid ancilla layout for LCU circuit.")
            slice_vec = [0j] * (2**num_sites)
            for amp in amps:
                bits = amp["bitstring"]
                system_bits = bits[:num_sites]
                selector_str = bits[num_sites : num_sites + selector_bits]
                phase_bit = bits[-1]
                if phase_bit != "1":
                    continue
                if any(b == "1" for b in selector_str):
                    continue
                idx = bits_to_index_be(system_bits)
                slice_vec[idx] = complex(float(amp["re"]), float(amp["im"]))
            norm = sum(abs(z) ** 2 for z in slice_vec) ** 0.5
            if norm == 0:
                raise SystemExit("Postselection slice has zero norm.")
            slice_vec = [z / norm for z in slice_vec]
            return [{"re": float(z.real), "im": float(z.imag)} for z in slice_vec]
        # No ancillas present: fall through to dense path.
    
    if bitlen != num_sites:
        raise SystemExit(
            f"Amplitude bitstrings do not match expected num_sites: got bitlen={bitlen}, expected={num_sites}"
        )


    dense = [0j] * (2**num_sites)
    for amp in amps:
        bits = amp["bitstring"]
        if len(bits) != num_sites:
            raise SystemExit("Amplitude bitstrings do not match num_sites.")
        idx = bits_to_index_be(bits)
        dense[idx] = complex(float(amp["re"]), float(amp["im"]))
    return [{"re": float(z.real), "im": float(z.imag)} for z in dense]


def ensure_binary(case: str) -> Path:
    source = HS_SOURCES.get(case)
    if source is None:
        raise SystemExit(f"Unknown Quipper case '{case}'")
    source_path = QUIPPER_DIR / source
    bin_path = BIN_DIR / case
    latest_source_mtime = max(path.stat().st_mtime for path in all_sources())
    if (
        not bin_path.exists()
        or bin_path.stat().st_mtime < latest_source_mtime
    ):
        compile_case(source_path, bin_path)
    return bin_path


def compile_case(source_path: Path, bin_path: Path) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    command = (
        "source ~/.zshrc-quipper"
        + f" && cd {shlex.quote(str(ROOT))}"
        + " && ghc"
        + f" -i{shlex.quote(str(QUIPPER_DIR))}"
        + f" -outputdir {shlex.quote(str(BUILD_DIR))}"
        + f" -odir {shlex.quote(str(BUILD_DIR))}"
        + f" -hidir {shlex.quote(str(BUILD_DIR))}"
        + " -package containers -package random -package newsynth"
        + " -O2"
        + f" {shlex.quote(str(source_path))}"
        + f" -o {shlex.quote(str(bin_path))}"
    )
    proc = subprocess.run(
        ["arch", "-x86_64", "zsh", "--login", "-c", command],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode().strip()
        raise SystemExit(f"GHC failed while compiling {source_path.name}:\n{stderr}")


def all_sources() -> List[Path]:
    return sorted(QUIPPER_DIR.glob("*.hs"))


if __name__ == "__main__":
    main()
