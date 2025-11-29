#!/usr/bin/env python3
"""CLI adapter for Quipper circuits used by the benchmarking harness."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
QUIPPER_DIR = ROOT / "programs" / "quipper"
BUILD_DIR = QUIPPER_DIR / "build"
BIN_DIR = QUIPPER_DIR / "bin"

HS_SOURCES = {
    "tfim_trotter": "tfim_trotter.hs",
    "tfim_lcu": "tfim_lcu.hs",
    "heis_trotter": "heis_trotter.hs",
    "heis_lcu": "heis_lcu.hs",
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
        state = amplitudes_to_statevector(amplitudes)
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


def amplitudes_to_statevector(amplitudes: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    def entry_index(item: Dict[str, Any]) -> int:
        value = item.get("index")
        return int(value) if value is not None else 0

    ordered = sorted(amplitudes, key=entry_index)
    state = []
    for amp in ordered:
        try:
            real = float(amp["re"])
            imag = float(amp["im"])
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Malformed amplitude entry: {amp}") from exc
        state.append({"re": real, "im": imag})
    return state


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
