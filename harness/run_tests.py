"""Cross-language Hamiltonian simulation harness."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, leave=True):
        return x

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from harness.reference_hamiltonians import (
        heis_xxx_hamiltonian,
        tfim_hamiltonian,
        time_evolve,
        zero_state,
    )
    # from harness.reference_shors import (
    #     make_shors
    # )
except ImportError:  # pragma: no cover
    import pathlib
    import sys

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from harness.reference_hamiltonians import (
        heis_xxx_hamiltonian,
        tfim_hamiltonian,
        time_evolve,
        zero_state,
    )
    from harness.reference_shors import (
        make_shors
    )
    from harness.reference_shors_value import calculate_shors_factors

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
config_home = ROOT / ".config"
config_home.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str(config_home))
PYTKET_CONFIG = ROOT / ".pytket_config"
PYTKET_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("PYTKET_CONFIG_DIR", str(PYTKET_CONFIG))
dotnet_home = ROOT / ".dotnet"
dotnet_home.mkdir(exist_ok=True)
os.environ.setdefault("DOTNET_CLI_HOME", str(dotnet_home))
os.environ.setdefault("DOTNET_SKIP_FIRST_TIME_EXPERIENCE", "1")
os.environ.setdefault("DOTNET_CLI_TELEMETRY_OPTOUT", "1")

CaseConfig = Dict[str, Any]


@dataclass
class Case:
    name: str
    model: str
    config: CaseConfig


class Adapter:
    def run(self, config: CaseConfig) -> np.ndarray:
        raise NotImplementedError


class PythonAdapter(Adapter):
    def __init__(self, module_path: str):
        self.module_path = module_path

    def run(self, config: CaseConfig) -> np.ndarray:
        module = importlib.import_module(self.module_path)
        func = getattr(module, "run_simulation")
        result = func(config)
        return ensure_statevector(result)


class UnavailableAdapter(Adapter):
    def __init__(self, reason: str):
        self.reason = reason

    def run(self, config: CaseConfig) -> np.ndarray:
        raise RuntimeError(self.reason)


class CliAdapter(Adapter):
    def __init__(self, command: List[str], cwd: Optional[str] = None):
        self.command = command
        self.cwd = cwd

    def run(self, config: CaseConfig) -> np.ndarray:
        proc = subprocess.run(
            self.command,
            input=json.dumps(config).encode("utf-8"),
            capture_output=True,
            cwd=self.cwd,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(self.command)} exited with code {proc.returncode}: {proc.stderr.decode().strip()}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Invalid JSON from {' '.join(self.command)}: {proc.stdout.decode(errors='ignore')}"
            ) from exc
        if "value" in payload:
            return payload["value"]
        state = payload.get("statevector")
        if state is None:
            raise RuntimeError("CLI result missing 'statevector' key")
        vec = np.array(
            [complex(entry["re"], entry["im"]) for entry in state], dtype=np.complex128
        )
        return vec


def ensure_statevector(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    for attr in ("state_vector", "ket", "to_numpy"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            vec = fn()
            return np.asarray(vec)
    raise TypeError(f"Cannot convert object of type {type(obj)} to statevector")


def compute_reference(case: Case) -> np.ndarray | list[int]:
    cfg = case.config
    
    if case.model == "tfim":
        num_sites = int(cfg["num_sites"])
        params = cfg.get("params", {})
        total_time = float(cfg["time"])
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 0.5))
        field = float(params.get("field", 0.2))

        H = tfim_hamiltonian(num_sites, J, h)
    elif case.model == "heis":
        num_sites = int(cfg["num_sites"])
        params = cfg.get("params", {})
        total_time = float(cfg["time"])
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 0.5))
        field = float(params.get("field", 0.2))

        H = heis_xxx_hamiltonian(num_sites, J, field)
    elif case.model == "shors21_2":
        # shor's case, get corresponding parameters
        t=cfg.get("t",6) 
        N=cfg.get("N",21)
        a=cfg.get("a",2)
        print("\t**Making Shor's circuit...")
        state = make_shors(t, N, a)
        print("\t**Shor's circuit made")
        return state # final state vector for shor's
    elif case.model == "shors_21_2_value":
        # shor's value case, get corresponding parameters
        t=cfg.get("t",6) 
        N=cfg.get("N",21)
        a=cfg.get("a",2)
        print("\t**Calculating Shor's factors...")
        factors = calculate_shors_factors(N)
        print(f"\t**Expected factors: {factors}")
        
        return factors # return the list of factors for shor's value case
    else:
        raise ValueError(f"Unknown model: {case.model}")

    psi0 = zero_state(num_sites)
    return time_evolve(H, psi0, total_time)


CASES: List[Case] = [
    Case(
        name="tfim_trotter",
        model="tfim",
        config={
            "num_sites": 3,
            "time": 0.3,
            "params": {"J": 1.2, "h": 0.8, "trotter_steps": 4},
        },
    ),
    Case(
        name="tfim_lcu",
        model="tfim",
        config={
            "num_sites": 3,
            "num_ancilla": 4,
            "time": 0.3,
            "params": {"J": 1.2, "h": 0.8, "lcu_precision": 0.1},
        },
    ),
    Case(
        name="heis_trotter",
        model="heis",
        config={
            "num_sites": 3,
            "time": 0.3,
            "params": {"J": 0.8, "field": 0.2, "trotter_steps": 4},
        },
    ),
    Case(
        name="heis_lcu",
        model="heis",
        config={
            "num_sites": 3,
            "num_ancilla": 5,
            "time": 0.3,
            "params": {"J": 0.8, "field": 0.2, "lcu_precision": 0.1},
        },
    ),
    Case(
        name="shors21_2",
        model="shors21_2",
        config={
            "t":6, # keep it small so that our computer don't explode :)
            "N":21,
            "a":2
        }
    ),
    Case(
        name="shors21_2_value",
        model="shors_21_2_value",
        config={
            "t":6, 
            "N":21
        }
    )
]

CASE_SUFFIX = {
    "tfim_trotter": "tfim_trotter",
    "tfim_lcu": "tfim_lcu",
    "heis_trotter": "heis_trotter",
    "heis_lcu": "heis_lcu",
    "shors21_2":"shors", # general shor's file
    "shors21_2_value":"shors_value" # shor's value file
}

PYTHON_BASES = {
    "cirq": "programs.cirq",
    "cudaq": "programs.cudaq",
    "guppy": "programs.guppy",
    "pennylane": "programs.pennylane",
    "pyquil": "programs.pyquil",
    "qsharp": "programs.qsharp",
    "qiskit": "programs.qiskit",
    "qrisp": "programs.qrisp",
    "qualtran": "programs.qualtran",
}

ADAPTERS: Dict[str, Dict[str, Adapter]] = {}
for lang, base in PYTHON_BASES.items():
    for case, suffix in CASE_SUFFIX.items():
        ADAPTERS.setdefault(lang, {})[case] = PythonAdapter(f"{base}.{suffix}")

CLI_LANGUAGES = {
    "silq": [sys.executable, str(ROOT / "programs" / "silq" / "run_cli.py")],
}

for lang, cmd in CLI_LANGUAGES.items():
    for case in CASES:
        ADAPTERS.setdefault(lang, {})[case.name] = CliAdapter(cmd + [case.name])

NON_PYTHON: Dict[str, str] = {}

for lang, reason in NON_PYTHON.items():
    ADAPTERS[lang] = {case.name: UnavailableAdapter(reason) for case in CASES}

ALL_LANGUAGES = sorted(ADAPTERS.keys())
ALL_CASE_NAMES = [case.name for case in CASES]


@dataclass
class Result:
    language: str
    case: str
    success: bool
    fidelity: Optional[float]
    runs: int
    time_mean: Optional[float]
    time_std: Optional[float]
    message: str


TOLERANCE = 1e-2


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-language Hamiltonian simulation tests.")
    parser.add_argument(
        "--languages",
        nargs="+",
        metavar="LANG",
        help="Subset of languages to run (default: all).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help="Subset of cases to run (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available languages and cases, then exit.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of runs for benchmarking (default: 1).",
    )
    parser.add_argument(
        "--json",
        nargs=1,
        metavar="FILE",
        type=pathlib.Path,
        help="Output results to a json file. If FILE already exists, the new results will be merged into the existing file.",
    )
    return parser.parse_args(argv)


def resolve_selection(
    requested: Optional[List[str]], available: List[str], kind: str
) -> List[str]:
    if not requested:
        return available

    # If items in `requested` contain commas, we want to split them
    for item in requested:
        if "," in item:
            requested += item.split(",")
    requested = [item for item in requested if "," not in item]

    unknown = [item for item in requested if item not in available]
    if unknown:
        raise SystemExit(f"Unknown {kind}: {', '.join(unknown)}. Available: {', '.join(available)}")
    # Preserve user-specified order, drop duplicates.
    seen = set()
    ordered: List[str] = []
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.runs <= 0:
        raise ValueError("Number of runs must be at least 1.")

    if args.list:
        print("Languages:", ", ".join(ALL_LANGUAGES))
        print("Cases:", ", ".join(ALL_CASE_NAMES))
        return

    selected_languages = resolve_selection(args.languages, ALL_LANGUAGES, "language")
    selected_cases_names = resolve_selection(args.cases, ALL_CASE_NAMES, "case")
    selected_cases = [case for case in CASES if case.name in selected_cases_names]

    results: List[Result] = []
    for language in selected_languages:
        for case in selected_cases:
            print(f"Running \'{case.name}\' in \'{language}\'")
            # Handle N/A cases up front (delegations or partial implementations).
            adapter = ADAPTERS.get(language, {}).get(case.name)
            if adapter is None:
                results.append(Result(language, case.name, False, None, 0, None, None, "No adapter"))
                continue
            try:
                times = []
                if args.runs > 1:
                    state = adapter.run(case.config)  # Unmeasured warmup run
                else:
                    state = None
                for _ in tqdm(range(args.runs), leave=False):
                    start = time.perf_counter()
                    state = adapter.run(case.config)
                    end = time.perf_counter()
                    times.append(end - start)
                assert state is not None
                times = np.array(times)
                time_mean = float(np.mean(times))
                time_std = float(np.std(times))

                expected = compute_reference(case)
                print("\t**Computing fidelity...")
                if isinstance(expected, list):
                    # shor's factoring
                    success = False
                    if state in expected:
                        print(f"\t**Factor from shor: {state}")
                        success = True
                    message = f"ok" if success else f"incorrect value {state}, expected one of {expected}"
                    fidelity = 1.0 if success else 0.0
                    results.append(Result(language, case.name, success, fidelity, args.runs, time_mean, time_std, message))
                    continue
                else:
                    fidelity = compute_fidelity(state, expected)
                    success = fidelity >= 1 - TOLERANCE
                    message = "ok" if success else "low fidelity"
                    results.append(Result(language, case.name, success, fidelity, args.runs, time_mean, time_std, message))
            except Exception as exc:
                results.append(Result(language, case.name, False, None, 0, None, None, str(exc)))
    print_summary(results)

    if args.json is not None:
        filepath = args.json[0]

        old_results = dict()
        if filepath.is_file():
            with open(filepath, 'r') as f:
                old_results = json.load(f)

        with open(filepath, 'w') as f:
            new_results = dict()
            for res in results:
                new_results[res.language + "/" + res.case] = asdict(res)
            json.dump(old_results | new_results, f, indent=2)


def compute_fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    state = normalize(state)
    reference = normalize(reference)
    return float(np.abs(np.vdot(reference, state)) ** 2)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128).flatten()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero vector")
    return vec / norm


def print_summary(results: List[Result]):
    print("\n")
    header = f"{'Language':<20}{'Case':<16}{'Status':<8}{'Fidelity':<12}{'Execution Time':<16}Message"
    print(header)
    print("-" * len(header))
    for res in results:
        if res.message.startswith("N/A:"):
            status = "N/A"
        else:
            status = "PASS" if res.success else "FAIL"
        fid_str = f"{res.fidelity:.4f}" if res.fidelity is not None else "-"
        time_str = f"{res.time_mean:.4f}±{res.time_std:.4f}" \
            if (res.time_mean is not None and res.time_std is not None) else "-"
        print(f"{res.language:<20}{res.case:<16}{status:<8}{fid_str:<12}{time_str:<16}{res.message}")


if __name__ == "__main__":
    main()
