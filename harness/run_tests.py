"""Cross-language Hamiltonian simulation harness."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from harness.reference_hamiltonians import (
        heis_xxx_hamiltonian,
        tfim_hamiltonian,
        time_evolve,
        zero_state,
    )
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
        state = payload.get("statevector")
        if state is None:
            raise RuntimeError("CLI result missing 'statevector' key")
        vec = np.array([complex(entry["re"], entry["im"]) for entry in state], dtype=np.complex128)
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


def compute_reference(case: Case) -> np.ndarray:
    cfg = case.config
    num_sites = int(cfg["num_sites"])
    params = cfg.get("params", {})
    total_time = float(cfg["time"])
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 0.5))
    field = float(params.get("field", 0.2))

    if case.model == "tfim":
        H = tfim_hamiltonian(num_sites, J, h)
    else:
        H = heis_xxx_hamiltonian(num_sites, J, field)

    psi0 = zero_state(num_sites)
    return time_evolve(H, psi0, total_time)


CASES: List[Case] = [
    Case(
        name="tfim_trotter",
        model="tfim",
        config={
            "num_sites": 2,
            "time": 0.1,
            "params": {"J": 1.0, "h": 0.5, "trotter_steps": 4},
        },
    ),
    Case(
        name="tfim_lcu",
        model="tfim",
        config={
            "num_sites": 2,
            "time": 0.1,
            "params": {"J": 1.0, "h": 0.5, "lcu_precision": 1e-2},
        },
    ),
    Case(
        name="heis_trotter",
        model="heis",
        config={
            "num_sites": 2,
            "time": 0.1,
            "params": {"J": 0.8, "field": 0.2, "trotter_steps": 4},
        },
    ),
    Case(
        name="heis_lcu",
        model="heis",
        config={
            "num_sites": 2,
            "time": 0.1,
            "params": {"J": 0.8, "field": 0.2, "lcu_precision": 1e-2},
        },
    ),
]

CASE_SUFFIX = {
    "tfim_trotter": "tfim_trotter",
    "tfim_lcu": "tfim_lcu",
    "heis_trotter": "heis_trotter",
    "heis_lcu": "heis_lcu",
}

PYTHON_BASES = {
    "cirq": "programs.cirq",
    "hml": "programs.hml",
    "openqasm": "programs.openqasm",
    "pennylane": "programs.pennylane",
    "pyquil": "programs.pyquil",
    "qsharp": "programs.qsharp",
    "qiskit": "programs.qiskit",
    "qrisp": "programs.qrisp",
    "qualtran": "programs.qualtran",
    "tket": "programs.tket",
    "strawberryfields": "programs.strawberryfields",
}

ADAPTERS: Dict[str, Dict[str, Adapter]] = {}
for lang, base in PYTHON_BASES.items():
    for case, suffix in CASE_SUFFIX.items():
        ADAPTERS.setdefault(lang, {})[case] = PythonAdapter(f"{base}.{suffix}")

CLI_LANGUAGES = {
    "silq": [sys.executable, str(ROOT / "programs" / "silq" / "run_cli.py")],
    "quipper": [sys.executable, str(ROOT / "programs" / "quipper" / "run_cli.py")],
}

for lang, cmd in CLI_LANGUAGES.items():
    for case in CASES:
        ADAPTERS.setdefault(lang, {})[case.name] = CliAdapter(cmd + [case.name])

NON_PYTHON: Dict[str, str] = {}

for lang, reason in NON_PYTHON.items():
    ADAPTERS[lang] = {case.name: UnavailableAdapter(reason) for case in CASES}

ALL_LANGUAGES = sorted(ADAPTERS.keys())


@dataclass
class Result:
    language: str
    case: str
    success: bool
    fidelity: Optional[float]
    message: str


TOLERANCE = 1e-6


def main():
    results: List[Result] = []
    for language in ALL_LANGUAGES:
        for case in CASES:
            adapter = ADAPTERS.get(language, {}).get(case.name)
            if adapter is None:
                results.append(Result(language, case.name, False, None, "No adapter"))
                continue
            try:
                state = adapter.run(case.config)
                expected = compute_reference(case)
                fidelity = compute_fidelity(state, expected)
                success = fidelity >= 1 - TOLERANCE
                message = "ok" if success else "low fidelity"
                results.append(Result(language, case.name, success, fidelity, message))
            except Exception as exc:
                results.append(Result(language, case.name, False, None, str(exc)))
    print_summary(results)


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
    header = f"{'Language':<20}{'Case':<15}{'Status':<8}{'Fidelity':<12}Message"
    print(header)
    print("-" * len(header))
    for res in results:
        status = "PASS" if res.success else "FAIL"
        fid_str = f"{res.fidelity:.4f}" if res.fidelity is not None else "-"
        print(f"{res.language:<20}{res.case:<15}{status:<8}{fid_str:<12}{res.message}")


if __name__ == "__main__":
    main()
