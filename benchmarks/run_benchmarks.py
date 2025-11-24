#!/usr/bin/env python3
"""Comprehensive benchmarking harness for Hamiltonian simulation programs.

This script iterates over every language × case combination defined in
``harness.run_tests`` (currently 13 languages and 4 benchmark cases), invokes the
existing correctness adapters, extracts circuit metrics when possible, and
writes the aggregated results to JSON or CSV following
``benchmarks/benchmark_results_schema.json``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.run_tests import (  # type: ignore  # noqa: E402
    ADAPTERS,
    CASES,
    Case,
    CliAdapter,
    PythonAdapter,
    UnavailableAdapter,
)

from programs.common import pauli_models  # type: ignore  # noqa: E402


LANGUAGE_ALIASES = {
    "openqasm": "cirq",
    "hml": "cirq",
    "silq": "cirq",
    "quipper": "qiskit",
}


@dataclass
class BenchmarkResult:
    benchmark_id: str
    language: str
    hamiltonian: str
    method: str
    system_size: int
    parameter_set: Dict[str, Any]
    implementation_path: str
    backend: Optional[str]
    compilation_time_seconds: Optional[float]
    execution_time_seconds: Optional[float]
    total_gate_count: Optional[int]
    two_qubit_gate_count: Optional[int]
    circuit_depth: Optional[int]
    qubit_count: Optional[int]
    native_gate_set: Optional[Iterable[str]]
    timestamp_utc: str
    tool_versions: Dict[str, str]
    status: str
    notes: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hamiltonian simulation benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "latest_results.json",
        help="Output file path (JSON or CSV depending on --format).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Serialization format for collected results.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(sorted(ADAPTERS.keys())),
        help="Comma-separated subset of languages to benchmark (default: all).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(case.name for case in CASES),
        help="Comma-separated subset of cases (tfim_trotter, tfim_lcu, heis_trotter, heis_lcu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_languages = {lang.strip().lower() for lang in args.languages.split(",") if lang.strip()}
    selected_cases = {case.strip() for case in args.cases.split(",") if case.strip()}

    case_lookup = {case.name: case for case in CASES if case.name in selected_cases}
    timestamp = datetime.now(timezone.utc).isoformat()

    results: List[BenchmarkResult] = []
    for language in sorted(selected_languages):
        adapters = ADAPTERS.get(language)
        if adapters is None:
            for case_name in case_lookup:
                results.append(
                    empty_result(
                        language=language,
                        case=case_lookup[case_name],
                        timestamp=timestamp,
                        status="skipped",
                        notes="Language not registered in harness adapters.",
                    )
                )
            continue
        for case_name, case in case_lookup.items():
            adapter = adapters.get(case_name)
            result = run_single_benchmark(language, case, adapter, timestamp)
            results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        payload = [asdict(result) for result in results]
        args.output.write_text(json.dumps(payload, indent=2))
    else:
        rows = [asdict(result) for result in results]
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(results)} benchmark rows to {args.output}")


def run_single_benchmark(language: str, case: Case, adapter, timestamp: str) -> BenchmarkResult:
    config = copy.deepcopy(case.config)
    hamiltonian = "TFIM" if case.name.startswith("tfim") else "HeisenbergXXX"
    method = "Trotter" if "trotter" in case.name else "LCU"
    system_size = int(config.get("num_sites", 0))
    parameter_set = {"time": config.get("time")}
    parameter_set.update(config.get("params", {}))
    impl_path = resolve_implementation_path(language, case.name)

    backend_name = None
    compilation_time = None
    gate_metrics: Dict[str, Any] = {}
    metric_note: Optional[str] = None

    metrics, metric_note = collect_metrics(language, case.name, config)
    if metrics:
        backend_name = metrics.get("backend")
        compilation_time = metrics.get("compilation_time_seconds")
        gate_metrics = metrics

    notes = metric_note

    if adapter is None:
        status = "skipped"
        if notes:
            notes = f"{notes} | Adapter unavailable"
        else:
            notes = "Adapter unavailable"
        return empty_result(language, case, timestamp, status=status, notes=notes)

    if isinstance(adapter, UnavailableAdapter):
        status = "skipped"
        reason = getattr(adapter, "reason", "Unavailable adapter")
        if notes:
            notes = f"{notes} | {reason}"
        else:
            notes = reason
        return empty_result(language, case, timestamp, status=status, notes=notes)

    run_start = perf_counter()
    try:
        adapter.run(config)
        execution_time = perf_counter() - run_start
        status = "ok"
        message = notes
    except Exception as exc:  # pragma: no cover - defensive logging
        execution_time = perf_counter() - run_start
        status = "error"
        message = f"Execution failed: {exc}"
        if notes:
            message = f"{notes} | {message}"
        notes = message
        return BenchmarkResult(
            benchmark_id=build_benchmark_id(language, case, config),
            language=language,
            hamiltonian=hamiltonian,
            method=method,
            system_size=system_size,
            parameter_set=parameter_set,
            implementation_path=impl_path,
            backend=backend_name,
            compilation_time_seconds=compilation_time,
            execution_time_seconds=execution_time,
            total_gate_count=gate_metrics.get("total_gate_count"),
            two_qubit_gate_count=gate_metrics.get("two_qubit_gate_count"),
            circuit_depth=gate_metrics.get("circuit_depth"),
            qubit_count=gate_metrics.get("qubit_count"),
            native_gate_set=gate_metrics.get("native_gate_set"),
            timestamp_utc=timestamp,
            tool_versions=detect_tool_versions(language),
            status=status,
            notes=message,
        )

    return BenchmarkResult(
        benchmark_id=build_benchmark_id(language, case, config),
        language=language,
        hamiltonian=hamiltonian,
        method=method,
        system_size=system_size,
        parameter_set=parameter_set,
        implementation_path=impl_path,
        backend=backend_name,
        compilation_time_seconds=compilation_time,
        execution_time_seconds=execution_time,
        total_gate_count=gate_metrics.get("total_gate_count"),
        two_qubit_gate_count=gate_metrics.get("two_qubit_gate_count"),
        circuit_depth=gate_metrics.get("circuit_depth"),
        qubit_count=gate_metrics.get("qubit_count"),
        native_gate_set=gate_metrics.get("native_gate_set"),
        timestamp_utc=timestamp,
        tool_versions=detect_tool_versions(language),
        status=status,
        notes=notes,
    )


def build_benchmark_id(language: str, case: Case, config: Dict[str, Any]) -> str:
    params = config.get("params", {})
    parts = [language, case.name, f"n{config.get('num_sites', 0)}"]
    for key in sorted(params):
        parts.append(f"{key}{params[key]}")
    return "-".join(parts)


def resolve_implementation_path(language: str, case_name: str) -> str:
    base = ROOT / "programs" / language
    candidates = [base / f"{case_name}.py", base / f"{case_name}.qs", base / f"{case_name}.slq", base / f"{case_name}.hs"]
    for path in candidates:
        if path.exists():
            return str(path.relative_to(ROOT))
    if (base / "run_cli.py").exists():
        return str((base / "run_cli.py").relative_to(ROOT))
    return str(base.relative_to(ROOT))


def empty_result(language: str, case: Case, timestamp: str, status: str, notes: Optional[str]) -> BenchmarkResult:
    config = case.config
    hamiltonian = "TFIM" if case.name.startswith("tfim") else "HeisenbergXXX"
    method = "Trotter" if "trotter" in case.name else "LCU"
    system_size = int(config.get("num_sites", 0))
    parameter_set = {"time": config.get("time")}
    parameter_set.update(config.get("params", {}))
    return BenchmarkResult(
        benchmark_id=build_benchmark_id(language, case, config),
        language=language,
        hamiltonian=hamiltonian,
        method=method,
        system_size=system_size,
        parameter_set=parameter_set,
        implementation_path=resolve_implementation_path(language, case.name),
        backend=None,
        compilation_time_seconds=None,
        execution_time_seconds=None,
        total_gate_count=None,
        two_qubit_gate_count=None,
        circuit_depth=None,
        qubit_count=None,
        native_gate_set=None,
        timestamp_utc=timestamp,
        tool_versions=detect_tool_versions(language),
        status=status,
        notes=notes,
    )


def collect_metrics(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    provider_name = LANGUAGE_ALIASES.get(language, language)
    provider = METRIC_REGISTRY.get(provider_name)
    if provider is None:
        return {}, f"No metric extractor registered for {language}"
    try:
        return provider(language, case_name, config)
    except Exception as exc:  # pragma: no cover - data collection shouldn't fail benchmarking
        return {}, f"Metric extraction failed: {exc}"


# --------------------------------------------------------------------------------------
# Metric extraction helpers
# --------------------------------------------------------------------------------------

def cirq_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import cirq  # type: ignore
        from programs.cirq import common as cirq_common  # type: ignore
        from programs.cirq import tfim_lcu as cirq_tfim_lcu  # type: ignore
        from programs.cirq import heis_lcu as cirq_heis_lcu  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Cirq stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    start = perf_counter()
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, qubits = cirq_common.trotterize_tfim(
            num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0)), time_total, steps
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, qubits = cirq_common.trotterize_heisenberg_xxx(
            num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2)), time_total, steps
        )
    elif case_name == "tfim_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.tfim_pauli_terms(num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, qubits, _, _ = cirq_tfim_lcu._build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
    elif case_name == "heis_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.heisenberg_pauli_terms(num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, qubits, _ = cirq_heis_lcu._build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
    else:
        return {}, f"Unsupported case {case_name} for Cirq metrics"
    compilation_time = perf_counter() - start
    metrics = describe_cirq_circuit(circuit, qubits)
    metrics["backend"] = "cirq.Simulator"
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


def qiskit_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        from qiskit import transpile  # type: ignore
        from qiskit.providers.fake_provider import FakeJakarta  # type: ignore
        from programs.qiskit import common as qiskit_common  # type: ignore
        from programs.qiskit import lcu_common as qiskit_lcu_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Qiskit stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    backend = FakeJakarta()
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = qiskit_common.trotterize_tfim(
            num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0)), time_total, steps
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = qiskit_common.trotterize_heisenberg_xxx(
            num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2)), time_total, steps
        )
    elif case_name == "tfim_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.tfim_pauli_terms(num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, _ = qiskit_lcu_common.build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
    elif case_name == "heis_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.heisenberg_pauli_terms(num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, _ = qiskit_lcu_common.build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
    else:
        return {}, f"Unsupported case {case_name} for Qiskit metrics"
    start = perf_counter()
    compiled = transpile(circuit, backend=backend, optimization_level=2)
    compilation_time = perf_counter() - start
    metrics = describe_qiskit_circuit(compiled)
    metrics["backend"] = backend.name()
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


def tket_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        from pytket.extensions.qiskit import tk_to_qiskit  # type: ignore
        from qiskit import transpile  # type: ignore
        from qiskit.providers.fake_provider import FakeJakarta  # type: ignore
        from programs.tket import common as tket_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"pytket stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = tket_common.trotterize_tfim(
            num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0)), time_total, steps
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = tket_common.trotterize_heisenberg_xxx(
            num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2)), time_total, steps
        )
    else:
        return {}, f"TKET metric extractor not implemented for {case_name}"
    qc = tk_to_qiskit(circuit)
    backend = FakeJakarta()
    start = perf_counter()
    compiled = transpile(qc, backend=backend, optimization_level=2)
    compilation_time = perf_counter() - start
    metrics = describe_qiskit_circuit(compiled)
    metrics["backend"] = backend.name()
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


def pyquil_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        from pyquil import Program  # type: ignore
        from pyquil.quilbase import Gate  # type: ignore
        from programs.pyquil import common as pyquil_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"PyQuil stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        prog, _ = pyquil_common.trotterize_tfim(
            num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0)), time_total, steps
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        prog, _ = pyquil_common.trotterize_heisenberg_xxx(
            num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2)), time_total, steps
        )
    else:
        return {}, f"PyQuil metric extractor not implemented for {case_name}"
    return describe_pyquil_program(prog, Gate), None


def pennylane_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import pennylane as qml  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"PennyLane unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    steps = int(params.get("trotter_steps", 2))

    if case_name == "tfim_trotter":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 1.0))

        def build():
            dt = time_total / steps
            for _ in range(steps):
                for i in range(num_sites - 1):
                    qml.PauliRot(2 * J * dt, wires=[i, i + 1], pauli_word="ZZ")
                for i in range(num_sites):
                    qml.PauliRot(2 * h * dt, wires=[i], pauli_word="X")

    elif case_name == "heis_trotter":
        J = float(params.get("J", 1.0))
        field = float(params.get("field", 0.2))

        def build():
            dt = time_total / steps
            for _ in range(steps):
                for i in range(num_sites - 1):
                    for axis in ("XX", "YY", "ZZ"):
                        qml.PauliRot(2 * J * dt, wires=[i, i + 1], pauli_word=axis)
                for i in range(num_sites):
                    qml.PauliRot(2 * field * dt, wires=[i], pauli_word="Z")

    else:
        return {}, f"PennyLane metric extractor not implemented for {case_name}"

    dev = qml.device("default.qubit", wires=num_sites)

    @qml.qnode(dev)
    def circuit():
        build()
        return qml.state()

    specs = qml.specs(circuit)()
    resources = specs.get("resources")
    total_ops = getattr(resources, "num_gates", None) if resources else None
    two_qubit = None
    depth = getattr(resources, "depth", None) if resources else None
    gate_names = None
    qubit_count = num_sites
    if resources is not None:
        sizes = getattr(resources, "gate_sizes", {})
        two_qubit = sizes.get(2)
        gate_types = getattr(resources, "gate_types", {})
        gate_names = sorted(gate_types.keys()) if gate_types else None
        qubit_count = getattr(resources, "num_wires", num_sites)
    metrics = {
        "backend": "qml.default.qubit",
        "compilation_time_seconds": specs.get("execution_time", None),
        "total_gate_count": total_ops,
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": depth,
        "qubit_count": qubit_count,
        "native_gate_set": gate_names,
    }
    return metrics, None


def strawberryfields_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import strawberryfields as sf  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {}, f"Strawberry Fields dependency unavailable: {exc}"
    num_qubits = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    prog, note = build_strawberryfields_program(case_name, num_qubits, time_total, params)
    if prog is None:
        return {}, note
    total_ops = len(prog.circuit)
    two_mode_ops = sum(1 for cmd in prog.circuit if len(cmd.reg) == 2)
    metrics = {
        "backend": "sf.Engine(fock)",
        "compilation_time_seconds": None,
        "total_gate_count": total_ops,
        "two_qubit_gate_count": two_mode_ops,
        "circuit_depth": None,
        "qubit_count": num_qubits,
        "native_gate_set": sorted({type(cmd.op).__name__ for cmd in prog.circuit}),
    }
    return metrics, note


def qrisp_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    # LCU cases delegate to Qiskit, so reuse those metrics.
    if "lcu" in case_name:
        metrics, note = qiskit_metric_provider("qiskit", case_name, config)
        if note:
            note = f"Qrisp LCU via Qiskit fallback: {note}"
        else:
            note = "Metrics via Qiskit fallback"
        return metrics, note
    return {}, "Qrisp native metric extraction pending"


def qualtran_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    if case_name != "tfim_trotter":
        return cirq_metric_provider(language, case_name, config)
    try:
        import cirq  # type: ignore
        from programs.qualtran import common as qualtran_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Qualtran dependencies unavailable: {exc}"
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    steps = int(params.get("trotter_steps", 32))
    init_angle = float(params.get("init_angle", 0.0))
    qubits = cirq.LineQubit.range(num_sites)
    circuit = cirq.Circuit()
    for q in qubits:
        if init_angle != 0.0:
            circuit.append(cirq.ry(init_angle).on(q))
    dt = total_time / steps
    for _ in range(steps):
        qualtran_common.apply_ising_step(
            circuit,
            qubits,
            qualtran_common.IsingZZUnitary(nsites=num_sites, angle=2 * float(params.get("J", 1.0)) * dt),
        )
        qualtran_common.apply_ising_step(
            circuit,
            qubits,
            qualtran_common.IsingXUnitary(nsites=num_sites, angle=2 * float(params.get("h", 1.0)) * dt),
        )
    metrics = describe_cirq_circuit(circuit, qubits)
    metrics["backend"] = "cirq.Simulator"
    metrics["compilation_time_seconds"] = None
    return metrics, None


def placeholder_metric_provider(language: str, case_name: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    return {}, "Placeholder language with no metric extractor"


METRIC_REGISTRY: Dict[str, Callable[[str, str, Dict[str, Any]], Tuple[Dict[str, Any], Optional[str]]]] = {
    "cirq": cirq_metric_provider,
    "qiskit": qiskit_metric_provider,
    "pyquil": pyquil_metric_provider,
    "pennylane": pennylane_metric_provider,
    "tket": tket_metric_provider,
    "strawberryfields": strawberryfields_metric_provider,
    "qrisp": qrisp_metric_provider,
    "qualtran": qualtran_metric_provider,
    "qsharp": placeholder_metric_provider,
}


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def describe_cirq_circuit(circuit: "cirq.Circuit", qubits: Iterable["cirq.Qid"]) -> Dict[str, Any]:
    ops = list(circuit.all_operations())
    two_qubit = sum(1 for op in ops if len(op.qubits) == 2)
    gate_set = sorted({type(op.gate).__name__ if getattr(op, "gate", None) else type(op).__name__ for op in ops})
    return {
        "total_gate_count": len(ops),
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": len(circuit),
        "qubit_count": len(set(qubits)),
        "native_gate_set": gate_set,
    }


def describe_qiskit_circuit(circuit) -> Dict[str, Any]:  # type: ignore[valid-type]
    counts = circuit.count_ops()
    two_qubit_gates = {"cx", "cz", "swap", "iswap", "ecr"}
    two_qubit = sum(int(count) for gate, count in counts.items() if gate.lower() in two_qubit_gates)
    return {
        "total_gate_count": int(sum(counts.values())),
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": int(circuit.depth()),
        "qubit_count": int(circuit.num_qubits),
        "native_gate_set": sorted(counts.keys()),
    }


def describe_pyquil_program(prog, gate_type) -> Dict[str, Any]:  # type: ignore[valid-type]
    instructions = [inst for inst in prog if isinstance(inst, gate_type)]
    total = len(instructions)
    two_qubit = sum(1 for inst in instructions if len(inst.qubits) == 2)
    depth = estimate_gate_depth(instructions)
    qubits = sorted({int(q.index) for inst in instructions for q in inst.qubits})
    gate_set = sorted({inst.name for inst in instructions})
    return {
        "backend": "pyquil.NumpyWavefunctionSimulator",
        "compilation_time_seconds": None,
        "total_gate_count": total,
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": depth,
        "qubit_count": len(qubits),
        "native_gate_set": gate_set,
    }


def estimate_gate_depth(instructions: List["Gate"]) -> int:
    layer: Dict[int, int] = {}
    max_depth = 0
    for inst in instructions:
        qubits = [int(q.index) for q in inst.qubits]
        start = max((layer.get(q, 0) for q in qubits), default=0)
        depth = start + 1
        for q in qubits:
            layer[q] = depth
        max_depth = max(max_depth, depth)
    return max_depth


def build_strawberryfields_program(case_name: str, num_qubits: int, total_time: float, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    try:
        import strawberryfields as sf  # type: ignore
        from programs.strawberryfields import common as sf_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, f"Strawberry Fields helpers unavailable: {exc}"
    prog = sf.Program(2 * num_qubits)
    with prog.context as q:
        for i in range(num_qubits):
            sf.ops.Fock(1) | q[2 * i]
            sf.ops.Fock(0) | q[2 * i + 1]
        if case_name == "tfim_trotter":
            steps = int(params.get("trotter_steps", 10))
            dt = total_time / steps
            theta_x = float(params.get("h", 1.0)) * dt
            theta_zz = float(params.get("J", 1.0)) * dt
            for _ in range(steps):
                for i in range(num_qubits - 1):
                    sf_common.apply_logical_zz(q, i, i + 1, theta_zz)
                for i in range(num_qubits):
                    sf_common.apply_logical_x(q, i, theta_x)
        elif case_name == "heis_trotter":
            steps = int(params.get("trotter_steps", 10))
            dt = total_time / steps
            coupling = float(params.get("J", 1.0)) * dt
            field = float(params.get("field", 0.2)) * dt
            for _ in range(steps):
                for i in range(num_qubits - 1):
                    sf_common.apply_logical_x(q, i, coupling)
                    sf_common.apply_logical_x(q, i + 1, coupling)
                    sf_common.apply_logical_zz(q, i, i + 1, coupling)
                for i in range(num_qubits):
                    sf_common.apply_logical_z(q, i, field)
        elif case_name == "tfim_lcu":
            terms = []
            coupling = float(params.get("J", 1.0)) * total_time
            field = float(params.get("h", 1.0)) * total_time
            for i in range(num_qubits):
                terms.append(("X", i, None, field))
            for i in range(num_qubits - 1):
                terms.append(("ZZ", i, i + 1, coupling))
            for kind, idx, jdx, angle in terms:
                if kind == "X":
                    sf_common.apply_logical_x(q, idx, angle)
                elif kind == "ZZ" and jdx is not None:
                    sf_common.apply_logical_zz(q, idx, jdx, angle)
        elif case_name == "heis_lcu":
            coupling = float(params.get("J", 1.0)) * total_time
            field = float(params.get("field", 0.2)) * total_time
            for i in range(num_qubits - 1):
                sf_common.apply_logical_x(q, i, coupling)
                sf_common.apply_logical_x(q, i + 1, coupling)
                sf_common.apply_logical_zz(q, i, i + 1, coupling)
            for i in range(num_qubits):
                sf_common.apply_logical_z(q, i, field)
        else:
            return None, f"Strawberry Fields metrics not implemented for {case_name}"
    return prog, None


def detect_tool_versions(language: str) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    if language in {"cirq", "openqasm", "hml", "silq", "qualtran"}:
        try:
            import cirq  # type: ignore

            versions["cirq"] = getattr(cirq, "__version__", "unknown")
        except Exception:
            pass
    if language in {"qiskit", "quipper", "qrisp"}:
        try:
            import qiskit  # type: ignore

            versions["qiskit"] = getattr(qiskit, "__version__", "unknown")
        except Exception:
            pass
    if language == "pyquil":
        try:
            import pyquil  # type: ignore

            versions["pyquil"] = getattr(pyquil, "__version__", "unknown")
        except Exception:
            pass
    if language == "pennylane":
        try:
            import pennylane as qml  # type: ignore

            versions["pennylane"] = getattr(qml, "__version__", "unknown")
        except Exception:
            pass
    if language == "tket":
        try:
            import pytket  # type: ignore

            versions["pytket"] = getattr(pytket, "__version__", "unknown")
        except Exception:
            pass
    if language == "strawberryfields":
        try:
            import strawberryfields as sf  # type: ignore

            versions["strawberryfields"] = getattr(sf, "__version__", "unknown")
        except Exception:
            pass
    return versions


if __name__ == "__main__":
    main()
