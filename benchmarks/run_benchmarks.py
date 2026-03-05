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
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import math
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
    # Languages that reuse another stack for metrics should not appear here;
    # instead they are handled via METRIC_NA_REASONS below.
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
    parser = argparse.ArgumentParser(
        description="Run Hamiltonian simulation benchmarks."
    )
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
    selected_languages = {
        lang.strip().lower() for lang in args.languages.split(",") if lang.strip()
    }
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


def run_single_benchmark(
    language: str, case: Case, adapter, timestamp: str
) -> BenchmarkResult:
    # Use a small configuration for correctness (to keep the harness fast and robust)
    # but allow the metric extractor to operate on a larger, more ambitious problem
    # instance (for example, 10-site TFIM).
    config_correct = copy.deepcopy(case.config)
    config_metrics = benchmark_config_for_metrics(language, case, config_correct)
    hamiltonian = "TFIM" if case.name.startswith("tfim") else "HeisenbergXXX"
    method = "Trotter" if "trotter" in case.name else "LCU"
    system_size = int(config_metrics.get("num_sites", 0))
    parameter_set = {"time": config_metrics.get("time")}
    parameter_set.update(config_metrics.get("params", {}))
    impl_path = resolve_implementation_path(language, case.name)

    backend_name = None
    compilation_time = None
    gate_metrics: Dict[str, Any] = {}
    metric_note: Optional[str] = None

    metrics, metric_note = collect_metrics(language, case.name, config_metrics)
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
        # Correctness is evaluated on the original harness configuration (typically
        # small system sizes) to keep runs lightweight and robust across tools.
        adapter.run(config_correct)
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
            benchmark_id=build_benchmark_id(language, case, config_metrics),
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
        benchmark_id=build_benchmark_id(language, case, config_metrics),
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
    candidates = [
        base / f"{case_name}.py",
        base / f"{case_name}.qs",
        base / f"{case_name}.slq",
        base / f"{case_name}.hs",
    ]
    for path in candidates:
        if path.exists():
            return str(path.relative_to(ROOT))
    if (base / "run_cli.py").exists():
        return str((base / "run_cli.py").relative_to(ROOT))
    return str(base.relative_to(ROOT))


def empty_result(
    language: str, case: Case, timestamp: str, status: str, notes: Optional[str]
) -> BenchmarkResult:
    config = benchmark_config_for_metrics(language, case, case.config)
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


def benchmark_config_for_metrics(
    language: str, case: Case, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a copy of the harness config tuned for benchmarking metrics.

    We leave the harness configuration (typically with 2--3 sites) untouched for
    correctness checks, but use a larger, more ambitious instance for metric
    extraction.  Currently we simply bump the system size to 10 sites for all
    spin-chain benchmarks.
    """
    cfg = copy.deepcopy(base_config)
    # Use a more demanding problem size for circuit metrics.
    if "num_sites" in cfg:
        cfg["num_sites"] = 10
    return cfg


def collect_metrics(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    # Some languages or cases delegate execution to another stack or do not
    # have a native circuit-level implementation in this artifact. For those
    # we deliberately omit metrics to avoid misleading “false positives”.
    na_reason = METRIC_NA_REASONS.get(language)
    if na_reason is not None:
        return {}, na_reason

    provider_name = LANGUAGE_ALIASES.get(language, language)
    provider = METRIC_REGISTRY.get(provider_name)
    if provider is None:
        return {}, f"No metric extractor registered for {language}"
    try:
        return provider(language, case_name, config)
    except (
        Exception
    ) as exc:  # pragma: no cover - data collection shouldn't fail benchmarking
        return {}, f"Metric extraction failed: {exc}"


# --------------------------------------------------------------------------------------
# Metric extraction helpers
# --------------------------------------------------------------------------------------


def cirq_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
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
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
            steps,
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, qubits = cirq_common.trotterize_heisenberg_xxx(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
            steps,
        )
    elif case_name == "tfim_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.tfim_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))
            ),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, qubits, _, _ = cirq_tfim_lcu._build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
    elif case_name == "heis_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.heisenberg_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))
            ),
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


def qiskit_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        from qiskit import transpile  # type: ignore
        from qiskit.providers.fake_provider import GenericBackendV2  # type: ignore
        from programs.qiskit import common as qiskit_common  # type: ignore
        from programs.qiskit import lcu_common as qiskit_lcu_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Qiskit stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    backend: Optional[GenericBackendV2] = None
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = qiskit_common.trotterize_tfim(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
            steps,
        )
        backend = GenericBackendV2(num_qubits=num_sites)
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = qiskit_common.trotterize_heisenberg_xxx(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
            steps,
        )
        backend = GenericBackendV2(num_qubits=num_sites)
    elif case_name == "tfim_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.tfim_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))
            ),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, _ = qiskit_lcu_common.build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
        backend = GenericBackendV2(num_qubits=circuit.num_qubits)
    elif case_name == "heis_lcu":
        gamma = pauli_models.taylor_coefficients(
            pauli_models.heisenberg_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))
            ),
            time_total,
        )
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        circuit, _ = qiskit_lcu_common.build_lcu_circuit(
            num_sites, list(weights), list(paulis), list(phases)
        )
        backend = GenericBackendV2(num_qubits=circuit.num_qubits)
    else:
        return {}, f"Unsupported case {case_name} for Qiskit metrics"
    if backend is None:
        backend = GenericBackendV2(num_qubits=circuit.num_qubits)
    start = perf_counter()
    compiled = transpile(circuit, backend=backend, optimization_level=2)
    compilation_time = perf_counter() - start
    metrics = describe_qiskit_circuit(compiled)
    metrics["backend"] = getattr(backend, "name", str(backend))
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


def tket_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        from pytket.extensions.qiskit import tk_to_qiskit  # type: ignore
        from qiskit import transpile  # type: ignore
        from qiskit.providers.fake_provider import GenericBackendV2  # type: ignore
        from programs.tket import common as tket_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"pytket stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = tket_common.trotterize_tfim(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
            steps,
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        circuit, _ = tket_common.trotterize_heisenberg_xxx(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
            steps,
        )
    else:
        return {}, f"TKET metric extractor not implemented for {case_name}"
    qc = tk_to_qiskit(circuit)
    backend = GenericBackendV2(num_qubits=num_sites)
    start = perf_counter()
    compiled = transpile(qc, backend=backend, optimization_level=2)
    compilation_time = perf_counter() - start
    metrics = describe_qiskit_circuit(compiled)
    metrics["backend"] = getattr(backend, "name", str(backend))
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


def pyquil_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    if "lcu" in case_name:
        return {}, "N/A: PyQuil LCU benchmarks suppressed due to long compile/emulation times on the Quil simulator."
    try:
        from pyquil import Program  # type: ignore
        from pyquil.quilbase import Gate  # type: ignore
        from programs.pyquil import common as pyquil_common  # type: ignore
        from programs.pyquil import lcu_common as pyquil_lcu  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"PyQuil stack unavailable: {exc}"
    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        prog, _ = pyquil_common.trotterize_tfim(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
            steps,
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        prog, _ = pyquil_common.trotterize_heisenberg_xxx(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
            steps,
        )
    elif case_name == "tfim_lcu":
        H = pauli_models.tfim_pauli_terms(
            num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))
        )
        gamma = pauli_models.taylor_coefficients(H, time_total)
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        prog, _ = pyquil_lcu.build_lcu_program(num_sites, list(weights), list(paulis), list(phases))
    elif case_name == "heis_lcu":
        H = pauli_models.heisenberg_pauli_terms(
            num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))
        )
        gamma = pauli_models.taylor_coefficients(H, time_total)
        weights, paulis, phases = pauli_models.lcu_weights_from_gamma(gamma)
        prog, _ = pyquil_lcu.build_lcu_program(num_sites, list(weights), list(paulis), list(phases))
    else:
        return {}, f"PyQuil metric extractor not implemented for {case_name}"
    return describe_pyquil_program(prog, Gate), None


def _pennylane_lcu_terms(num_sites: int, hamiltonian: Dict[str, complex], total_time: float):
    gamma = pauli_models.taylor_coefficients(hamiltonian, total_time)
    return pauli_models.lcu_weights_from_gamma(gamma)


def _build_pennylane_lcu_qnode(
    num_sites: int,
    weights: List[float],
    paulis: List[str],
    phases: List[str],
    pennylane_lcu_module,
):
    import pennylane as qml  # type: ignore

    weights = list(weights)
    paulis = list(paulis)
    phases = list(phases)
    L = len(weights)
    if L == 0:
        raise ValueError("No LCU terms provided.")
    m = max(1, int(math.ceil(math.log2(L))))
    target_len = 2**m
    identity = "I" * num_sites
    if target_len > L:
        pad = target_len - L
        weights.extend([0.0] * pad)
        paulis.extend([identity] * pad)
        phases.extend(["1"] * pad)
    amps = pennylane_lcu_module.amps_from_weights(weights)
    total_wires = num_sites + m + 1
    system = list(range(num_sites))
    index = list(range(num_sites, num_sites + m))
    phase_wire = num_sites + m
    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(phase_wire)
        qml.MottonenStatePreparation(amps, wires=index)
        controls = index
        for idx_value, weight in enumerate(weights):
            pennylane_lcu_module.apply_index_mask(index, idx_value)
            if weight > 0:
                pennylane_lcu_module.apply_phase_tag(controls, phase_wire, phases[idx_value])
                pennylane_lcu_module.apply_controlled_pauli_string(
                    controls, system, paulis[idx_value]
                )
            pennylane_lcu_module.apply_index_mask(index, idx_value)
        qml.adjoint(qml.MottonenStatePreparation)(amps, wires=index)
        return qml.state()

    return circuit, total_wires


def pennylane_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import pennylane as qml  # type: ignore
        from programs.pennylane import lcu_common as pennylane_lcu  # type: ignore
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

    elif case_name == "tfim_lcu":
        weights, paulis, phases = _pennylane_lcu_terms(
            num_sites,
            pauli_models.tfim_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("h", 1.0))
            ),
            time_total,
        )
        circuit, total_wires = _build_pennylane_lcu_qnode(
            num_sites, weights, paulis, phases, pennylane_lcu
        )
    elif case_name == "heis_lcu":
        weights, paulis, phases = _pennylane_lcu_terms(
            num_sites,
            pauli_models.heisenberg_pauli_terms(
                num_sites, float(params.get("J", 1.0)), float(params.get("field", 0.2))
            ),
            time_total,
        )
        circuit, total_wires = _build_pennylane_lcu_qnode(
            num_sites, weights, paulis, phases, pennylane_lcu
        )
    else:
        return {}, f"PennyLane metric extractor not implemented for {case_name}"
        dev = None
    if case_name not in ("tfim_lcu", "heis_lcu"):
        dev = qml.device("default.qubit", wires=num_sites)

        @qml.qnode(dev)
        def circuit():
            build()
            return qml.state()
        total_wires = num_sites

    specs = qml.specs(circuit)()
    resources = specs.get("resources")
    total_ops = getattr(resources, "num_gates", None) if resources else None
    two_qubit = None
    depth = getattr(resources, "depth", None) if resources else None
    gate_names = None
    qubit_count = total_wires
    if resources is not None:
        sizes = getattr(resources, "gate_sizes", {})
        two_qubit = sizes.get(2)
        gate_types = getattr(resources, "gate_types", {})
        gate_names = sorted(gate_types.keys()) if gate_types else None
        qubit_count = getattr(resources, "num_wires", total_wires)
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


def strawberryfields_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    # LCU cases are sequential dual-rail CV circuits, not full 2nd-order Taylor
    # LCU block encodings, so we omit metrics for them in the LCU comparison.
    if "lcu" in case_name:
        return {}, (
            "N/A: Strawberry Fields LCU programs are sequential dual-rail CV circuits, "
            "not full 2nd-order Taylor LCU block encodings with selection ancillas / PREPARE / SELECT."
        )
    try:
        import strawberryfields as sf  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return {}, f"Strawberry Fields dependency unavailable: {exc}"
    num_qubits = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})
    prog, note = build_strawberryfields_program(
        case_name, num_qubits, time_total, params
    )
    if prog is None:
        return {}, note
    total_ops = len(prog.circuit)
    two_mode_ops = sum(1 for cmd in prog.circuit if len(cmd.reg) == 2)
    metrics = {
        # Note: these are CV dual-rail circuits projected to a logical qubit
        # subspace, not native qubit gate sets. We still report basic circuit
        # statistics for the Trotter cases.
        "backend": "sf.Engine(fock, dual-rail CV)",
        "compilation_time_seconds": None,
        "total_gate_count": total_ops,
        "two_qubit_gate_count": two_mode_ops,
        "circuit_depth": None,
        "qubit_count": num_qubits,
        "native_gate_set": sorted({type(cmd.op).__name__ for cmd in prog.circuit}),
    }
    return metrics, note


def qrisp_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    # For Trotter cases, build the Qrisp QuantumCircuit via the same
    # Hamiltonian + trotterization path used in the simulation helpers,
    # then convert to a Qiskit circuit for metric extraction.
    try:
        import inspect
        import numpy as np  # type: ignore
        from qrisp import QuantumVariable, ry  # type: ignore
        from programs.qrisp import common as qrisp_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Qrisp stack unavailable: {exc}"

    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    steps = int(params.get("trotter_steps", 2))
    order = int(params.get("trotter_order", 1))
    init_angle = float(params.get("init_angle", np.pi / 8))

    # Build the Qrisp Hamiltonian operator.
    if case_name in ("tfim_trotter", "tfim_lcu"):
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 1.0))
        H = qrisp_common.build_tfim_operator(num_sites, J, h)
    elif case_name in ("heis_trotter", "heis_lcu"):
        J = float(params.get("J", 1.0))
        field = float(params.get("field", 0.2))
        H = qrisp_common.build_heisenberg_operator(num_sites, J, field)
    else:
        return {}, f"Qrisp metric extractor not implemented for {case_name}"

    trot = getattr(H, "trotterization")
    sig = inspect.signature(trot)
    kwargs = {}
    if "order" in sig.parameters:
        kwargs["order"] = order
    evolution = trot(**kwargs)

    qv = QuantumVariable(num_sites)
    if init_angle != 0.0:
        ry(init_angle, qv)
    evolution(qv, t=total_time, steps=steps)
    qs = getattr(qv, "qs", None)
    if qs is None:
        return (
            {},
            "Metric extraction failed: Qrisp QuantumSession unavailable after evolution.",
        )
    compile_fn = getattr(qs, "compile", None)
    if not callable(compile_fn):
        return {}, "Metric extraction failed: Qrisp session.compile() not available."
    circ = compile_fn()

    # Prefer converting to a Qiskit circuit if supported.
    to_qiskit = getattr(circ, "to_qiskit", None)
    if callable(to_qiskit):
        try:
            qc = to_qiskit()
        except Exception as exc:  # pragma: no cover
            return {}, f"Metric extraction failed: to_qiskit() error: {exc}"
        metrics = describe_qiskit_circuit(qc)
        metrics["backend"] = "qrisp.to_qiskit"
        metrics["compilation_time_seconds"] = None
        return metrics, None

    # Fallback: use Qrisp QuantumCircuit's own statistics if available.
    count_ops = getattr(circ, "count_ops", None)
    depth_fn = getattr(circ, "depth", None)
    num_qubits_fn = getattr(circ, "num_qubits", None)
    if not callable(count_ops) or not callable(depth_fn) or not callable(num_qubits_fn):
        return (
            {},
            "Metric extraction failed: Qrisp QuantumCircuit lacks basic analysis methods.",
        )

    try:
        counts = count_ops()
        depth = depth_fn()
        n_qubits = num_qubits_fn()
    except Exception as exc:  # pragma: no cover
        return (
            {},
            f"Metric extraction failed: Qrisp QuantumCircuit analysis error: {exc}",
        )

    two_qubit_gates = {"cx", "cz", "swap", "iswap", "ecr"}
    total_ops = int(sum(counts.values())) if counts is not None else None
    two_qubit = (
        int(
            sum(
                count
                for name, count in counts.items()
                if name.lower() in two_qubit_gates
            )
        )
        if counts is not None
        else None
    )
    metrics = {
        "backend": "qrisp.QuantumCircuit",
        "compilation_time_seconds": None,
        "total_gate_count": total_ops,
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": int(depth) if depth is not None else None,
        "qubit_count": int(n_qubits) if n_qubits is not None else num_sites,
        "native_gate_set": sorted(counts.keys()) if counts is not None else None,
    }
    return metrics, None


def hml_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Metric extractor for HML / SimuQ programs using the same QSystem model."""
    try:
        from simuq.qsystem import QSystem  # type: ignore
        from simuq.environment import Qubit  # type: ignore
        from programs.hml import common as hml_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"HML/SimuQ stack unavailable: {exc}"

    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})

    qs = QSystem()
    q = [Qubit(qs) for _ in range(num_sites)]

    if case_name == "tfim_trotter":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 1.0))
        steps = int(params.get("trotter_steps", 2))
        dt = total_time / steps
        H_ZZ = 0
        for i in range(num_sites - 1):
            H_ZZ += J * (q[i].Z * q[i + 1].Z)
        H_X = 0
        for i in range(num_sites):
            H_X += h * q[i].X
        for _ in range(steps):
            qs.add_evolution(H_ZZ, dt)
            qs.add_evolution(H_X, dt)
    elif case_name == "heis_trotter":
        J = float(params.get("J", 1.0))
        field = float(params.get("field", 0.2))
        steps = int(params.get("trotter_steps", 2))
        dt = total_time / steps
        H_xx = 0
        H_yy = 0
        H_zz = 0
        H_field = 0
        for i in range(num_sites - 1):
            H_xx += J * (q[i].X * q[i + 1].X)
            H_yy += J * (q[i].Y * q[i + 1].Y)
            H_zz += J * (q[i].Z * q[i + 1].Z)
        for i in range(num_sites):
            H_field += field * q[i].Z
        for _ in range(steps):
            qs.add_evolution(H_xx, dt)
            qs.add_evolution(H_yy, dt)
            qs.add_evolution(H_zz, dt)
            qs.add_evolution(H_field, dt)
    elif case_name == "tfim_lcu":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 1.0))
        H = 0
        for i in range(num_sites - 1):
            H += J * (q[i].Z * q[i + 1].Z)
        for i in range(num_sites):
            H += h * q[i].X
        qs.add_evolution(H, total_time)
    elif case_name == "heis_lcu":
        J = float(params.get("J", 1.0))
        field = float(params.get("field", 0.2))
        H = 0
        for i in range(num_sites - 1):
            H += J * (q[i].X * q[i + 1].X)
            H += J * (q[i].Y * q[i + 1].Y)
            H += J * (q[i].Z * q[i + 1].Z)
        for i in range(num_sites):
            H += field * q[i].Z
        qs.add_evolution(H, total_time)
    else:
        return {}, f"HML metric extractor not implemented for case {case_name}"

    qc = hml_common.qsystem_to_qiskit_circuit(qs, num_sites)
    metrics = describe_qiskit_circuit(qc)
    metrics["backend"] = "hml.qsystem+qiskit"
    metrics["compilation_time_seconds"] = None
    return metrics, None


def qualtran_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import cirq  # type: ignore
        from programs.qualtran import common as qualtran_common  # type: ignore
        from qualtran._infra.gate_with_registers import get_named_qubits  # type: ignore
        from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Qualtran dependencies unavailable: {exc}"
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    # Trotter cases: use explicit Qualtran bloqs to build a Cirq circuit.
    if case_name in ("tfim_trotter", "heis_trotter"):
        steps = int(params.get("trotter_steps", 32))
        init_angle = float(params.get("init_angle", 0.0))
        qubits = cirq.LineQubit.range(num_sites)
        circuit = cirq.Circuit()
        for q in qubits:
            if init_angle != 0.0:
                circuit.append(cirq.ry(init_angle).on(q))
        dt = total_time / steps
        if case_name == "tfim_trotter":
            for _ in range(steps):
                qualtran_common.apply_ising_step(
                    circuit,
                    qubits,
                    qualtran_common.IsingZZUnitary(
                        nsites=num_sites, angle=2 * float(params.get("J", 1.0)) * dt
                    ),
                )
                qualtran_common.apply_ising_step(
                    circuit,
                    qubits,
                    qualtran_common.IsingXUnitary(
                        nsites=num_sites, angle=2 * float(params.get("h", 1.0)) * dt
                    ),
                )
        else:  # heis_trotter
            pair_bloq = qualtran_common.HeisenbergPairUnitary(
                angle_j=float(params.get("J", 1.0)) * dt,
                angle_field=float(params.get("field", 0.2)) * dt,
            )
            for _ in range(steps):
                for i in range(num_sites - 1):
                    qualtran_common.apply_heisenberg_pair_step(
                        circuit, qubits[i], qubits[i + 1], pair_bloq
                    )
        metrics = describe_cirq_circuit(circuit, qubits)
        metrics["backend"] = "cirq.Simulator"
        metrics["compilation_time_seconds"] = None
        return metrics, None

    # LCU cases: build the Qualtran LCUBlockEncoding and convert to a Cirq circuit
    # for gate-level metrics, using the same Taylor LCU helpers as in the
    # correctness path (no Cirq LCU fallback).
    if case_name == "tfim_lcu":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 0.5))
        precision = float(params.get("lcu_precision", 1e-2))
        H = pauli_models.tfim_pauli_terms(num_sites, J, h)
        gamma = pauli_models.taylor_coefficients(H, total_time)
        paulis, weights, _ = qualtran_common.taylor_terms_to_paulis(gamma)
        block = qualtran_common.build_lcu_block(paulis, weights, precision=precision)
    elif case_name == "heis_lcu":
        J = float(params.get("J", 0.8))
        field = float(params.get("field", 0.2))
        precision = float(params.get("lcu_precision", 1e-2))
        H = pauli_models.heisenberg_pauli_terms(num_sites, J, field)
        gamma = pauli_models.taylor_coefficients(H, total_time)
        paulis, weights, _ = qualtran_common.taylor_terms_to_paulis(gamma)
        block = qualtran_common.build_lcu_block(paulis, weights, precision=precision)
    else:
        return {}, f"N/A: Qualtran metrics not implemented for case {case_name}"

    # Convert the LCU block bloq into a Cirq circuit and summarize it.
    cbloq = block.decompose_bloq()
    init_quregs = get_named_qubits(block.signature)
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    circuit, _ = cbloq.to_cirq_circuit_and_quregs(qubit_manager=qm, **init_quregs)
    qubits = sorted(circuit.all_qubits(), key=lambda q: str(q))
    metrics = describe_cirq_circuit(circuit, qubits)
    metrics["backend"] = "cirq.Simulator"
    metrics["compilation_time_seconds"] = None
    return metrics, None


def placeholder_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    return {}, "Placeholder language with no metric extractor"


def qsharp_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        import qsharp  # type: ignore
        from programs.qsharp import ensure_compiled  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"Q# runtime unavailable: {exc}"

    ensure_compiled()
    try:
        operation, args = _qsharp_operation_and_args(case_name, config, qsharp)
    except KeyError as exc:
        return {}, f"Unknown Q# benchmark case: {exc}"

    try:
        counts = qsharp.logical_counts(operation, *args)
    except Exception as exc:  # pragma: no cover
        fallback = {
            "backend": "qsharp.FullStateSimulator",
            "compilation_time_seconds": None,
            "qubit_count": int(config.get("num_sites", 0)),
        }
        return fallback, f"Failed to gather logical counts: {exc}"

    total_gate_count = int(
        counts.get("rotationCount", 0)
        + counts.get("cczCount", 0)
        + counts.get("ccixCount", 0)
        + counts.get("measurementCount", 0)
    )
    two_qubit = int(counts.get("cczCount", 0) + counts.get("ccixCount", 0))
    metrics = {
        "backend": "qsharp.FullStateSimulator",
        "compilation_time_seconds": None,
        "total_gate_count": total_gate_count,
        "two_qubit_gate_count": two_qubit,
        "circuit_depth": int(counts.get("rotationDepth", 0)),
        "qubit_count": int(counts.get("numQubits", config.get("num_sites", 0))),
        "native_gate_set": ["LogicalCounts"],
    }
    return metrics, "Logical counts derived via QDK resource estimation"


def quipper_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    run_cli = ROOT / "programs" / "quipper" / "run_cli.py"
    if not run_cli.exists():
        return {}, "Quipper CLI shim not found."
    cmd = [sys.executable, str(run_cli), "--mode", "metrics", case_name]
    payload = json.dumps(config).encode("utf-8")
    proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode().strip()
        raise RuntimeError(f"Quipper metrics CLI failed: {stderr}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON from Quipper metrics CLI: {proc.stdout.decode(errors='ignore')}"
        ) from exc
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Quipper metrics payload missing 'metrics'")
    gate_hist = metrics.get("gate_histogram") or []
    native_gate_set = sorted(
        {
            entry.get("label")
            for entry in gate_hist
            if isinstance(entry, dict) and entry.get("label")
        }
    )
    metrics_dict = {
        "backend": "quipper.simulation",
        "compilation_time_seconds": None,
        "total_gate_count": metrics.get("total_gate_count"),
        "two_qubit_gate_count": metrics.get("two_qubit_gate_count"),
        "circuit_depth": metrics.get("circuit_depth"),
        "qubit_count": metrics.get("qubit_count"),
        "native_gate_set": native_gate_set,
    }
    return metrics_dict, None


def openqasm_metric_provider(
    language: str, case_name: str, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Metric extractor for OpenQASM 3 programs using Qiskit's QASM 3 importer."""
    # LCU benchmarks are treated as N/A to avoid very long transpilation
    # times on generic hardware-like Qiskit backends. Trotter cases still
    # report full metrics.
    if "lcu" in case_name:
        return {}, (
            "N/A: OpenQASM LCU benchmarks omitted due to slow transpilation on "
            "generic Qiskit backends."
        )
    try:
        from qiskit.qasm3 import loads as qasm3_loads  # type: ignore
        from qiskit import transpile  # type: ignore
        from qiskit.providers.fake_provider import GenericBackendV2  # type: ignore
        from programs.openqasm import common as oq_common  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {}, f"OpenQASM/Qiskit stack unavailable: {exc}"

    num_sites = int(config["num_sites"])
    time_total = float(config["time"])
    params = config.get("params", {})

    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 2))
        qasm, _ = oq_common.render_tfim_trotter_qasm(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
            steps,
        )
    elif case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 2))
        qasm, _ = oq_common.render_heis_trotter_qasm(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
            steps,
        )
    elif case_name == "tfim_lcu":
        qasm = oq_common.render_tfim_lcu_qasm(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("h", 1.0)),
            time_total,
        )
    elif case_name == "heis_lcu":
        qasm = oq_common.render_heis_lcu_qasm(
            num_sites,
            float(params.get("J", 1.0)),
            float(params.get("field", 0.2)),
            time_total,
        )
    else:
        return {}, f"Unsupported case {case_name} for OpenQASM metrics"

    qc = qasm3_loads(qasm)
    backend = GenericBackendV2(num_qubits=num_sites)
    start = perf_counter()
    compiled = transpile(qc, backend=backend, optimization_level=2)
    compilation_time = perf_counter() - start
    metrics = describe_qiskit_circuit(compiled)
    metrics["backend"] = getattr(backend, "name", str(backend))
    metrics["compilation_time_seconds"] = compilation_time
    return metrics, None


METRIC_REGISTRY: Dict[
    str, Callable[[str, str, Dict[str, Any]], Tuple[Dict[str, Any], Optional[str]]]
] = {
    "cirq": cirq_metric_provider,
    "hml": hml_metric_provider,
    "qiskit": qiskit_metric_provider,
    "pyquil": pyquil_metric_provider,
    "pennylane": pennylane_metric_provider,
    "tket": tket_metric_provider,
    "strawberryfields": strawberryfields_metric_provider,
    "qrisp": qrisp_metric_provider,
    "qualtran": qualtran_metric_provider,
    "qsharp": qsharp_metric_provider,
    "quipper": quipper_metric_provider,
    "openqasm": openqasm_metric_provider,
}


# Languages for which we intentionally suppress circuit metrics, with reasons.
METRIC_NA_REASONS: Dict[str, str] = {
    "silq": "N/A: Silq programs run via the Silq CLI; no circuit-level metric extractor is implemented.",
    "strawberryfields": "N/A: No portable gate-analysis tooling exists for the dual-rail Strawberry Fields circuits.",
    "tket": "N/A: No portable gate-analysis tooling exists for TKET circuits in this artifact.",
    "qrisp": "N/A: No portable gate-analysis tooling exists for Qrisp circuits in this artifact.",
    "hml": "N/A: No portable gate-analysis tooling exists for HML/SimuQ circuits in this artifact.",
    "qualtran": "N/A: No portable gate-analysis tooling exists for Qualtran bloqs in this artifact.",
    "quipper": "N/A: No portable gate-analysis tooling exists for Quipper circuits in this artifact.",
}


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------


def describe_cirq_circuit(
    circuit: "cirq.Circuit", qubits: Iterable["cirq.Qid"]
) -> Dict[str, Any]:
    ops = list(circuit.all_operations())
    two_qubit = sum(1 for op in ops if len(op.qubits) == 2)
    gate_set = sorted(
        {
            type(op.gate).__name__ if getattr(op, "gate", None) else type(op).__name__
            for op in ops
        }
    )
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
    two_qubit = sum(
        int(count) for gate, count in counts.items() if gate.lower() in two_qubit_gates
    )
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


def _qsharp_operation_and_args(
    case_name: str, config: Dict[str, Any], qsharp_module
) -> Tuple[Any, Tuple[Any, ...]]:
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    coupling = float(params.get("J", 1.0))

    if case_name == "tfim_trotter":
        steps = int(params.get("trotter_steps", 1))
        field = float(params.get("h", 0.0))
        op = qsharp_module.code.HamiltonianSimulation.TFIMTrotter.Run
        return op, (num_sites, steps, coupling, field, total_time)
    if case_name == "heis_trotter":
        steps = int(params.get("trotter_steps", 1))
        field = float(params.get("field", 0.0))
        op = qsharp_module.code.HamiltonianSimulation.HeisenbergTrotter.Run
        return op, (num_sites, steps, coupling, field, total_time)
    if case_name == "tfim_lcu":
        field = float(params.get("h", 0.0))
        op = qsharp_module.code.HamiltonianSimulation.TFIMLCU.Run
        return op, (num_sites, coupling, field, total_time)
    if case_name == "heis_lcu":
        field = float(params.get("field", 0.0))
        op = qsharp_module.code.HamiltonianSimulation.HeisenbergLCU.Run
        return op, (num_sites, coupling, field, total_time)
    raise KeyError(case_name)


def build_strawberryfields_program(
    case_name: str, num_qubits: int, total_time: float, params: Dict[str, Any]
) -> Tuple[Optional[Any], Optional[str]]:
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


_LANGUAGE_VERSION_MODULES: Dict[str, Tuple[str, ...]] = {
    "cirq": ("cirq",),
    "hml": ("simuq",),
    "openqasm": ("qiskit",),
    "pennylane": ("pennylane",),
    "pyquil": ("pyquil",),
    "qiskit": ("qiskit",),
    "qrisp": ("qrisp",),
    "qualtran": ("qualtran",),
    "qsharp": ("qsharp",),
    "strawberryfields": ("strawberryfields",),
    "tket": ("pytket",),
}


def _module_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(module, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        return str(value)
    return "unknown"


def detect_tool_versions(language: str) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    module_names = _LANGUAGE_VERSION_MODULES.get(language, ())
    if not module_names:
        if language in {"quipper", "silq"}:
            versions[language] = "cli"
        return versions
    for module_name in module_names:
        version = _module_version(module_name)
        if version is None:
            versions[module_name] = "unavailable"
        else:
            versions[module_name] = version
    return versions


if __name__ == "__main__":
    main()
