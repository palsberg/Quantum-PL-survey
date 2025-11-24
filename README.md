# Hamiltonian Simulation Survey

This repository accompanies our ACM CSUR-style survey of Hamiltonian simulation across a dozen quantum programming ecosystems. For each language we implement the Transverse-Field Ising Model (TFIM) and the Heisenberg XXX Hamiltonian with both Lie–Trotterization and Linear Combination of Unitaries (LCU), evaluate qualitative ergonomics, and collect quantitative metrics (gate counts, depth, runtime).

## Repository Layout

- `programs/` – Reference implementations grouped by language (Cirq, Pennylane, PyQuil, Q#, Qiskit, Qrisp, Qualtran, Silq, Strawberry Fields, Tket, etc.).
- `programs/common/` – Shared utilities for Pauli models and Taylor-series helpers used by several LCU backends.
- `harness/` – Cross-language correctness runner that compares each program against NumPy reference evolutions.
- `benchmarks/` – Automated benchmarking scripts plus the latest JSON/CSV output.
- `Notes/` – Design notes, per-language findings, and Hamiltonian derivations.
- `Paper/` – LaTeX source for the survey manuscript and companion tables/figures.

## Getting Started

1. Clone the repo and enter it:
   ```bash
   git clone <repo-url>
   cd Hamiltonian_Simulation
   ```
2. Create and activate a Python 3.11+ virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. Install the language backends you need using the pinned versions listed below, e.g.
   ```bash
   pip install cirq-core==1.4.0 qiskit==2.2.3 qsharp==1.22.0
   ```
   (You can install all packages at once, or only those required for the languages you plan to run.)

## Language Dependencies (pinned to our `.venv`)

| Language(s)                            | Python package(s) & version(s)                          | Notes |
|---------------------------------------|---------------------------------------------------------|-------|
| Cirq, HML, OpenQASM helper circuits   | `cirq-core==1.4.0`, `numpy==1.26.4`                     | OpenQASM rendering replays into Cirq simulators. |
| Pennylane                             | `pennylane==0.43.1`, `pennylane-lightning==0.43.0`, `jax==0.4.28`, `jaxlib==0.4.28` | Lightning backend gives fast state vectors. |
| PyQuil                                | `pyquil==4.17.0`, `rpcq==3.11.0`, `qcs-sdk-python==0.21.21` | Requires Rigetti QCS client libraries even for local sims. |
| Qiskit                                | `qiskit==2.2.3`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.43.1`, `rustworkx==0.17.1` | Aer powers statevector sims and gate metrics. |
| Q# (modern QDK)                       | `qsharp==1.22.0`, `qsharp-widgets==1.22.0`, `qsharp` runtime auto-installs the .NET QDK. | No separate `dotnet build` required; Python driver runs everything. |
| Qrisp                                 | `qrisp==0.6.3`                                          | Provides high-level Hamiltonian constructs used in `programs/qrisp`. |
| Qualtran                              | `qualtran==0.6.1`, `cotengra==0.7.5`, `quimb==1.11.2`   | Used for advanced TFIM decompositions on Cirq backends. |
| Strawberry Fields                     | `StrawberryFields==0.19.0`, `thewalrus==0.21.0`, `quantum-blackbird==0.5.0` | Supports Gaussian CV encodings of the qubit Hamiltonians. |
| Tket                                  | `pytket==2.10.3`, `pytket-qiskit==0.73.0`               | Tket targets circuit synthesis; we export to Qiskit for simulation. |
| Common math / tooling (multi-language)| `scipy==1.16.3`, `sympy==1.13.0`, `pandas==2.3.3`, `matplotlib==3.10.7` | Shared across benchmarking/plots. |
| Non-Python toolchains                 | **Silq** (requires the Silq compiler toolchain, install from https://silq.ethz.ch).<br>**Quipper** (requires GHC + Quipper libraries, see https://www.mathstat.dal.ca/~selinger/quipper/). | Our CLI wrappers assume those toolchains are on `PATH`. |

Install whichever rows correspond to the languages you intend to execute (e.g., `pip install qsharp==1.22.0` before running the Q# programs). The versions above mirror the ones in our `.venv` and are what we cite in the paper.

## Running the Cross-Language Test Harness

The harness executes every available language/case pair, normalizes the resulting statevector, and reports fidelity against a NumPy reference evolution.

```bash
source .venv/bin/activate
python harness/run_tests.py
```

Useful options:

- `--languages LANG1,LANG2` limits execution to specific languages (matching the keys in `languages.py`).
- `--cases tfim_trotter,heis_lcu` restricts which benchmark cases run.

Example (only Q# and Pennylane, TFIM trotter):

```bash
python harness/run_tests.py --languages qsharp,pennylane --cases tfim_trotter
```

The harness automatically sets `DOTNET_CLI_HOME`, `PYTKET_CONFIG_DIR`, etc., so no extra environment preparation is needed beyond installing the packages above.

## Running Benchmarks

The benchmarking script reuses the harness adapters, but additionally collects compilation/runtime METRICS per language and persists them to JSON or CSV (matching `benchmarks/benchmark_results_schema.json`).

```bash
source .venv/bin/activate
python benchmarks/run_benchmarks.py \
    --languages qsharp,qiskit \
    --cases tfim_trotter,heis_trotter \
    --format json \
    --output benchmarks/latest_results.json
```

What you get:

- Human-readable progress plus a summary of successes/failures.
- Structured output (JSON/CSV) with gate counts, depth, qubit counts, backend/compilation timings, and fidelity status for each `(language, case)` combination.
- For Q#, the script now calls the modern QDK resource estimator (`qsharp.logical_counts`) so you get logical gate metrics in addition to simulator timings.
