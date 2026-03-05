# Hamiltonian Simulation Survey

This repository accompanies our ACM CSUR-style survey of Hamiltonian simulation across a dozen quantum programming ecosystems. For each language we implement the Transverse-Field Ising Model (TFIM) and the Heisenberg XXX Hamiltonian with both Lie–Trotterization and Linear Combination of Unitaries (LCU), evaluate qualitative ergonomics, and collect quantitative metrics (gate counts, depth, runtime).

## Repository Layout

- `programs/` – Reference implementations grouped by language (Cirq, Pennylane, PyQuil, Q#, Qiskit, Qrisp, Qualtran, Silq, Strawberry Fields, Tket, etc.).
- `programs/common/` – Shared utilities for Pauli models and Taylor-series helpers used by several LCU backends.
- `harness/` – Cross-language correctness runner that compares each program against NumPy reference evolutions.
- `benchmarks/` – Automated benchmarking scripts plus the latest JSON/CSV output.

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
| Cirq, HML, OpenQASM helper circuits   | `cirq-core==1.4.0`, `numpy==2.3`, `openfermion==1.7.1` | Used for Cirq Trotter/LCU helpers and some HML backends. |
| Guppy                                 | `guppylang==0.21.8`, `tket==0.12.16`                    | Programs are executed using Selene, which is included in guppylang. |
| OpenQASM 3                            | `qiskit==2.2.3`                                         | We use `qiskit.qasm3.loads` to import OpenQASM 3 programs into Qiskit for simulation and metrics. |
| Pennylane                             | `pennylane==0.43.1`, `pennylane-lightning==0.43.0`, `jax==0.4.28`, `jaxlib==0.4.28` | Lightning backend gives fast state vectors. |
| PyQuil                                | `pyquil==4.17.0`, `rpcq==3.11.0`, `qcs-sdk-python==0.21.21` | Requires Rigetti QCS client libraries even for local sims. |
| Qiskit                                | `qiskit==2.2.3`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.43.1`, `rustworkx==0.17.1` | Aer powers statevector sims and gate metrics; also provides the OpenQASM 3 importer. |
| Q# (modern QDK)                       | `qsharp==1.22.0`, `qsharp-widgets==1.22.0`, `qsharp` runtime auto-installs the .NET QDK. | No separate `dotnet build` required; Python driver runs everything. |
| Qrisp                                 | `qrisp==0.7.0`                                          | Provides high-level Hamiltonian constructs used in `programs/qrisp`. |
| Qualtran                              | `qualtran==0.6.1`, `cotengra==0.7.5`, `quimb==1.11.2`   | Used for advanced TFIM decompositions on Cirq backends. |
| Strawberry Fields                     | `StrawberryFields==0.19.0`, `thewalrus==0.21.0`, `quantum-blackbird==0.5.0` | Supports Gaussian CV encodings of the qubit Hamiltonians. |
| Tket                                  | `pytket==2.10.3`, `pytket-qiskit==0.73.0`               | Tket targets circuit synthesis; we export to Qiskit for simulation. |
| Common math / tooling (multi-language)| `scipy==1.16.3`, `sympy==1.13.0`, `pandas==2.3.3`, `matplotlib==3.10.7` | Shared across benchmarking/plots. |
| Non-Python toolchains                 | **Silq** (requires the Silq compiler toolchain, install from https://silq.ethz.ch).<br>**Quipper** (requires GHC + Quipper libraries, see https://www.mathstat.dal.ca/~selinger/quipper/). | Our CLI wrappers assume those toolchains are on `PATH`. |

Install whichever rows correspond to the languages you intend to execute (e.g., `pip install qsharp==1.22.0` before running the Q# programs). The versions above mirror the ones in our `.venv` and are what we cite in the paper.

### Running CUDA-Q in Docker

To execute our CUDA-Q programs, we use the CUDA-Q Docker containers, which allow
us to run on any platform (with or without an Nvidia gpu).

1. Start Docker. If you're on macOS, this probably means starting Docker
   Desktop.
2. Run the provided script: `./cudaq.sh`. This script automatically downloads
   the CUDA-Q Docker image and starts a container in your current shell.
3. From this shell, run `cd Hamiltonian-Simulation`. This takes you to a mounted
   version of the repo from inside the container. Now you can execute the CUDA-Q
   programs normally with our test harness.
4. To exit the container type `exit` or `Ctrl+D`. To restart the container,
   rerun `./cudaq.sh`.

### Quipper on Apple Silicon (and how to adapt the CLI elsewhere)

Quipper is the only stack in this repo that depends on our machine-specific setup (Apple Silicon running GHC 8.6.5 under Rosetta). The key points for the harness are:

1. We launch Quipper via Rosetta: `arch -x86_64 zsh --login` and source `~/.zshrc-quipper`, which exports `PATH` (to include `~/.local/bin`), sources `~/.ghcup/env`, and pins `GHC_ENVIRONMENT`.
2. `programs/quipper/run_cli.py` shells out twice: once to compile each `.hs` file and once to run it, prepending that Rosetta command each time. The relevant snippets are the `command = ("source ~/.zshrc-quipper && ...")` strings inside `compile_case(...)` and `invoke_quipper(...)`.

If you are running on native x86 Linux/Windows, or if your ghcup install lives somewhere else, edit those two command builders:

- Replace `["arch","-x86_64","zsh","--login","-c", "source ~/.zshrc-quipper && ..."]` with whatever launches a shell where `quipper` and its libraries are available (e.g., `["/bin/bash","-lc", "source /opt/quipper/env.sh && ..."]`).
- Adjust the include/output flags if you store `QuipperCommon.hs` or the build artifacts in a different directory (`-i`, `-odir`, `-hidir` arguments in `compile_case`).

Once you can run `python programs/quipper/run_cli.py tfim_trotter <<< '{}'` successfully, the harness/benchmarks will work on your system as well.

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
