# A Survey of Quantum Programming Languages

This repository accompanies our ACM CSUR-style survey of ten different quantum
programming languages. For each language we implement four versions of
Hamiltonian simulation (Transverse-Field Ising Model and Heisenberg XXX
Hamiltonian, using Trotterization and Linear Combination of Unitaries) and one
version of Shor's algorithm, for a total of five programs in each language.
We provide a testing harness that measures the fidelity and execution times
of our programs, as well as scripts to generate the tables in our paper.

## Repository Layout

- `programs/` – Reference implementations grouped by language (Cirq, CUDA-Q,
                Guppy, Pennylane, PyQuil, Q#, Qiskit, Qualtran, Qrisp,
                and Silq). We also provide implementations in Quipper, which we
                decided to not include in the paper.
- `programs/common/` – Shared utilities for Pauli models and Taylor-series
                       helpers used by several LCU backends.
- `harness/` – Cross-language correctness runner that compares each program
               against NumPy reference evolutions and measures execution times.
- `scripts/` – Scripts to generate the LaTeX tables.

## Getting Started

1. Clone the repo and enter it:
   ```bash
   git clone <repo-url>
   cd Hamiltonian-Simulation
   ```
2. Create and activate a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. Manually install the language backends you need using the pinned versions
   listed below, e.g.
   ```bash
   pip install cirq-core==1.4.0 qiskit==2.2.3 qsharp==1.22.0
   ```
   or automatically install all of them at once:
   ```bash
   pip install --no-deps -r requirements.txt
   ```
   Note that because of conflicting numpy requirements (Cirq requires numpy 1.x,
   while Guppy and Qrisp require numpy 2.x), you cannot manually install all
   languages at the same time. If you run into an error when trying to manually
   install two different languages, then instead use the second method to
   install all at once, ignoring dependency conflicts with `--no-deps`.
4. For **CUDA-Q**, **Quipper**, and **Silq**, see below.

## Language Dependencies

| Language(s)                           | Python package(s) & version(s)                          | Notes |
|---------------------------------------|---------------------------------------------------------|-------|
| Cirq                                  | `cirq-core==1.4.0`, `numpy==1.26.4`, `openfermion==1.7.0` | Used for Cirq Trotter/LCU helpers |
| CUDA-Q                                | We use a Docker container with CUDA-Q installed (see below). | Run the container with `./cudaq.sh`. |
| Guppy                                 | `guppylang==0.21.8`, `pytket==2.13.0`, `tket==0.12.16`  | Programs are executed using Selene, which is included in guppylang. |
| Pennylane                             | `pennylane==0.43.1`, `pennylane-lightning==0.43.0`, `jax==0.4.28`, `jaxlib==0.4.28`, `gast==0.7.0` | Lightning backend gives fast state vectors. |
| PyQuil                                | `pyquil==4.17.0`, `rpcq==3.11.0`, `qcs-sdk-python==0.21.21` | Requires Rigetti QCS client libraries even for local sims. |
| Q# (modern QDK)                       | `qsharp==1.22.0`, `qsharp-widgets==1.22.0`.<br>The `qsharp` package auto-installs the .NET QDK. | No separate `dotnet build` required; Python driver runs everything. |
| Qiskit                                | `qiskit==2.2.3`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.43.1`, `rustworkx==0.17.1` | Aer powers statevector sims and gate metrics; also provides the OpenQASM 3 importer. |
| Qualtran                              | `qualtran==0.6.1`, `cotengra==0.7.5`, `quimb==1.11.2`   | Used for advanced TFIM decompositions on Cirq backends. |
| Qrisp                                 | `qrisp==0.7.0`                                          | Provides high-level Hamiltonian constructs used in `programs/qrisp`. |
| Common math / tooling (multi-language)| `scipy==1.17.0`, `numpy==2.3.0`, `sympy==1.13.0`, `pandas==2.3.3`, `matplotlib==3.10.7` | Shared across multiple languages. |
| Non-Python toolchains                 | **Quipper** (requires GHC + Quipper libraries, see https://www.mathstat.dal.ca/~selinger/quipper, also see the section below).<br>**Silq** (requires the Silq compiler toolchain, install from https://github.com/silq-lang/silq, compiling from commit 61f3949). | Our CLI wrappers assume those toolchains are on `PATH`. |

Install whichever rows correspond to the languages you intend to execute (e.g.,
`pip install qsharp==1.22.0` before running the Q# programs). The versions above
mirror the ones in our `.venv` and are what we cite in the paper.

### Running CUDA-Q in Docker

To execute our CUDA-Q programs, we use the CUDA-Q Docker containers, which allow
us to run on any platform (with or without an Nvidia GPU).

1. Start Docker. If you're on macOS, this probably means starting Docker
   Desktop.
2. Run the provided script: `./cudaq.sh`. This script automatically downloads
   the CUDA-Q Docker image and starts a container in your current shell.
3. You should now be in the container's shell, and you should see a prompt like
   `cudaq@cudaq:~/Hamiltonian-Simulation$`. Now you can execute the
   CUDA-Q programs normally with our test harness (see below).
4. To stop the container, type `exit` or `Ctrl+D`. To restart the container,
   rerun `./cudaq.sh`.

### Quipper on Apple Silicon (and how to adapt the CLI elsewhere)

We chose not to discuss Quipper in our paper, but we do have implementations of
the benchmarks in this repository. This section outlines how do install Quipper.

To install Quipper on Apple Silicon, see `programs/quipper/Quipper_setup.md`.

Quipper is the only stack in this repo that depends on our machine-specific
setup (Apple Silicon running GHC 8.6.5 under Rosetta). The key points for the
harness are:

1. We launch Quipper via Rosetta: `arch -x86_64 zsh --login` and source
   `~/.zshrc-quipper`, which exports `PATH` (to include `~/.local/bin`), sources
   `~/.ghcup/env`, and pins `GHC_ENVIRONMENT`.
2. `programs/quipper/run_cli.py` shells out twice: once to compile each `.hs`
   file and once to run it, prepending that Rosetta command each time. The
   relevant snippets are the `command = ("source ~/.zshrc-quipper && ...")`
   strings inside `compile_case(...)` and `invoke_quipper(...)`.

If you are running on native x86 Linux/Windows, or if your ghcup install lives
somewhere else, edit those two command builders:

- Replace `["arch","-x86_64","zsh","--login","-c", "source ~/.zshrc-quipper && ..."]`
  with whatever launches a shell where `quipper` and its libraries are available
  (e.g., `["/bin/bash","-lc", "source /opt/quipper/env.sh && ..."]`).
- Adjust the include/output flags if you store `QuipperCommon.hs` or the build
  artifacts in a different directory (`-i`, `-odir`, `-hidir` arguments in
  `compile_case`).

Once you can run `python programs/quipper/run_cli.py tfim_trotter <<< '{}'`
successfully, the harness/benchmarks will work on your system as well.

## Running the Cross-Language Test Harness

The test harness executes every available language/case pair and calculates the
fidelity compared to a reference implementation.

```bash
source .venv/bin/activate
python harness/run_tests.py
```

Options:
```
-h, --help            show this help message and exit
--languages LANG [LANG ...]
                      Subset of languages to run (default: all).
--cases CASE [CASE ...]
                      Subset of cases to run (default: all).
--list                List available languages and cases, then exit.
--runs N              Number of runs for benchmarking (default: 1).
--json FILE           Output results to a json file. If FILE already exists,
                        the new results will be merged into the existing file.
```

Example (only Q# and Pennylane, TFIM trotter):

```bash
python harness/run_tests.py --languages=qsharp,pennylane --cases=tfim_trotter
```

## Benchmarking Execution Time

The test harness can also be used to benchmark the programs, recording the mean
and standard deviation of execution time across a number of runs. All testing
results can be outputted to a JSON file.

To generate the execution times in the paper, we use a dedicated script
(`scripts/benchmark.sh`), which invokes the test harness and records execution
times. **This script requires `jq` to be installed.** To generate the table run the following from the root directory of this
repository:

```bash
mkdir -p benchmarks
scripts/benchmark.sh benchmarks/most.json cirq guppy pennylane pyquil qsharp qiskit qualtran qrisp silq
./cudaq.sh

# [Inside the container]
scripts/benchmark.sh benchmarks/cudaq.json cudaq
exit

# [Outside the container]
jq -s '.[0] * .[1]' benchmarks/most.json benchmarks/cudaq.json > benchmarks/all.json
python scripts/results_to_latex.py -i benchmarks/all.json -o Paper/benchmarks.tex
```

## Counting Line Numbers

We provide a script (`scripts/recompute_program_loc.py`) to compute the number
of lines in each program and output a LaTeX table.

```bash
python scripts/recompute_program_loc.py
```
