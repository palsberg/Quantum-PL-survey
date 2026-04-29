#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# note the default directories below!
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Paper" / "benchmark_table.tex"

COLUMNS = [
    (("Shors",), "Shor's alg."),
    (("TFIM", "Trotter"), "Trotter"),
    (("TFIM", "LCU"), "LCU"),
    (("HeisenbergXXX", "Trotter"), "Trotter"),
    (("HeisenbergXXX", "LCU"), "LCU"),
]

LANG_GROUP_1 = [
    "cirq", "cudaq", "guppy", "pennylane", "pyquil",
    "qsharp", "qiskit", "qualtran",
]
LANG_GROUP_2 = ["qrisp", "silq"]

DISPLAY_NAMES = {
    "cirq": "Cirq", "cudaq": "CUDA-Q", "guppy": "Guppy",
    "pennylane": "PennyLane", "pyquil": "PyQuil", "qsharp": r"Q\#",
    "qiskit": "Qiskit", "qualtran": "Qualtran", "qrisp": "Qrisp",
    "silq": "Silq",
}


def build_table(results):
    table = {}
    for id in results:
        r = results[id]
        if not r.get("success"):
            continue
        lang = r["language"]
        case = r["case"]
        match case:
            case "tfim_trotter": key = ("TFIM", "Trotter")
            case "tfim_lcu":     key = ("TFIM", "LCU")
            case "heis_trotter": key = ("HeisenbergXXX", "Trotter")
            case "heis_lcu":     key = ("HeisenbergXXX", "LCU")
            case "shors21_2":    key = ("Shors",)
            case _: raise ValueError('Invalid case:', case)
        table.setdefault(lang, {})[key] = r.get("time_mean")
    return table

def fmt(val):
    return f"{val*1000:.2f}" if val is not None else "??"

def lang_row(lang, table):
    col_keys = [k for k, _ in COLUMNS]
    display = DISPLAY_NAMES.get(lang, lang)
    cells = [fmt(table.get(lang, {}).get(k)) for k in col_keys]
    return f"    {display:<16s} & " + " & ".join(f"{c:>8s}" for c in cells) + r" \\"

def generate_latex(table):
    all_langs = LANG_GROUP_1 + LANG_GROUP_2
    # append any extra languages from results
    for l in sorted(table):
        if l not in all_langs:
            all_langs.append(l)

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Execution times in milliseconds.}",
        r"  \begin{tabular}{c|r|rr|rr}",
        r"    \toprule",
        r"             & \multicolumn{1}{c|}{Factoring}",
        r"             & \multicolumn{2}{c|}{TFIM} & \multicolumn{2}{c}{Heisenberg} \\",
        r"    Language & Shor's alg. & Trotter & LCU & Trotter & LCU \\",
        r"    \midrule",
    ]

    for lang in LANG_GROUP_1:
        lines.append(lang_row(lang, table))

    lines.append(r"    \hline")

    group2 = [l for l in all_langs if l not in LANG_GROUP_1]
    for lang in group2:
        lines.append(lang_row(lang, table))

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \label{tab:experimental-results}",
        r"\end{table*}",
        "%",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark JSON to LaTeX table.")
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = json.loads(args.input.read_text())
    table = build_table(results)
    if not table:
        print("No results with status 'ok' found.", file=sys.stderr)
        sys.exit(1)

    latex = generate_latex(table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
