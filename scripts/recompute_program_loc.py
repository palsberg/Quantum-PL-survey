#!/usr/bin/env python3
"""
Recompute program-size LOC for all 13×4 benchmark programs.

For each language × {TFIM, Heisenberg} × {Trotter, LCU} we:
  * Start from the benchmark entrypoint (e.g., programs/cirq/tfim_trotter.py).
  * Optionally inline language-local helper definitions (only those whose
    names appear in the entrypoint or other inlined helpers).
  * For non-Python stacks (Q#, Quipper, Silq, OpenQASM) count LOC directly
    in the native source / generated artifact, ignoring Python shims.
  * Strip comments and blank lines before counting.

The resulting counts are written into Paper/program_size_table.tex.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def count_loc(lines: List[str], lang: str) -> int:
    """Count non-empty, non-comment lines for a given language."""
    count = 0
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if lang in {"python", "strawberry"}:
            if stripped.startswith("#"):
                continue
        elif lang in {"qsharp", "openqasm", "silq"}:
            if stripped.startswith("//"):
                continue
        elif lang == "haskell":
            if stripped.startswith("--"):
                continue
        else:
            # Default: no special comment handling beyond blank lines.
            pass
        count += 1
    return count


def build_python_helper_defs(mod_paths: List[Path]) -> Dict[str, List[str]]:
    """Return mapping name -> definition lines for top-level defs/classes."""
    helpers: Dict[str, List[str]] = {}
    for mod_path in mod_paths:
        if not mod_path.exists():
            continue
        lines = read_lines(mod_path)
        current_name: str | None = None
        current_block: List[str] = []
        for line in lines:
            m = re.match(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m:
                # Flush previous block.
                if current_name is not None and current_block:
                    helpers[current_name] = current_block
                current_name = m.group(2)
                current_block = [line]
            else:
                if current_name is not None:
                    current_block.append(line)
        if current_name is not None and current_block:
            helpers[current_name] = current_block
    return helpers


def discover_used_helpers(
    texts: List[str], helper_defs: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Recursively discover helper defs whose names appear in the given texts."""
    used: Dict[str, List[str]] = {}
    queue: List[str] = list(texts)
    name_patterns: Dict[str, re.Pattern[str]] = {
        name: re.compile(r"\b" + re.escape(name) + r"\b") for name in helper_defs
    }
    while queue:
        text = queue.pop()
        for name, body in helper_defs.items():
            if name in used:
                continue
            pat = name_patterns[name]
            if pat.search(text):
                used[name] = body
                queue.append("\n".join(body))
    return used


def python_case_loc(
    entry_path: Path,
    helper_mods: List[Path],
) -> int:
    if not entry_path.exists():
        return 0
    entry_lines = read_lines(entry_path)
    helpers = build_python_helper_defs(helper_mods)
    used_helpers = discover_used_helpers(["\n".join(entry_lines)], helpers)
    all_lines: List[str] = []
    all_lines.extend(entry_lines)
    for body in used_helpers.values():
        all_lines.extend(body)
    # Python and most stacks here share the same comment style.
    return count_loc(all_lines, lang="python")


def build_qsharp_helper_defs(common_path: Path) -> Dict[str, List[str]]:
    """Return mapping name -> definition lines for Q# operations/functions.

    We do a simple scan for lines starting with 'operation' or 'function' and
    treat each such block up to the next declaration as a helper. This is
    sufficient for Common.qs, which contains only top-level declarations
    inside a namespace.
    """
    helpers: Dict[str, List[str]] = {}
    if not common_path.exists():
        return helpers
    lines = read_lines(common_path)
    current_name: str | None = None
    current_block: List[str] = []
    for line in lines:
        m = re.match(r"^\s*(operation|function)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if m:
            if current_name is not None and current_block:
                helpers[current_name] = current_block
            current_name = m.group(2)
            current_block = [line]
        else:
            if current_name is not None:
                current_block.append(line)
    if current_name is not None and current_block:
        helpers[current_name] = current_block
    return helpers


def qsharp_case_loc(entry_path: Path, common_path: Path) -> int:
    """Count LOC in the Q# entrypoint plus only the Common.qs helpers it uses."""
    if not entry_path.exists():
        return 0
    entry_lines = read_lines(entry_path)
    helpers = build_qsharp_helper_defs(common_path)
    used_helpers = discover_used_helpers(["\n".join(entry_lines)], helpers)
    all_lines: List[str] = []
    all_lines.extend(entry_lines)
    for body in used_helpers.values():
        all_lines.extend(body)
    return count_loc(all_lines, lang="qsharp")


def build_haskell_helper_defs(mod_paths: List[Path]) -> Dict[str, List[str]]:
    """Return mapping name -> definition lines for top-level Haskell functions.

    We treat a line of the form 'name :: ...' as the start of a helper block and
    capture everything up to the next such signature. This is sufficient for
    QuipperCommon.hs, which only contains simple top-level helpers.
    """
    helpers: Dict[str, List[str]] = {}
    for mod_path in mod_paths:
        if not mod_path.exists():
            continue
        lines = read_lines(mod_path)
        current_name: str | None = None
        current_block: List[str] = []
        for line in lines:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_']*)\s*::", line)
            if m:
                if current_name is not None and current_block:
                    helpers[current_name] = current_block
                current_name = m.group(1)
                current_block = [line]
            else:
                if current_name is not None:
                    current_block.append(line)
        if current_name is not None and current_block:
            helpers[current_name] = current_block
    return helpers


def haskell_case_loc(entry_path: Path, helper_paths: List[Path]) -> int:
    """Count LOC in the Haskell entrypoint plus only the helper defs it uses."""
    if not entry_path.exists():
        return 0
    entry_lines = read_lines(entry_path)
    helpers = build_haskell_helper_defs(helper_paths)
    used_helpers = discover_used_helpers(["\n".join(entry_lines)], helpers)
    all_lines: List[str] = []
    all_lines.extend(entry_lines)
    for body in used_helpers.values():
        all_lines.extend(body)
    return count_loc(all_lines, lang="haskell")


def silq_case_loc(entry_path: Path) -> int:
    if not entry_path.exists():
        return 0
    return count_loc(read_lines(entry_path), lang="silq")


def openqasm_case_loc(case_name: str) -> int:
    """Count LOC on a synthetic 10-site OpenQASM program for this case.

    We generate QASM via the project-local emitters for a 10-qubit instance
    (matching the TFIM/Heisenberg benchmark configuration) and count
    non-comment, non-blank lines in the resulting IR, without running any
    simulator or backend.
    """
    try:
        from programs.openqasm import common as oq_common  # type: ignore
    except Exception:
        return 0

    num_sites = 10
    if case_name == "tfim_trotter":
        qasm, _ = oq_common.render_tfim_trotter_qasm(
            num_sites=num_sites,
            J=1.0,
            h=0.5,
            total_time=0.1,
            steps=4,
        )
    elif case_name == "heis_trotter":
        qasm, _ = oq_common.render_heis_trotter_qasm(
            num_sites=num_sites,
            J=0.8,
            field=0.2,
            total_time=0.1,
            steps=4,
        )
    elif case_name == "tfim_lcu":
        qasm = oq_common.render_tfim_lcu_qasm(
            num_sites=num_sites,
            J=1.0,
            h=0.5,
            total_time=0.1,
        )
    elif case_name == "heis_lcu":
        qasm = oq_common.render_heis_lcu_qasm(
            num_sites=num_sites,
            J=0.8,
            field=0.2,
            total_time=0.1,
        )
    else:
        return 0
    lines = qasm.splitlines()
    return count_loc(lines, lang="openqasm")


def main() -> None:
    # Languages and their LOC per (TFIM/Heis × Trotter/LCU).
    results: Dict[str, Dict[str, int]] = {}

    # Python-hosted languages with helper modules.
    py_langs = {
        "cirq": {
            "trotter_helpers": [ROOT / "programs" / "cirq" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "cirq" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
        "hml": {
            "trotter_helpers": [ROOT / "programs" / "hml" / "common.py"],
            "lcu_helpers": [ROOT / "programs" / "hml" / "common.py"],
        },
        "pennylane": {
            "trotter_helpers": [ROOT / "programs" / "pennylane" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "pennylane" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
        "pyquil": {
            "trotter_helpers": [ROOT / "programs" / "pyquil" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "pyquil" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
        "qiskit": {
            "trotter_helpers": [ROOT / "programs" / "qiskit" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "qiskit" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
        "qrisp": {
            "trotter_helpers": [ROOT / "programs" / "qrisp" / "common.py"],
            "lcu_helpers": [ROOT / "programs" / "qrisp" / "common.py"],
        },
        "qualtran": {
            "trotter_helpers": [ROOT / "programs" / "qualtran" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "qualtran" / "common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
        "strawberryfields": {
            "trotter_helpers": [ROOT / "programs" / "strawberryfields" / "common.py"],
            "lcu_helpers": [ROOT / "programs" / "strawberryfields" / "common.py"],
        },
        "tket": {
            "trotter_helpers": [ROOT / "programs" / "tket" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "tket" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
        },
    }

    # Cases we care about.
    cases = [
        ("tfim_trotter", "TFIM", "Trotter"),
        ("tfim_lcu", "TFIM", "LCU"),
        ("heis_trotter", "Heis", "Trotter"),
        ("heis_lcu", "Heis", "LCU"),
    ]

    # Python languages first.
    for lang, cfg in py_langs.items():
        lang_res: Dict[str, int] = {}
        for case_name, ham, method in cases:
            entry = ROOT / "programs" / lang / f"{case_name}.py"
            if method == "Trotter":
                helpers = cfg["trotter_helpers"]
            else:
                helpers = cfg["lcu_helpers"]
            loc = python_case_loc(entry, helpers)
            key = f"{ham}_{method}"
            lang_res[key] = loc
        results[lang] = lang_res

    # OpenQASM: count the generated QASM artifacts.
    oq_lang = "openqasm"
    oq_res: Dict[str, int] = {}
    for case_name, ham, method in cases:
        loc = openqasm_case_loc(case_name)
        oq_res[f"{ham}_{method}"] = loc
    results[oq_lang] = oq_res

    # Q#: use Q# source only (no Python shims).
    qsharp_res: Dict[str, int] = {}
    qsharp_dir = ROOT / "programs" / "qsharp"
    common_qs = qsharp_dir / "Common.qs"
    qsharp_entry = {
        "tfim_trotter": qsharp_dir / "TFIMTrotter.qs",
        "tfim_lcu": qsharp_dir / "TFIMLCU.qs",
        "heis_trotter": qsharp_dir / "HeisenbergTrotter.qs",
        "heis_lcu": qsharp_dir / "HeisenbergLCU.qs",
    }
    for case_name, ham, method in cases:
        loc = qsharp_case_loc(qsharp_entry[case_name], common_qs)
        qsharp_res[f"{ham}_{method}"] = loc
    results["qsharp"] = qsharp_res

    # Quipper: Haskell source + shared QuipperCommon.
    quipper_res: Dict[str, int] = {}
    quipper_dir = ROOT / "programs" / "quipper"
    quipper_entry = {
        "tfim_trotter": quipper_dir / "tfim_trotter.hs",
        "tfim_lcu": quipper_dir / "tfim_lcu.hs",
        "heis_trotter": quipper_dir / "heis_trotter.hs",
        "heis_lcu": quipper_dir / "heis_lcu.hs",
    }
    quipper_helpers = [
        quipper_dir / "QuipperCommon.hs",
        quipper_dir / "QuipperLcuCommon.hs",
    ]
    for case_name, ham, method in cases:
        loc = haskell_case_loc(quipper_entry[case_name], quipper_helpers)
        quipper_res[f"{ham}_{method}"] = loc
    results["quipper"] = quipper_res

    # Silq: native Silq sources only.
    silq_res: Dict[str, int] = {}
    silq_dir = ROOT / "programs" / "silq"
    silq_entry = {
        "tfim_trotter": silq_dir / "tfim_trotter.slq",
        "tfim_lcu": silq_dir / "tfim_lcu.slq",
        "heis_trotter": silq_dir / "heis_trotter.slq",
        "heis_lcu": silq_dir / "heis_lcu.slq",
    }
    for case_name, ham, method in cases:
        loc = silq_case_loc(silq_entry[case_name])
        silq_res[f"{ham}_{method}"] = loc
    results["silq"] = silq_res

    # Sanity: ensure we have all 13 languages used in the table.
    ordered_langs = [
        "cirq",
        "hml",
        "openqasm",
        "pennylane",
        "pyquil",
        "qiskit",
        "qrisp",
        "qsharp",
        "qualtran",
        "quipper",
        "silq",
        "strawberryfields",
        "tket",
    ]

    # Pretty-print to LaTeX table.
    out_path = ROOT / "Paper" / "program_size_table_new.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("  \\centering\n")
        f.write(
            "  \\caption{Lines of source code for TFIM and Heisenberg isotropic chain benchmark programs. "
            "Counts are computed from a temporary version of each program where language--local helper "
            "definitions actually used by the benchmark (for example, \\texttt{common.py}, "
            "\\texttt{lcu\\_common.py}, and \\texttt{pauli\\_models.py}) are inlined, comments and blank "
            "lines are removed, and non--Python stacks are measured on their native artifacts (Q\\#, "
            "Quipper, Silq, and generated OpenQASM~3).}\n"
        )
        f.write("  \\label{tab:program-size}\n")
        f.write("  \\begin{tabular}{l|rr|rr}\n")
        f.write("    \\toprule\n")
        f.write(
            "             & \\multicolumn{2}{c|}{TFIM} & \\multicolumn{2}{c}{Heisenberg} \\\\\n"
        )
        f.write("    Language & Trotter & LCU & Trotter & LCU \\\\\n")
        f.write("    \\midrule\n")

        name_map = {
            "cirq": "Cirq",
            "hml": "HML",
            "openqasm": "OpenQASM~3",
            "pennylane": "PennyLane",
            "pyquil": "PyQuil",
            "qiskit": "Qiskit",
            "qrisp": "Qrisp",
            "qsharp": "Q\\#",
            "qualtran": "Qualtran",
            "quipper": "Quipper",
            "silq": "Silq",
            "strawberryfields": "Strawberry Fields",
            "tket": "pytket",
        }

        # Languages whose LCU entries are pseudocode / delegated.
        dagger_langs = {"hml", "qrisp", "silq", "strawberryfields"}

        for lang in ordered_langs:
            row = results.get(lang, {})
            tfim_t = row.get("TFIM_Trotter", 0)
            tfim_l = row.get("TFIM_LCU", 0)
            heis_t = row.get("Heis_Trotter", 0)
            heis_l = row.get("Heis_LCU", 0)

            def fmt_lcu(val: int, is_dagger: bool) -> str:
                if is_dagger:
                    return f"{val}$^{{\\dagger}}$"
                return str(val)

            is_dagger = lang in dagger_langs
            f.write(
                f"    {name_map[lang]:<16} & {tfim_t} & {fmt_lcu(tfim_l, is_dagger)} "
                f"& {heis_t} & {fmt_lcu(heis_l, is_dagger)} \\\\\n"
            )

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table*}\n")
        f.write("%\n")
        f.write(
            "% Entries marked with $^{\\dagger}$ denote LCU programs that are pseudocode\n"
            "% sketches or rely on delegated LCU behavior (HML/SimuQ, Qrisp, Silq, Strawberry Fields),\n"
            "% rather than full Taylor LCU implementations in the given language.\n"
        )

    print(f"Wrote updated LOC table to {out_path}")


if __name__ == "__main__":
    main()
