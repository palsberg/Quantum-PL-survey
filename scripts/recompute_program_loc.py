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
import io
import tokenize
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _strip_c_style_blocks(text: str) -> str:
    """Strip /* ... */ blocks used by C-style languages."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _strip_haskell_blocks(text: str) -> str:
    """Strip {- ... -} Haskell block comments."""
    return re.sub(r"\{\-.*?\-\}", "", text, flags=re.DOTALL)


def _strip_python_comments_and_docstrings(text: str) -> str:
    """Remove Python comments and docstrings, preserving code tokens only."""
    out_parts: List[str] = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    except Exception:
        # Fallback: keep original text if tokenization fails.
        return text

    for tok in tokens:
        tok_type = tok.type
        tok_str = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end

        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out_parts.append(" " * (start_col - last_col))

        if tok_type == tokenize.COMMENT:
            # Drop comments.
            pass
        elif tok_type == tokenize.STRING and prev_toktype in {
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.DEDENT,
            tokenize.NL,
            tokenize.ENDMARKER,
        }:
            # Drop likely module/class/function docstrings.
            pass
        else:
            out_parts.append(tok_str)

        prev_toktype = tok_type
        last_col = end_col
        last_lineno = end_line

    return "".join(out_parts)


def _count_python_logical_lines(lines: List[str]) -> int:
    """Count Python logical lines after stripping comments/docstrings."""
    text = "\n".join(lines)
    stripped = _strip_python_comments_and_docstrings(text)
    count = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(stripped).readline):
            if tok.type == tokenize.NEWLINE:
                count += 1
    except Exception:
        # Fallback to physical non-empty lines.
        for raw in stripped.splitlines():
            if raw.strip():
                count += 1
    return count


def count_loc(lines: List[str], lang: str) -> int:
    """Count non-empty, non-comment logical lines for a given language."""
    if lang == "python":
        return _count_python_logical_lines(lines)

    text = "\n".join(lines)
    if lang in {"qsharp", "silq", "cudaq", "openqasm", "guppy"}:
        text = _strip_c_style_blocks(text)
    elif lang == "haskell":
        text = _strip_haskell_blocks(text)

    count = 0
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if lang in {"qsharp", "openqasm", "silq", "cudaq", "guppy"}:
            if stripped.startswith("//"):
                continue
        elif lang == "haskell":
            if stripped.startswith("--"):
                continue
        # For non-Python languages, optionally treat obvious continuation-only
        # lines as formatting rather than logic.
        if stripped in {"(", ")", "[", "]", "{", "}"}:
            continue
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


def silq_case_loc(silq_paths: List[Path]) -> int:
    total_loc = 0
    for path in silq_paths:
        if not path.exists():
            continue
        total_loc += count_loc(read_lines(path), lang="silq")
    return total_loc


def python_shors_case_loc(entry_path: Path, helper_mods: List[Path]) -> int:
    """Count Shor entry plus helper module files for fairness across languages."""
    if not entry_path.exists():
        return 0
    all_lines = read_lines(entry_path)
    for helper in helper_mods:
        if helper.exists():
            all_lines.extend(read_lines(helper))
    return count_loc(all_lines, lang="python")



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
            "shors_helpers": [
                ROOT / "programs" / "cirq" / "shor" / "modularexponentiation.py",
                ROOT / "programs" / "cirq" / "shor" / "quantumorderfinding.py",
            ],
        },
        "cudaq": {
            "trotter_helpers": [ROOT / "programs" / "cudaq" / "common.py"],
            "lcu_helpers": [ROOT / "programs" / "cudaq" / "lcu_common.py"],
            "shors_helpers": [],
        },
        "guppy": {
            "trotter_helpers": [ROOT / "programs" / "guppy" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "guppy" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
            "shors_helpers": [],
        },
        "pennylane": {
            "trotter_helpers": [ROOT / "programs" / "pennylane" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "pennylane" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
            "shors_helpers": [],
        },
        "pyquil": {
            "trotter_helpers": [ROOT / "programs" / "pyquil" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "pyquil" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
            "shors_helpers": [],
        },
        "qiskit": {
            "trotter_helpers": [ROOT / "programs" / "qiskit" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "qiskit" / "lcu_common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
            "shors_helpers": [],
        },
        "qrisp": {
            "trotter_helpers": [ROOT / "programs" / "qrisp" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "common" / "pauli_models.py",
                ROOT / "programs" / "qrisp" / "common.py",
                ROOT / "programs" / "qrisp" / "lcu_common.py",
            ],
            "shors_helpers": [],
        },
        "qualtran": {
            "trotter_helpers": [ROOT / "programs" / "qualtran" / "common.py"],
            "lcu_helpers": [
                ROOT / "programs" / "qualtran" / "common.py",
                ROOT / "programs" / "common" / "pauli_models.py",
            ],
            "shors_helpers": [
                ROOT / "programs" / "qualtran" / "shors.py",
            ],
        },
    }

    # Cases we care about.
    cases = [
        ("tfim_trotter", "TFIM", "Trotter"),
        ("tfim_lcu", "TFIM", "LCU"),
        ("heis_trotter", "Heis", "Trotter"),
        ("heis_lcu", "Heis", "LCU"),
        ("shors_value", "Factoring", "Shors")
    ]

    # Python languages first.
    for lang, cfg in py_langs.items():
        lang_res: Dict[str, int] = {}
        for case_name, ham, method in cases:
            entry = ROOT / "programs" / lang / f"{case_name}.py"
            if method == "Trotter":
                helpers = cfg["trotter_helpers"]
            elif method == "LCU":
                helpers = cfg["lcu_helpers"]
            elif method == "Shors":
                helpers = cfg["shors_helpers"]
            else:
                assert False
            if method == "Shors":
                loc = python_shors_case_loc(entry, helpers)
            else:
                loc = python_case_loc(entry, helpers)
            key = f"{ham}_{method}"
            lang_res[key] = loc
        results[lang] = lang_res

    # Q#: use Q# source only (no Python shims).
    qsharp_res: Dict[str, int] = {}
    qsharp_dir = ROOT / "programs" / "qsharp"
    common_qs = qsharp_dir / "Common.qs"
    qsharp_entry = {
        "tfim_trotter": qsharp_dir / "TFIMTrotter.qs",
        "tfim_lcu": qsharp_dir / "TFIMLCU.qs",
        "heis_trotter": qsharp_dir / "HeisenbergTrotter.qs",
        "heis_lcu": qsharp_dir / "HeisenbergLCU.qs",
        "shors_value": qsharp_dir / "shors.qs",
    }
    for case_name, ham, method in cases:
        if case_name in qsharp_entry:
            loc = qsharp_case_loc(qsharp_entry[case_name], common_qs)
        else:
            loc = 0
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
        "shors_value": quipper_dir / "shors.hs",
    }
    quipper_helpers = [
        quipper_dir / "QuipperCommon.hs",
        quipper_dir / "QuipperLcuCommon.hs",
        quipper_dir / "QuipperSimulationCLI.hs",
    ]
    for case_name, ham, method in cases:
        if case_name in quipper_entry:
            loc = haskell_case_loc(quipper_entry[case_name], quipper_helpers)
        else:
            loc = 0
        quipper_res[f"{ham}_{method}"] = loc
    results["quipper"] = quipper_res

    # Silq: native Silq sources only.
    silq_res: Dict[str, int] = {}
    silq_dir = ROOT / "programs" / "silq"
    silq_entry = {
        "tfim_trotter": [silq_dir / "tfim_trotter.slq"],
        "tfim_lcu": [
            silq_dir / "tfim_lcu.slq",
            silq_dir / "lcu_common.slq",
            silq_dir / "map.slq",
        ],
        "heis_trotter": [silq_dir / "heis_trotter.slq"],
        "heis_lcu": [
            silq_dir / "heis_lcu.slq",
            silq_dir / "lcu_common.slq",
            silq_dir / "map.slq",
        ],
        "shors_value": [silq_dir / "shors_value.slq"],
    }
    for case_name, ham, method in cases:
        if case_name in silq_entry:
            loc = silq_case_loc(silq_entry[case_name])
        else:
            loc = 0
        silq_res[f"{ham}_{method}"] = loc
    results["silq"] = silq_res

    # Sanity: ensure we have all 10 languages used in the table.
    ordered_langs = [
        "cirq",
        "cudaq",
        "guppy",
        "pennylane",
        "pyquil",
        "qsharp",
        "qiskit",
        "qualtran",
        "qrisp",
        "silq",
    ]

    # Pretty-print to LaTeX table.
    out_path = ROOT / "Paper" / "program_size_table_new.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("  \\centering\n")
        f.write(
            "  \\caption{The Ten Languages and the Lines of Source Code for Our Benchmark Programs}\n"
        )
        f.write("  \\label{tab:program-size}\n")
        f.write("  \\begin{tabular}{l|r|rr|rr}\n")
        f.write("    \\toprule\n")
        f.write(
            "             & Factoring & \\multicolumn{2}{c|}{TFIM} & \\multicolumn{2}{c}{Heisenberg} \\\\\n"
        )
        f.write("    Language & Shor's & Trotter & LCU & Trotter & LCU \\\\\n")
        f.write("    \\midrule\n")

        name_map = {
            "cirq": "Cirq",
            "cudaq": "CUDA-Q",
            "guppy": "Guppy",
            "openqasm": "OpenQASM~3",
            "pennylane": "PennyLane",
            "pyquil": "PyQuil",
            "qiskit": "Qiskit",
            "qrisp": "Qrisp",
            "qsharp": "Q\\#",
            "qualtran": "Qualtran",
            "silq": "Silq",
        }

        for lang in ordered_langs:
            row = results.get(lang, {})
            tfim_t = row.get("TFIM_Trotter", 0)
            tfim_l = row.get("TFIM_LCU", 0)
            heis_t = row.get("Heis_Trotter", 0)
            heis_l = row.get("Heis_LCU", 0)
            shors = row.get("Factoring_Shors", 0)

            f.write(
                f"    {name_map[lang]:<16} & {shors} & {tfim_t} & {tfim_l} "
                f"& {heis_t} & {heis_l} \\\\\n"
            )

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table*}\n")
        f.write("%\n")

    print(f"Wrote updated LOC table to {out_path}")


if __name__ == "__main__":
    main()
