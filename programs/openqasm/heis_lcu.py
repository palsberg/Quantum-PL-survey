"""Heisenberg XXX via OpenQASM 3 LCU block executed on a Qiskit backend."""

from __future__ import annotations

from typing import Any, Dict

from . import common as oq_common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))

    # Emit OpenQASM 3 for the LCU block-encoding as an artifact.
    qasm = oq_common.render_heis_lcu_qasm(num_sites, J, field, total_time)
    oq_common._write_qasm("heis_lcu", qasm, params)

    # Execute the QASM program using Qiskit's OpenQASM 3 importer and
    # statevector simulator, then project onto the logical system subspace
    # corresponding to selection = |0...0>, phase_anc = |1>, junk = |0...0>.
    from qiskit.qasm3 import loads as qasm3_loads  # type: ignore
    from qiskit.quantum_info import Statevector  # type: ignore
    import numpy as np

    qc = qasm3_loads(qasm)
    full = Statevector.from_instruction(qc).data

    sel_reg = next(qr for qr in qc.qregs if qr.name == "selection")
    junk_reg = next(qr for qr in qc.qregs if qr.name == "junk")
    num_sel = sel_reg.size
    num_junk = junk_reg.size

    dim_sys = 2**num_sites
    dim_sel = 2**num_sel
    dim_phase = 2
    dim_junk = 2**num_junk

    full = full.reshape((dim_sys, dim_sel, dim_phase, dim_junk))
    slice_vec = full[:, 0, 1, 0]
    norm = np.linalg.norm(slice_vec)
    if norm == 0:
        raise RuntimeError("OpenQASM Heisenberg LCU produced zero amplitude on |sel=0, phase=1, junk=0>.")
    return slice_vec / norm


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.2, "params": {"J": 1.0, "field": 0.3}}
    print(run_simulation(cfg))
