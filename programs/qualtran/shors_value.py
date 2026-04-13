from __future__ import annotations
from typing import Any, Dict
import numpy as np
from .shors import find_factor



def run_simulation(config: Dict[str, Any]):
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))

    if N != 21 or a != 2:
        raise ValueError(
            f"This implementation is specialized for N=21, a=2 (got N={N}, a={a})."
        )

    print(f"\t**Building Qualtran QPE Bloq for N={N}, a={a}, t={t}...")
    qpe_bloq = _build_qpe_circuit(t=t, N=N, a=a)
    print("\t**Bloq built; simulating...")

    init_quregs = get_named_qubits(qpe_bloq.signature)
    qm = InteropQubitManager(cirq.ops.SimpleQubitManager())
    circuit, quregs_out = qpe_bloq.flatten().to_cirq_circuit_and_quregs(qubit_manager=qm, **init_quregs)
    sig_qubits = merge_qubits(qpe_bloq.signature, **quregs_out)
    extra_qubits = sorted(set(circuit.all_qubits()) - set(sig_qubits), key=str)
    qubit_order = list(sig_qubits) + extra_qubits
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    full_sv = np.asarray(result.final_state_vector, dtype=np.complex128)

    sig_bits = sum(reg.total_bits() for reg in qpe_bloq.signature)
    anc_bits = len(extra_qubits)
    sv2 = full_sv.reshape((1 << sig_bits, 1 << anc_bits)) if anc_bits else full_sv.reshape((1 << sig_bits, 1))
    projected = sv2[:, 0]
    projected = projected / np.linalg.norm(projected)

    # Marginalize the x register; exponent register is the first t qubits
    m = int(ceil(log2(N)))
    sv_matrix = projected.reshape(1 << t, 1 << m)
    probs = np.sum(np.abs(sv_matrix)**2, axis=1)
    probs = probs / probs.sum()

    # Sample 2048 shots from the exponent (counting) register
    rng = np.random.default_rng(int(config.get("seed", 0)))
    samples = rng.choice(1 << t, size=2048, p=probs)

    from collections import Counter
    counts = Counter(samples.tolist())

    for meas_int, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if meas_int == 0:
            continue
        phase = Fraction(meas_int, 1 << t).limit_denominator(N)
        r = phase.denominator
        if r > 0 and pow(a, r, N) == 1 and r % 2 == 0:
            ar2 = pow(a, r // 2, N)
            d = gcd(ar2 - 1, N)
            if 1 < d < N:
                return np.array(int(d))
            d = gcd(ar2 + 1, N)
            if 1 < d < N:
                return np.array(int(d))

    raise RuntimeError("Shor failed to find a factor with the given configuration.")



# def run_simulation(config: Dict[str, Any]) -> int:
#     n = int(config.get("N", 21))
#     if n < 2:
#         raise ValueError("N must be >= 2")

#     max_attempts = int(config.get("max_attempts", 1)) # 1 attempt across languages for now
#     retries = int(config.get("retries", 1)) # 1 retry across langaufes

#     for _ in range(retries):
#         factor = find_factor(n=n, max_attempts=max_attempts)
#         if factor is not None:
#             return np.array(factor)

#     raise ValueError(f"No non-trivial factor found for N={n}")
