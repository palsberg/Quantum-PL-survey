"""Pauli-sum helpers shared by multiple LCU implementations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def tfim_pauli_terms(num_sites: int, J: float, h: float) -> Dict[str, complex]:
    """Return a dict mapping Pauli strings to coefficients for TFIM."""
    coeffs: Dict[str, complex] = {}
    for i in range(num_sites - 1):
        chars = ["I"] * num_sites
        chars[i] = "Z"
        chars[i + 1] = "Z"
        s = "".join(chars)
        coeffs[s] = coeffs.get(s, 0.0) + J
    for i in range(num_sites):
        chars = ["I"] * num_sites
        chars[i] = "X"
        s = "".join(chars)
        coeffs[s] = coeffs.get(s, 0.0) + h
    return coeffs


def heisenberg_pauli_terms(num_sites: int, J: float, field: float) -> Dict[str, complex]:
    """Return a dict mapping Pauli strings to coefficients for Heisenberg XXX."""
    coeffs: Dict[str, complex] = {}
    for i in range(num_sites - 1):
        for axis in ("X", "Y", "Z"):
            chars = ["I"] * num_sites
            chars[i] = axis
            chars[i + 1] = axis
            s = "".join(chars)
            coeffs[s] = coeffs.get(s, 0.0) + J
    for i in range(num_sites):
        chars = ["I"] * num_sites
        chars[i] = "Z"
        s = "".join(chars)
        coeffs[s] = coeffs.get(s, 0.0) + field
    return coeffs


def multiply_single_paulis(a: str, b: str) -> Tuple[complex, str]:
    """Multiply two single-qubit Paulis."""
    if a == "I":
        return 1.0 + 0j, b
    if b == "I":
        return 1.0 + 0j, a
    if a == b:
        return 1.0 + 0j, "I"
    table = {
        ("X", "Y"): (1j, "Z"),
        ("Y", "X"): (-1j, "Z"),
        ("Y", "Z"): (1j, "X"),
        ("Z", "Y"): (-1j, "X"),
        ("Z", "X"): (1j, "Y"),
        ("X", "Z"): (-1j, "Y"),
    }
    return table[(a, b)]


def multiply_pauli_strings(p: str, q: str) -> Tuple[complex, str]:
    """Multiply two Pauli strings of equal length."""
    assert len(p) == len(q)
    phase = 1.0 + 0j
    out = []
    for a, b in zip(p, q):
        ph, r = multiply_single_paulis(a, b)
        phase *= ph
        out.append(r)
    return phase, "".join(out)


def taylor_coefficients(H: Dict[str, complex], t: float) -> Dict[str, complex]:
    """Return coefficients for the 2nd-order Taylor polynomial of e^{-iHt}."""
    paulis = list(H.keys())
    n = len(next(iter(paulis))) if paulis else 0
    identity = "I" * n

    coeff_I = 0.0 + 0j
    H2: Dict[str, complex] = {}

    for P, a in H.items():
        coeff_I += a * a

    for i, P_l in enumerate(paulis):
        a_l = H[P_l]
        for j, P_m in enumerate(paulis):
            if i == j:
                continue
            a_m = H[P_m]
            phase, prod = multiply_pauli_strings(P_l, P_m)
            H2[prod] = H2.get(prod, 0.0 + 0j) + a_l * a_m * phase

    gamma: Dict[str, complex] = {}
    gamma_identity = 1.0 - 0.5 * (t**2) * coeff_I
    if identity in H2:
        gamma_identity -= 0.5 * (t**2) * H2[identity]
    gamma[identity] = gamma_identity

    for P, coeff in H2.items():
        if P == identity:
            continue
        gamma[P] = gamma.get(P, 0.0 + 0j) - 0.5 * (t**2) * coeff

    for P, a in H.items():
        gamma[P] = gamma.get(P, 0.0 + 0j) - 1j * t * a

    return gamma


def lcu_weights_from_gamma(gamma: Dict[str, complex]) -> Tuple[list[float], list[str], list[str]]:
    """Split complex coefficients into nonnegative weights with phase tags."""
    weights: list[float] = []
    paulis: list[str] = []
    phases: list[str] = []

    for P, coeff in gamma.items():
        if abs(coeff) < 1e-12:
            continue
        real = coeff.real
        imag = coeff.imag
        if real > 0:
            weights.append(real)
            paulis.append(P)
            phases.append("1")
        elif real < 0:
            weights.append(-real)
            paulis.append(P)
            phases.append("-1")
        if imag > 0:
            weights.append(imag)
            paulis.append(P)
            phases.append("i")
        elif imag < 0:
            weights.append(-imag)
            paulis.append(P)
            phases.append("-i")

    return weights, paulis, phases
