"""Strawberry Fields program adapters and compatibility shims."""

from __future__ import annotations

from typing import Any


def _ensure_simps_alias() -> None:
    """Provide scipy.integrate.simps for SF<0.21 running on SciPy>=1.14."""

    try:
        from scipy import integrate  # type: ignore
    except Exception:
        return

    simpson = getattr(integrate, "simpson", None)
    simps = getattr(integrate, "simps", None)
    if simps is None and callable(simpson):
        setattr(integrate, "simps", simpson)


_ensure_simps_alias()

# Re-export Strawberry Fields modules used by sibling packages.
def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough shim
    import strawberryfields as sf  # noqa: WPS433 - import inside shim

    return getattr(sf, name)


__all__ = ["ops", "Program", "Engine"]
