"""Program package compatibility helpers."""

import numpy as np

if not hasattr(np, "ComplexWarning"):
    try:
        from numpy.exceptions import ComplexWarning as _ComplexWarning

        np.ComplexWarning = _ComplexWarning  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        np.ComplexWarning = RuntimeWarning  # type: ignore[attr-defined]
