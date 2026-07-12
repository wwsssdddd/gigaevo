import numpy as np


N_POINTS = 24
SPHERE_TOL = 1e-7


def _as_real_array(output):
    try:
        raw = np.asarray(output)
    except Exception as exc:
        raise ValueError("output must be convertible to a numeric array") from exc

    if raw.ndim != 2 or raw.shape != (N_POINTS, 3):
        raise ValueError(
            f"expected shape ({N_POINTS}, 3), got {raw.shape}"
        )

    if np.iscomplexobj(raw):
        raise ValueError("coordinates must be real, not complex")

    try:
        points = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("all coordinates must be real numeric values") from exc

    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates contain NaN or infinity")

    return points


def _row_norms(points):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.hypot(np.hypot(points[:, 0], points[:, 1]), points[:, 2])


def validate(output):
    points = _as_real_array(output)

    norms = _row_norms(points)
    radial_errors = np.abs(norms - 1.0)
    if not np.all(np.isfinite(norms)) or np.any(radial_errors > SPHERE_TOL):
        largest_error = float(np.max(radial_errors))
        raise ValueError(
            "every point must lie on the unit sphere within "
            f"{SPHERE_TOL:g}; largest norm error is {largest_error:.3e}"
        )

    normalized = points / norms[:, None]

    first, second = np.triu_indices(N_POINTS, k=1)
    differences = normalized[first] - normalized[second]
    squared_distances = np.einsum(
        "ij,ij->i", differences, differences, optimize=True
    )

    min_squared_distance = float(np.min(squared_distances))
    min_distance = float(np.sqrt(min_squared_distance))

    return {"fitness": min_distance, "is_valid": 1}
