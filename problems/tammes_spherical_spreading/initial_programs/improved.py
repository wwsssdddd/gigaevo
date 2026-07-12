import numpy as np


N = 24
EXP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
             4096, 8192, 16384)
ITER = 4000
MAX_BACKTRACKS = 50
INITIAL_STEP = 0.08
STEP_GROWTH = 1.05
MAX_STEP = 0.15
STEP_SHRINK = 0.5
TOLERANCE = 5e-16


def _fibonacci_sphere():
    indices = np.arange(N, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / N
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )


def _pairwise_squared_distances(points):
    differences = points[:, None, :] - points[None, :, :]
    return np.einsum(
        "ijk,ijk->ij", differences, differences, optimize=True
    )


def _scaled_log_riesz_energy(points, exponent):
    squared_distances = _pairwise_squared_distances(points)
    upper = squared_distances[np.triu_indices(N, k=1)]
    log_terms = -0.5 * exponent * np.log(upper)
    largest = float(np.max(log_terms))
    return (largest + np.log(np.exp(log_terms - largest).sum())) / exponent


def _descent_direction(points, exponent):
    differences = points[:, None, :] - points[None, :, :]
    squared_distances = np.einsum(
        "ijk,ijk->ij", differences, differences, optimize=True
    )
    np.fill_diagonal(squared_distances, np.inf)

    logits = -0.5 * exponent * np.log(squared_distances)
    largest = float(np.max(logits[np.isfinite(logits)]))
    weights = np.exp(logits - largest)
    np.fill_diagonal(weights, 0.0)

    direction = np.sum(
        (weights / squared_distances)[:, :, None] * differences,
        axis=1,
    )
    direction -= (
        np.einsum("ij,ij->i", direction, points)[:, None] * points
    )

    rms_norm = float(np.sqrt(np.mean(np.einsum("ij,ij->i", direction, direction))))
    if rms_norm == 0.0:
        return np.zeros_like(points)
    return direction / rms_norm


def entrypoint():
    points = _fibonacci_sphere()

    for exponent in EXP:
        step = INITIAL_STEP
        value = _scaled_log_riesz_energy(points, exponent)

        for _ in range(ITER):
            direction = _descent_direction(points, exponent)
            local_step = step
            accepted = False

            for _ in range(MAX_BACKTRACKS):
                candidate = points + local_step * direction
                candidate /= np.linalg.norm(candidate, axis=1)[:, None]
                candidate_value = _scaled_log_riesz_energy(candidate, exponent)

                if candidate_value < value - TOLERANCE:
                    points = candidate
                    value = candidate_value
                    step = min(STEP_GROWTH * local_step, MAX_STEP)
                    accepted = True
                    break

                local_step *= STEP_SHRINK

            if not accepted:
                break

    return points
