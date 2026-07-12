import numpy as np
N = 24

def entrypoint():
    indices = np.arange(N, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / N
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))

    return np.column_stack(
        (
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            z,
        )
    )
