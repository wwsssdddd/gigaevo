import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate import SPHERE_TOL, validate


def _load_entrypoint(filename):
    path = ROOT / "initial_programs" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.entrypoint


FIBONACCI = _load_entrypoint("baseline.py")
RIESZ = _load_entrypoint("improved.py")
KNOWN_OPTIMAL_DISTANCE = 0.7442063311562076


@pytest.fixture(scope="session")
def fibonacci_points():
    return FIBONACCI()


@pytest.fixture(scope="session")
def riesz_points():
    return RIESZ()


def test_required_initial_program_files_exist():
    initial_programs = ROOT / "initial_programs"
    assert (initial_programs / "baseline.py").is_file()
    assert (initial_programs / "improved.py").is_file()


def test_fibonacci_baseline(fibonacci_points):
    result = validate(fibonacci_points)
    assert set(result) == {"fitness", "is_valid"}
    assert result["is_valid"] == 1
    assert result["fitness"] == pytest.approx(0.6300755023659717, abs=1e-12)


def test_riesz_baseline_is_near_optimal(riesz_points):
    result = validate(riesz_points)
    assert 0.7441 < result["fitness"] <= KNOWN_OPTIMAL_DISTANCE + 1e-12
    relative_gap = (KNOWN_OPTIMAL_DISTANCE - result["fitness"]) / KNOWN_OPTIMAL_DISTANCE
    assert relative_gap < 2e-5


def test_initial_programs_are_deterministic_and_improved_is_better(
    fibonacci_points, riesz_points
):
    assert np.array_equal(FIBONACCI(), fibonacci_points)
    assert np.array_equal(RIESZ(), riesz_points)
    assert validate(riesz_points)["fitness"] > validate(fibonacci_points)["fitness"]


def test_distance_identities(fibonacci_points):
    result = validate(fibonacci_points)
    first, second = np.triu_indices(24, k=1)
    inner_products = np.einsum(
        "ij,ij->i", fibonacci_points[first], fibonacci_points[second]
    )
    max_inner_product = float(np.max(inner_products))
    reconstructed = np.sqrt(2.0 - 2.0 * max_inner_product)
    assert reconstructed == pytest.approx(result["fitness"], abs=1e-14)
    angle = np.arccos(max_inner_product)
    assert 2.0 * np.sin(angle / 2.0) == pytest.approx(
        result["fitness"], abs=1e-14
    )


def test_rotation_and_permutation_invariance(fibonacci_points):
    angle = 0.731
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = fibonacci_points[::-1] @ rotation.T
    assert validate(transformed)["fitness"] == pytest.approx(
        validate(fibonacci_points)["fitness"], abs=2e-15
    )


def test_small_radial_error_is_projected_without_score_bias(fibonacci_points):
    scales = np.linspace(1.0 - 0.5 * SPHERE_TOL, 1.0 + 0.5 * SPHERE_TOL, 24)
    perturbed = fibonacci_points * scales[:, None]
    result = validate(perturbed)
    assert result["fitness"] == pytest.approx(
        validate(fibonacci_points)["fitness"], abs=2e-15
    )


def test_input_is_not_modified(fibonacci_points):
    candidate = fibonacci_points.copy()
    before = candidate.copy()
    validate(candidate)
    assert np.array_equal(candidate, before)


def test_near_duplicate_is_valid_and_scores_near_zero(fibonacci_points):
    candidate = fibonacci_points.copy()
    point = candidate[0]
    tangent = np.cross(point, np.array([1.0, 0.0, 0.0]))
    tangent /= np.linalg.norm(tangent)
    angle = 1e-12
    candidate[1] = np.cos(angle) * point + np.sin(angle) * tangent
    result = validate(candidate)
    assert 0.0 < result["fitness"] < 1e-10


def test_exact_duplicate_is_valid_and_scores_zero(fibonacci_points):
    candidate = fibonacci_points.copy()
    candidate[1] = candidate[0]
    result = validate(candidate)
    assert result["fitness"] == 0.0


def test_all_points_identical_are_valid_and_score_zero():
    candidate = np.tile(np.array([[1.0, 0.0, 0.0]]), (24, 1))
    result = validate(candidate)
    assert result["fitness"] == 0.0


def test_point_outside_sphere_tolerance_is_rejected(fibonacci_points):
    candidate = fibonacci_points.copy()
    candidate[0] *= 1.0 + 2.0 * SPHERE_TOL
    with pytest.raises(ValueError, match="unit sphere"):
        validate(candidate)


@pytest.mark.parametrize(
    "candidate, message",
    [
        (np.zeros((23, 3)), "expected shape"),
        (np.zeros((24, 2)), "expected shape"),
        (np.zeros((24, 3, 1)), "expected shape"),
        (np.zeros(72), "expected shape"),
        (None, "expected shape"),
        (np.full((24, 3), np.nan), "NaN or infinity"),
        (np.full((24, 3), np.inf), "NaN or infinity"),
        (np.ones((24, 3), dtype=np.complex128) * (1.0 + 1.0j), "complex"),
        (np.full((24, 3), "not-a-number", dtype=object), "real numeric"),
    ],
)
def test_malformed_outputs_are_rejected(candidate, message):
    with pytest.raises(ValueError, match=message):
        validate(candidate)


def test_huge_finite_values_are_rejected_cleanly():
    candidate = np.full((24, 3), np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="unit sphere"):
        validate(candidate)
