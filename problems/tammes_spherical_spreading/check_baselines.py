import importlib.util
from pathlib import Path

import numpy as np

from validate import validate


PROBLEM_DIR = Path(__file__).resolve().parent
INITIAL_PROGRAMS_DIR = PROBLEM_DIR / "initial_programs"
KNOWN_OPTIMAL_DISTANCE = 0.7442063311562076


def load_entrypoint(filename):
    path = INITIAL_PROGRAMS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.entrypoint


def check_program(filename):
    entrypoint = load_entrypoint(filename)
    points = entrypoint()
    result = validate(points)

    fitness = result["fitness"]
    absolute_gap = KNOWN_OPTIMAL_DISTANCE - fitness
    relative_gap = 100.0 * absolute_gap / KNOWN_OPTIMAL_DISTANCE

    print(filename)
    print(f"  shape: {np.asarray(points).shape}")
    print(f"  is_valid: {result['is_valid']}")
    print(f"  fitness: {fitness:.10f}")
    print(f"  absolute_gap: {absolute_gap:.10f}")
    print(f"  relative_gap: {relative_gap:.6f}%")


def main():
    check_program("baseline.py")
    check_program("improved.py")


if __name__ == "__main__":
    main()
