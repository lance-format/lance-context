from __future__ import annotations

import tomllib
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
LANCE_PYTHON_DEPS = {"pylance", "lancedb", "lance-namespace"}
PYTHON_VERSION_CLASSIFIERS = {
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
}


def _dependency_names(dependencies: list[str]) -> set[str]:
    return {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("=", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .strip()
        .lower()
        for dependency in dependencies
    }


def test_lance_python_packages_are_optional_runtime_dependencies() -> None:
    pyproject = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    runtime_deps = _dependency_names(project["dependencies"])
    optional_deps = project["optional-dependencies"]

    assert LANCE_PYTHON_DEPS.isdisjoint(runtime_deps)
    assert LANCE_PYTHON_DEPS <= _dependency_names(optional_deps["lance-python"])
    assert LANCE_PYTHON_DEPS <= _dependency_names(optional_deps["tests"])


def test_python_version_classifiers_match_supported_range() -> None:
    pyproject = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    classifiers = set(project["classifiers"])

    assert project["requires-python"] == ">=3.11,<3.14"
    assert PYTHON_VERSION_CLASSIFIERS <= classifiers
    assert "Programming Language :: Python :: 3.9" not in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers
