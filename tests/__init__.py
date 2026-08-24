"""Test package.

This file exists so `tests` is a real package rather than a loose folder,
which is what makes `from tests.conftest import ...` resolve.

Without it, that import only works when the repository root happens to be on
`sys.path` — which `python -m pytest` arranges (it prepends the working
directory) and a bare `pytest` does not. The result was a test suite that
passed locally and failed in CI at collection time, having run nothing at
all. With this file, pytest walks up past `tests/` to the first directory
without an `__init__.py`, puts *that* on the path, and both invocations
behave the same.
"""
