"""Pytest configuration for the simulation test suite."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--result-file", default=None)
