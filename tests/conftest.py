import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: integration tests that load SAM3/GND on CUDA (skip with -m 'not gpu')",
    )
