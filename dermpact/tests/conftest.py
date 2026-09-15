"""Shared pytest configuration for the dermpact test tree."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: loads the frozen five-member perception model through the inference worker (minutes on CPU)",
    )
