import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """avoid `pytest -m compile` failure if no other compile tests are present"""
    pass
