"""
pytest configuration for AU-Fukaya compiler tests.
Markers:
  slow   — requires training (minutes)
  corpus — requires /tmp/train_ids.json etc
"""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (training required)")
    config.addinivalue_line("markers", "corpus: marks tests requiring corpus files")
