"""Standalone acceptance is also runnable under pytest on development hosts."""
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "molt_adapter"))


@pytest.mark.skipif(os.environ.get("DSH_RUN_NATIVE_TESTS") != "1", reason="Requires host namespaces")
def test_molt_native_tokens_logprobs_reward(monkeypatch):
    from acceptance import accept
    from test_adapter import PIPELINE
    monkeypatch.setenv("DSH_VERIFIER_ROOT", str(PIPELINE))
    accept()
