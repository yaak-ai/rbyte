from pathlib import Path

import pytest
from testbook import testbook


def test_nuscenes(monkeypatch: pytest.MonkeyPatch) -> None:
    # hydra needs a relative `config_path`
    monkeypatch.chdir(Path(__file__).parent.parent.resolve() / "examples")

    with testbook("nuscenes.ipynb") as tb:
        tb.execute()
