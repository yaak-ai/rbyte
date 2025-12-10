from pathlib import Path

import pytest
from testbook import testbook


@pytest.mark.parametrize("file", ["nuscenes.ipynb"])
def test_notebook(file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as ctx:
        # hydra needs a relative `config_path`
        ctx.chdir(Path(__file__).parent.parent.resolve() / "examples")

        with testbook(file) as tb:
            tb.execute()
