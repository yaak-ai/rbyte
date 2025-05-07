from pathlib import Path

import pytest
from testbook import testbook


@pytest.mark.parametrize("file", ["nuscenes.ipynb"])
def test_example(file: str) -> None:
    with pytest.MonkeyPatch.context() as mp:
        # hydra needs a relative `config_path`
        mp.chdir(Path(__file__).parent.parent.resolve() / "examples")

        with testbook(file) as tb:
            tb.execute()
