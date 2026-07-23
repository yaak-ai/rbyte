from dataclasses import dataclass

import pytest
import rerun as rr
import torch
from pytest_lazy_fixtures import lf
from tensordict import TensorDict

import rbyte.viz.loggers.rerun_logger as rerun_logger_module
from rbyte import Dataset
from rbyte.viz.loggers.rerun_logger import RerunLogger, Schema


@dataclass(frozen=True)
class SendColumnsCall:
    entity_path: str
    indexes: list[rr.TimeColumn]
    columns: rr.ComponentColumnList


@pytest.mark.parametrize(
    ("rerun_logger", "dataset"),
    [
        ("mimicgen", lf("mimicgen_dataset")),
        ("nuscenes", lf("nuscenes_dataset")),
        ("yaak", lf("yaak_dataset")),
        ("zod", lf("zod_dataset")),
    ],
    indirect=["rerun_logger"],
)
def test_rerun_logger(rerun_logger: RerunLogger, dataset: Dataset) -> None:
    rerun_logger.log(dataset.get_batch([0]))


@pytest.fixture
def send_columns_calls(monkeypatch: pytest.MonkeyPatch) -> list[SendColumnsCall]:
    calls: list[SendColumnsCall] = []

    def send_columns(
        *,
        entity_path: str,
        indexes: list[rr.TimeColumn],
        columns: rr.ComponentColumnList,
    ) -> None:
        calls.append(
            SendColumnsCall(entity_path=entity_path, indexes=indexes, columns=columns)
        )

    monkeypatch.setattr(rerun_logger_module.rr, "send_columns", send_columns)

    return calls


@pytest.fixture
def time_index_data() -> TensorDict:
    return TensorDict(
        {
            "data": TensorDict(
                {
                    "dynamic_time_index": torch.tensor([1, 4]),
                    "scalar_dynamic": torch.tensor([7, 8]),
                    "scalar_static": torch.tensor([9]),
                    "timeline_a": torch.arange(10, 16),
                    "timeline_b": torch.arange(20, 26),
                },
                batch_size=[],
            )
        },
        batch_size=[],
    )


def _time_index_logger(
    *, scalar_path: str, time_index: dict[str, object]
) -> RerunLogger:
    schema = Schema.model_validate({
        "timeline_a": {
            "_target_": rr.TimeColumn,
            "columns": {"sequence": ["data", "timeline_a"]},
        },
        "timeline_b": {
            "_target_": rr.TimeColumn,
            "columns": {"sequence": ["data", "timeline_b"]},
        },
        "scalar": [
            {
                "_target_": rr.Scalars.columns,
                "columns": {"scalars": ["data", scalar_path]},
                "time_index": time_index,
            }
        ],
    })

    return RerunLogger(
        application_id="test", recording_name="test", schema=schema, spawn=False
    )


def _sent_time_values(call: SendColumnsCall) -> list[list[int]]:
    return [column.times.to_pylist() for column in call.indexes]


def test_rerun_logger_sends_static_time_index_to_rerun(
    send_columns_calls: list[SendColumnsCall], time_index_data: TensorDict
) -> None:
    logger = _time_index_logger(
        scalar_path="scalar_static", time_index={"index": "[-1:]"}
    )

    logger._log(time_index_data)  # ruff:ignore[private-member-access]

    [call] = send_columns_calls
    assert call.entity_path == "scalar"
    assert _sent_time_values(call) == [[15], [25]]


def test_rerun_logger_sends_dynamic_time_index_to_rerun(
    send_columns_calls: list[SendColumnsCall], time_index_data: TensorDict
) -> None:
    logger = _time_index_logger(
        scalar_path="scalar_dynamic",
        time_index={"path": ["data", "dynamic_time_index"]},
    )

    logger._log(time_index_data)  # ruff:ignore[private-member-access]

    [call] = send_columns_calls
    assert call.entity_path == "scalar"
    assert _sent_time_values(call) == [[11, 14], [21, 24]]
