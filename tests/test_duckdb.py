import pickle  # ruff:ignore[suspicious-pickle-import]

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from rbyte.samples.duckdb import DuckDBQuery


@pytest.mark.parametrize(
    "query", ["INSERT INTO values_table VALUES (1)", "SELECT 1; SELECT 2"]
)
def test_rejects_queries_other_than_a_single_select(query: str) -> None:
    with pytest.raises(ValidationError, match="invalid query"):
        DuckDBQuery(query=query)


def test_unregisters_views_after_query_failure() -> None:
    query = DuckDBQuery(query="SELECT missing_column FROM dataframe")

    with pytest.raises(duckdb.BinderException):
        query(dataframe=pl.DataFrame({"value": [1]}))

    assert (
        query.con.execute(
            "SELECT view_name FROM duckdb_views() WHERE view_name = 'dataframe'"
        ).fetchall()
        == []
    )


def test_initialized_connection_survives_pickle_round_trip() -> None:
    query = DuckDBQuery(query="SELECT $value AS value")
    expected = 42
    _ = query.con

    round_tripped = pickle.loads(  # ruff:ignore[suspicious-pickle-usage]
        pickle.dumps(query)
    )

    assert round_tripped(value=expected).item() == expected
