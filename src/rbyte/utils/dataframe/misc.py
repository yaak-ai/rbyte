from collections.abc import Generator, Mapping

import polars as pl
import polars._typing as plt


# TODO: https://github.com/pola-rs/polars/issues/12353  # noqa: FIX002
def unnest_all(
    schema: Mapping[str, plt.PolarsDataType], separator: str = "."
) -> Generator[pl.Expr]:
    def _unnest(
        schema: Mapping[str, plt.PolarsDataType], path: tuple[str, ...] = ()
    ) -> Generator[tuple[tuple[str, ...], plt.PolarsDataType]]:
        for name, dtype in schema.items():
            match dtype:
                case pl.Struct():
                    yield from _unnest(dtype.to_schema(), (*path, name))

                case _:
                    yield (*path, name), dtype

    for (col, *fields), _ in _unnest(schema):
        expr = pl.col(col)

        for field in fields:
            expr = expr.struct[field]

        name = separator.join(fields if not col else (col, *fields))

        yield expr.alias(name)
