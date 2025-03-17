
import polars as pl

def repeat_by_list(df: pl.DataFrame, col: str, n: int) -> pl.Series:
    """
    `repeat_by` implementation for lists
    TODO: replace with native polars `repeat_by` when 1.25.3 is released
    """

    df = df.with_columns(
        pl.col(col).explode().gather_every(2, offset=0).alias(f"{col}.x"),
        pl.col(col).explode().gather_every(2, offset=1).alias(f"{col}.y"),
    )
    df_exploded = df.select(
        pl.col(f"{col}.x").repeat_by(n).flatten(),
        pl.col(f"{col}.y").repeat_by(n).flatten(),
    )
    return df_exploded.with_columns(
        pl.concat_list([pl.col(f"{col}.x"), pl.col(f"{col}.y")]).alias(col)
    )[col]


def get_by_index(series: pl.Series, idx: int) -> pl.Series:
    """
    Get a value from a list by index
    """
    assert series.dtype == pl.List | series.dtype == pl.Array
    
        


