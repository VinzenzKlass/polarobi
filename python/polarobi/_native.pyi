from typing import Any

import polars as pl

def load(path: str) -> tuple[int, int, int]: ...
def add_vars(
    cmodel: Any,
    n: int,
    obj: pl.Series | None = None,
    lb: pl.Series | None = None,
    ub: pl.Series | None = None,
    vtype: pl.Series | None = None,
    names: pl.Series | None = None,
) -> None: ...
def add_constrs(
    cmodel: Any,
    cbeg: pl.Series,
    cind: pl.Series,
    cval: pl.Series,
    sense: pl.Series,
    rhs: pl.Series,
    names: pl.Series | None = None,
) -> None: ...

class ArrowArray:
    """An Arrow array exported through the Arrow PyCapsule interface."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

def get_attr(cmodel: Any, attr: str, ind: pl.Series, int: bool = False) -> ArrowArray: ...
def set_attr(cmodel: Any, attr: str, ind: pl.Series, values: pl.Series) -> None: ...
def register(pool: list[Any], cls: type[Any], cmodel: Any, start: int, count: int) -> None: ...
