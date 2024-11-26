from collections.abc import Hashable, Iterable

__unspecified = object()


# need signature for pipefunc
def make_dict(  # noqa: PLR0913
    *,
    k0: Hashable,
    v0: object,
    k1: Hashable = __unspecified,
    v1: object = __unspecified,
    k2: Hashable = __unspecified,
    v2: object = __unspecified,
    k3: Hashable = __unspecified,
    v3: object = __unspecified,
    k4: Hashable = __unspecified,
    v4: object = __unspecified,
) -> dict[Hashable, object]:
    keys = (k0, k1, k2, k3, k4)
    values = (v0, v1, v2, v3, v4)

    def items() -> Iterable[tuple[Hashable, object]]:
        for key, value in zip(keys, values, strict=True):
            if (key is __unspecified) and (value is __unspecified):
                continue

            elif (key is __unspecified) or (value is __unspecified):
                raise ValueError

            yield (key, value)

    return dict(items())
