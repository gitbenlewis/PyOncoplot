"""Metadata color helpers shared by renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional


def metadata_palette_spec(metadata_palette: Optional[Mapping[str, Any]], column: object) -> Any:
    if not metadata_palette:
        return None
    if column in metadata_palette:
        return metadata_palette[column]
    return metadata_palette.get(str(column))


def _resolve_named_metadata_palette(spec: str, column: object, kind: str) -> Any:
    from matplotlib import colormaps

    from . import palettes as pyoncoplot_palettes

    if spec in getattr(pyoncoplot_palettes, "__all__", ()):
        return getattr(pyoncoplot_palettes, spec)

    try:
        return colormaps.get_cmap(spec)
    except ValueError as exc:
        raise ValueError(
            f"Unknown metadata palette {spec!r} for {kind} metadata column {str(column)!r}."
        ) from exc


def _sequence_colors(spec: Any, column: object, kind: str) -> list[Any]:
    colors = list(spec)
    if not colors:
        raise ValueError(
            f"metadata_palette entry for {kind} metadata column {str(column)!r} "
            "must include at least one color."
        )
    return colors


def _coerce_categorical_colors(spec: Any, column: object, level_count: int) -> list[str]:
    from matplotlib.colors import Colormap, ListedColormap, to_hex

    if isinstance(spec, str):
        spec = _resolve_named_metadata_palette(spec, column, "categorical")

    if isinstance(spec, ListedColormap) and getattr(spec, "colors", None) is not None:
        return [
            to_hex(color, keep_alpha=False)
            for color in _sequence_colors(spec.colors, column, "categorical")
        ]

    if isinstance(spec, Colormap):
        if level_count <= 0:
            return []
        if level_count == 1:
            return [to_hex(spec(0.0), keep_alpha=False)]
        return [
            to_hex(spec(index / (level_count - 1)), keep_alpha=False)
            for index in range(level_count)
        ]

    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray, str)):
        return [
            to_hex(color, keep_alpha=False)
            for color in _sequence_colors(spec, column, "categorical")
        ]

    raise ValueError(
        f"metadata_palette entry for categorical metadata column {str(column)!r} "
        "must be a mapping, palette name, matplotlib colormap, or sequence of colors."
    )


def categorical_metadata_palette(
    spec: Any,
    column: object,
    levels: Sequence[object] = (),
) -> dict[str, Any]:
    if spec is None:
        return {}
    level_names = [str(level) for level in levels]
    if isinstance(spec, Mapping):
        supplied = {str(level): color for level, color in spec.items()}
        if level_names:
            return {level: supplied[level] for level in level_names if level in supplied}
        return supplied

    colors = _coerce_categorical_colors(spec, column, len(level_names))
    if not level_names:
        return {}
    return {
        level: colors[index % len(colors)]
        for index, level in enumerate(level_names)
    }


def numeric_metadata_colormap_spec(spec: Any) -> Any:
    if spec is None or isinstance(spec, Mapping):
        return None
    return spec


def _coerce_continuous_colormap(spec: Any, column: object):
    try:
        from matplotlib.colors import Colormap, LinearSegmentedColormap
    except ImportError as exc:
        raise ImportError("Continuous colormaps require the 'matplotlib' package.") from exc

    if isinstance(spec, Colormap):
        return spec

    if isinstance(spec, str):
        return _coerce_continuous_colormap(
            _resolve_named_metadata_palette(spec, column, "numeric"),
            column,
        )

    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray, str)):
        colors = list(spec)
        if not colors:
            raise ValueError(
                f"metadata_palette entry for numeric metadata column {str(column)!r} "
                "must include at least one color."
            )
        try:
            return LinearSegmentedColormap.from_list(f"pyoncoplot_{str(column)}", colors)
        except ValueError as exc:
            raise ValueError(
                f"metadata_palette entry for numeric metadata column {str(column)!r} "
                "must be a matplotlib colormap name or a sequence of valid colors."
            ) from exc

    raise ValueError(
        f"metadata_palette entry for numeric metadata column {str(column)!r} "
        "must be a matplotlib colormap name or a sequence of colors."
    )


def sample_numeric_metadata_colormap(
    spec: Any,
    value: object,
    min_value: float,
    max_value: float,
    column: object,
) -> str:
    from matplotlib.colors import to_hex

    span = max(max_value - min_value, 1e-9)
    fraction = min(1.0, max(0.0, (float(value) - min_value) / span))
    colormap = _coerce_continuous_colormap(spec, column)
    return to_hex(colormap(fraction), keep_alpha=False)


def numeric_metadata_colorscale(spec: Any, column: object, n: int = 257) -> list[list[object]]:
    from matplotlib.colors import to_hex

    colormap = _coerce_continuous_colormap(spec, column)
    n = max(2, int(n))
    return [
        [index / (n - 1), to_hex(colormap(index / (n - 1)), keep_alpha=False)]
        for index in range(n)
    ]


def continuous_colorscale(spec: Any, column: object, n: int = 257) -> list[list[object]]:
    return numeric_metadata_colorscale(spec, column, n=n)


def coerce_continuous_colormap(spec: Any, column: object):
    return _coerce_continuous_colormap(spec, column)


def numeric_metadata_endpoint_colors(spec: Any, column: object) -> tuple[str, str]:
    return (
        sample_numeric_metadata_colormap(spec, 0.0, 0.0, 1.0, column),
        sample_numeric_metadata_colormap(spec, 1.0, 0.0, 1.0, column),
    )
