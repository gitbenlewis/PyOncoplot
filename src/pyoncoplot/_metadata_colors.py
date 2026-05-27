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


def categorical_metadata_palette(spec: Any, column: object) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"metadata_palette entry for categorical metadata column {str(column)!r} "
            "must be a mapping of category values to colors."
        )
    return {str(level): color for level, color in spec.items()}


def numeric_metadata_colormap_spec(spec: Any) -> Any:
    if spec is None or isinstance(spec, Mapping):
        return None
    return spec


def _coerce_continuous_colormap(spec: Any, column: object):
    try:
        from matplotlib import colormaps
        from matplotlib.colors import Colormap, LinearSegmentedColormap
    except ImportError as exc:
        raise ImportError("Continuous metadata colormaps require the 'matplotlib' package.") from exc

    if isinstance(spec, Colormap):
        return spec

    if isinstance(spec, str):
        try:
            return colormaps.get_cmap(spec)
        except ValueError as exc:
            raise ValueError(
                f"Unknown colormap {spec!r} for numeric metadata column {str(column)!r}."
            ) from exc

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


def numeric_metadata_endpoint_colors(spec: Any, column: object) -> tuple[str, str]:
    return (
        sample_numeric_metadata_colormap(spec, 0.0, 0.0, 1.0, column),
        sample_numeric_metadata_colormap(spec, 1.0, 0.0, 1.0, column),
    )
