"""Color palettes and ramps for oncoplot annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import cm, colors, colormaps

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


tol_colors = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#661100",
    "#CC6677",
    "#882255",
    "#AA4499",
]

Iridescent = [
    "#FEFBE9",
    "#FCF7D5",
    "#F5F3C1",
    "#EAF0B5",
    "#DDECBF",
    "#D0E7CA",
    "#C2E3D2",
    "#B5DDD8",
    "#A8D8DC",
    "#9BD2E1",
    "#8DCBE4",
    "#81C4E7",
    "#7BBCE7",
    "#7EB2E4",
    "#88A5DD",
    "#9398D2",
    "#9B8AC4",
    "#9D7DB2",
    "#9A709E",
    "#906388",
    "#805770",
    "#684957",
    "#46353A",
]

# Colorblindness adjusted vega_10.
vega_10 = list(map(colors.to_hex, cm.tab10.colors))
vega_10_scanpy = vega_10.copy()
vega_10_scanpy[2] = "#279e68"
vega_10_scanpy[4] = "#aa40fc"
vega_10_scanpy[8] = "#b5bd61"

# Default matplotlib 2.0 palette.
vega_20 = list(map(colors.to_hex, cm.tab20.colors))

vega_20_scanpy = [
    *vega_20[0:14:2],
    *vega_20[16::2],
    *vega_20[1:15:2],
    *vega_20[17::2],
    "#ad494a",
    "#8c6d31",
]
vega_20_scanpy[2] = vega_10_scanpy[2]
vega_20_scanpy[4] = vega_10_scanpy[4]
vega_20_scanpy[7] = vega_10_scanpy[8]

default_20 = vega_20_scanpy

zeileis_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    "#7f7f7f",
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

default_28 = zeileis_28

godsnot_102 = [
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]

default_102 = godsnot_102


def make_greyzero_colormap(name: str, base_cmap: str):
    """Create a colormap with exact zero shown in gray before the base ramp."""

    cmap = colors.ListedColormap(
        ["gray"] + list(colormaps.get_cmap(base_cmap)(np.linspace(0, 1, 256))),
        name=name,
    )
    try:
        colormaps.register(cmap=cmap, name=name)
    except ValueError:
        pass
    return cmap


viridis_greyzero = make_greyzero_colormap("viridis_greyzero", "viridis")
plasma_greyzero = make_greyzero_colormap("plasma_greyzero", "plasma")
magma_greyzero = make_greyzero_colormap("magma_greyzero", "magma")
inferno_greyzero = make_greyzero_colormap("inferno_greyzero", "inferno")
cividis_greyzero = make_greyzero_colormap("cividis_greyzero", "cividis")
turbo_greyzero = make_greyzero_colormap("turbo_greyzero", "turbo")


def _plot_color_cycle(clists: "Mapping[str, Sequence[str]]") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig, axes = plt.subplots(nrows=len(clists))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.3, right=0.99)
    axes[0].set_title("Color Maps/Cycles", fontsize=14)

    for ax, (name, clist) in zip(axes, clists.items()):
        n = len(clist)
        ax.imshow(
            np.arange(n)[None, :].repeat(2, 0),
            aspect="auto",
            cmap=ListedColormap(clist),
            norm=BoundaryNorm(np.arange(n + 1) - 0.5, n),
        )
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.0
        fig.text(x_text, y_text, name, va="center", ha="right", fontsize=10)

    for ax in axes:
        ax.set_axis_off()
    fig.show()


__all__ = [
    "Iridescent",
    "cividis_greyzero",
    "default_20",
    "default_28",
    "default_102",
    "godsnot_102",
    "inferno_greyzero",
    "magma_greyzero",
    "make_greyzero_colormap",
    "plasma_greyzero",
    "tol_colors",
    "turbo_greyzero",
    "vega_10",
    "vega_10_scanpy",
    "vega_20",
    "vega_20_scanpy",
    "viridis_greyzero",
    "zeileis_28",
]


if __name__ == "__main__":
    _plot_color_cycle({name: colors for name, colors in globals().items() if isinstance(colors, list)})
