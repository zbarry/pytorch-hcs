"""
Utilities for data visualization
"""

# Because hv.opt.* are not generated til extensions are loaded
# pylint: disable=no-member

import holoviews as hv
import numpy as np
from holoviews.operation.datashader import rasterize

fontsize = dict(
    title=16,
    xlabel=15,
    ylabel=15,
    zlabel=15,
    xticks=13,
    minor_xticks=12,
    yticks=12,
    minor_yticks=11,
    zticks=13,
    legend=14,
    legend_title=15,
)

colorbar_opts = dict(major_label_text_font_size="12pt")


def set_hv_defaults():
    """ Useful HoloViews defaults. """
    hv.extension("bokeh", "matplotlib")

    # hv.opts.* don't show up until the backend is loaded

    image_opts = hv.opts.Image(
        aspect="equal",
        frame_width=350,
        fontsize=fontsize,
        colorbar=True,
        normalize=False,
        active_tools=["wheel_zoom"],
        cmap="gray",
        colorbar_opts=colorbar_opts,
    )

    heatmap_opts = hv.opts.HeatMap(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    rgb_opts = hv.opts.RGB(
        aspect="equal",
        active_tools=["wheel_zoom"],
        frame_width=350,
        fontsize=fontsize,
    )
    curve_opts = hv.opts.Curve(active_tools=["wheel_zoom"], fontsize=fontsize)
    path_opts = hv.opts.Path(active_tools=["wheel_zoom"], fontsize=fontsize)
    bars_opts = hv.opts.Bars(active_tools=["wheel_zoom"], fontsize=fontsize)
    scatter_opts = hv.opts.Scatter(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    histogram_opts = hv.opts.Histogram(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    violin_opts = hv.opts.Violin(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    boxwhisker_opts = hv.opts.BoxWhisker(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    errorbars_opts = hv.opts.ErrorBars(
        active_tools=["wheel_zoom"], fontsize=fontsize
    )
    layout_opts = hv.opts.Layout(fontsize=dict(title=20))

    _ = hv.DynamicMap(lambda: None)
    hv.DynamicMap.cache_size = 1
    hv.util.settings.OutputSettings.options["max_frames"] = 100000000

    hv.opts.defaults(
        heatmap_opts,
        image_opts,
        rgb_opts,
        curve_opts,
        path_opts,
        bars_opts,
        scatter_opts,
        histogram_opts,
        violin_opts,
        boxwhisker_opts,
        errorbars_opts,
        layout_opts,
    )


def make_rasterized_plot(image: np.ndarray,):
    """
    Helper function for configuring HoloViews image plots
    """

    if image.ndim == 2:
        cls = hv.Image
        opts = hv.opts.Image(aspect="equal", colorbar=True, tools=["hover"])
    else:
        cls = hv.RGB
        opts = hv.opts.RGB(aspect="equal")

    return rasterize(
        cls(image, bounds=(0, 0, image.shape[1], image.shape[0]),),
        precompute=True,
        aggregator="mean",
    ).opts(opts)
