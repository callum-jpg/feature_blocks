import matplotlib


def make_cmap(colors: list, add_background: bool = True):
    """For a list of hexcode colors, create a matplotlib
    colormap. It's intended to be a discrete colormap, so the number of colors
    will be the length of colors (+1 if add_background == True)"""
    # Make a copy to prevent in place modification of adata
    colors_copy = colors.copy()
    n = len(colors_copy)
    cmap_data = [matplotlib.colors.to_rgb(i) for x, i in enumerate(colors_copy)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", cmap_data, N=n
    )
    if add_background:
        cmap.set_bad(color="black")

    return cmap
