COLOR_CYCLE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True


def next_color(index):
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]
