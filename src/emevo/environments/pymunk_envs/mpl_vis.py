from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pymunk.matplotlib_util import DrawOptions

from emevo.env import Visualizer
from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv


class MplVisualizer(Visualizer):
    def __init__(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        figsize: tuple[float, float] | None = None,
        ax: Axes | None = None,
    ) -> None:
        if figsize is None:
            figsize = 8.0, 8.0
        if ax is None:
            self._fig = plt.figure(figsize=figsize)
            self._ax = self._fig.add_subplot()
        else:
            self._fig = ax.get_figure()
            self._ax = ax

        self._ax.set_aspect("equal")
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._draw_options = DrawOptions(self._ax)

    def render(self, env: PymunkEnv) -> Figure:
        space = env.get_space()
        space.debug_draw(self._draw_options)
        return self._fig

    def open(self) -> None:
        raise NotImplementedError("Open is not implemented for MPL vis")
