"""Qt widget with moderngl visualizer for advanced visualization.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import Callable

import moderngl
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QValueAxis,
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPainter, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6 import QtWidgets

from emevo.environments.moderngl_vis import MglRenderer
from emevo.environments.phyjax2d import Space, StateDict


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


@dataclasses.dataclass
class AppState:
    selected: int | None = None
    paused: bool = False
    paused_before: bool = False


class MglWidget(QOpenGLWidget):
    selectionChanged = Signal(int)

    def __init__(
        self,
        *,
        x_range: float,
        y_range: float,
        space: Space,
        stated: StateDict,
        figsize: tuple[float, float],
        sensor_fn: Callable[[StateDict], tuple[NDArray, NDArray]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # Set default format
        QSurfaceFormat.setDefaultFormat(_mgl_qsurface_fmt())
        super().__init__(parent)
        # init renderer
        self._figsize = int(figsize[0]), int(figsize[1])
        self._scaling = x_range / figsize[0], y_range / figsize[1]
        self._make_renderer = partial(
            MglRenderer,
            screen_width=self._figsize[0],
            screen_height=self._figsize[1],
            x_range=x_range,
            y_range=y_range,
            space=space,
            stated=stated,
            sensor_fn=sensor_fn,
        )
        self._state = AppState()
        self._initialized = False
        self._overlay_fns = []
        self._initial_state = stated

        self.setFixedSize(*self._figsize)
        self.setMouseTracking(True)

    def _set_default_viewport(self) -> None:
        self._ctx.viewport = 0, 0, *self._figsize
        self._fbo.viewport = 0, 0, *self._figsize

    def paintGL(self) -> None:
        if not self._initialized:
            if QGuiApplication.platformName() in ["eglfs", "wayland"]:
                self._ctx = moderngl.create_context(
                    require=410,
                    share=True,
                    backend="egl",  # type: ignore
                )
            else:
                self._ctx = moderngl.create_context(require=410)
            if self._ctx.error != "GL_NO_ERROR":
                raise RuntimeError(f"The following error occured: {self._ctx.error}")
            self._fbo = self._ctx.detect_framebuffer()
            self._renderer = self._make_renderer(self._ctx)
            self._initialized = True
        self.render(self._initial_state)

    def render(self, stated: StateDict) -> None:
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(stated)  # type: ignore

    def show(self, timer: QTimer):
        self._timer = timer
        self._timer.timeout.connect(self.update)  # type: ignore

    def _emit_selected(self, index: int | None) -> None:
        if index is None:
            self.selectionChanged.emit(-1)
        else:
            self.selectionChanged.emit(index)

    def mousePressEvent(self, evt: QMouseEvent) -> None:
        position = self._scale_position(evt.position())
        # query = self._env.get_space().point_query(
        #     position,
        #     0.0,
        #     shape_filter=make_filter(CollisionType.AGENT, CollisionType.FOOD),
        # )
        # if len(query) == 1:
        #     shape = query[0].shape
        #     if shape is not None:
        #         body_index = self._env.get_body_index(shape.body)
        #         if body_index is not None:
        #             self._state.pantool.start_drag(position, shape, body_index)
        #             self._emit_selected(body_index)
        #             self._paused_before = self._state.paused
        #             self._state.paused = True
        #             self._timer.stop()
        #             self.update()

    def mouseReleaseEvent(self, evt: QMouseEvent) -> None:
        pass

    @Slot()
    def pause(self) -> None:
        self._state.paused = True

    @Slot()
    def play(self) -> None:
        self._state.paused = False


class BarChart(QtWidgets.QWidget):
    def __init__(
        self,
        initial_values: dict[str, float | list[float]],
        categ: str = "Rewards",
        title: str = "Bar Chart",
        yrange_min: float | None = None,
        animation: bool = True,
    ) -> None:
        super().__init__()
        self._yrange_min = yrange_min

        self.barsets = {}
        self.series = QBarSeries()

        for name, value in initial_values.items():
            self._make_barset(name, value)

        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setTitle(title)
        if animation:
            self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        else:
            self.chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        self.axis_x = QBarCategoryAxis()
        self.axis_x.append([categ])
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self._update_yrange(initial_values.values())
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)

        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        self._chart_view = QChartView(self.chart)
        self._chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._chart_view.chart().show()
        self._chart_view.chart().legend().show()
        # create main layout
        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self._chart_view, 1, 1)
        self.setLayout(layout)
        self.setVisible(True)

    def _make_barset(self, name: str, value: float | list[float]) -> QBarSet:
        barset = QBarSet(name)
        if isinstance(value, float):
            barset.append(value)
        elif isinstance(value, list):
            for v in value:
                barset.append(v)
        else:
            raise ValueError(f"Invalid value for barset: {value}")
        self.barsets[name] = barset
        self.series.append(barset)
        return barset

    def _update_yrange(self, values: Iterable[float | list[float]]) -> None:
        values_arr = np.array(list(values))
        if self._yrange_min is None:
            yrange_min = np.min(values_arr)
        else:
            yrange_min = min(self._yrange_min, np.min(values_arr))
        self.axis_y.setRange(yrange_min, np.max(values_arr))

    @Slot(dict)
    def updateValues(self, values: dict[str, float | list[float]]) -> None:
        new_barsets = deque()
        for name, value in values.items():
            if name not in self.barsets:
                barset = self._make_barset(name, value)
                new_barsets.append(barset)
            elif isinstance(value, float):
                self.barsets[name].replace(0, value)
            elif isinstance(value, list):
                for i, vi in enumerate(value):
                    self.barsets[name].replace(i, vi)
            else:
                raise ValueError(f"Invalid value for barset {value}")

        for name in list(self.barsets.keys()):
            if name not in values:
                old_bs = self.barsets.pop(name)
                new_barsets.popleft().setColor(old_bs.color())
                self.series.remove(old_bs)
        self._update_yrange(values.values())
