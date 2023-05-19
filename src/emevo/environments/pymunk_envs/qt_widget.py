"""Qt widget with moderngl visualizer for advanced visualization.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import Any, Callable

import moderngl
import numpy as np
import pymunk
from pymunk.vec2d import Vec2d
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QSplineSeries,
    QValueAxis,
)
from PySide6.QtCore import QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPainter, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGridLayout, QWidget

from emevo.environments.pymunk_envs.moderngl_vis import MglRenderer
from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv
from emevo.environments.pymunk_envs.pymunk_utils import CollisionType, make_filter


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


@dataclasses.dataclass
class PanTool:
    """
    Handle mouse drag. Based on moderngl example code:
    https://github.com/moderngl/moderngl/blob/master/examples/renderer_example.py
    """

    body: pymunk.Body | None = None
    shape: pymunk.Shape | None = None
    body_index: int | None = None
    point: Vec2d = dataclasses.field(default_factory=Vec2d.zero)

    def start_drag(self, point: Vec2d, shape: pymunk.Shape, body_index: int) -> None:
        shape.color = shape.color._replace(a=100)  # type: ignore
        self.shape = shape
        self.body = shape.body
        self.body_index = body_index
        self.point = point

    def dragging(self, point: Vec2d) -> bool:
        if self.body is not None:
            delta = point - self.point
            self.point = point
            self.body.position = self.body.position + delta
            if self.body.space is not None:
                self.body.space.reindex_shapes_for_body(self.body)
            return True
        else:
            return False

    def stop_drag(self, point: Vec2d) -> None:
        if self.body is not None:
            self.dragging(point)
            self.shape.color = self.shape.color._replace(a=255)  # type: ignore
            self.body = None
            self.shape = None

    @property
    def is_dragging(self) -> bool:
        return self.body is not None


@dataclasses.dataclass
class AppState:
    pantool: PanTool = dataclasses.field(default_factory=PanTool)
    paused: bool = False
    paused_before: bool = False


def _do_nothing(_state: AppState) -> None:
    pass


class PymunkMglWidget(QOpenGLWidget):
    positionsChanged = Signal(int)
    selectionChanged = Signal(int)

    def __init__(
        self,
        *,
        env: PymunkEnv,
        timer: QTimer,
        app_state: AppState | None = None,
        figsize: tuple[float, float] | None = None,
        step_fn: Callable[[AppState], Iterable[tuple[str, Any]] | None] = _do_nothing,
        parent: QWidget | None = None,
    ) -> None:
        # Set default format
        QSurfaceFormat.setDefaultFormat(_mgl_qsurface_fmt())
        super().__init__(parent)
        # init renderer
        xlim, ylim = env.get_coordinate().bbox()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        self._figsize = int(figsize[0]), int(figsize[1])
        self._scaling = x_range / figsize[0], y_range / figsize[1]
        self._make_renderer = partial(
            MglRenderer,
            screen_width=self._figsize[0],
            screen_height=self._figsize[1],
            x_range=x_range,
            y_range=y_range,
            env=env,
        )
        self._step_fn = step_fn
        self._env = env
        self._state = AppState() if app_state is None else app_state
        self._initialized = False
        self._timer = timer
        self._timer.timeout.connect(self.update)  # type: ignore
        self._overlay_fns = []

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
        self.render()

    def render(self) -> None:
        overlays = self._step_fn(self._state)
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(self._env)  # type: ignore
        if overlays is not None:
            for overlay in overlays:
                self._renderer.overlay(*overlay)

    def _scale_position(self, position: QPointF) -> Vec2d:
        return Vec2d(
            position.x() * self._scaling[0],
            (self._figsize[1] - position.y()) * self._scaling[1],
        )

    def _emit_selected(self, index: int | None) -> None:
        if index is None:
            self.selectionChanged.emit(-1)
        else:
            self.selectionChanged.emit(index)

    def mousePressEvent(self, evt: QMouseEvent) -> None:
        position = self._scale_position(evt.position())
        query = self._env.get_space().point_query(
            position,
            0.0,
            shape_filter=make_filter(CollisionType.AGENT, CollisionType.FOOD),
        )
        if len(query) == 1:
            shape = query[0].shape
            if shape is not None:
                body_index = self._env.get_body_index(shape.body)
                if body_index is not None:
                    self._state.pantool.start_drag(position, shape, body_index)
                    self._emit_selected(body_index)
                    self._paused_before = self._state.paused
                    self._state.paused = True
                    self._timer.stop()
                    self.update()

    def mouseMoveEvent(self, evt: QMouseEvent) -> None:
        if self._state.pantool.shape is not None:
            new_pos = self._scale_position(evt.position())
            if self._state.pantool.dragging(new_pos):
                self.positionsChanged.emit(self._state.pantool.body_index)
                self.update()

    def mouseReleaseEvent(self, evt: QMouseEvent) -> None:
        if self._state.pantool.is_dragging:
            self._state.pantool.stop_drag(self._scale_position(evt.position()))
            self._emit_selected(None)
            self._state.paused = self._state.paused_before
            self._timer.start()
            self.update()

    @Slot()
    def pause(self) -> None:
        self._state.paused = True

    @Slot()
    def play(self) -> None:
        self._state.paused = False


class BarChart(QWidget):
    def __init__(
        self,
        initial_values: dict[str, float | list[float]],
        categ: str = "Rewards",
        title: str = "Bar Chart",
        yrange_min: float | None = None,
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
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

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
        layout = QGridLayout(self)
        layout.addWidget(self._chart_view, 1, 1)
        self.setLayout(layout)
        self.setVisible(True)

    def _make_barset(self, name: str, value: float | list[float]) -> QBarSet:
        barset = QBarSet(name)
        if isinstance(value, float):
            barset.append(value)
        else:
            for v in value:
                barset.append(v)
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
            else:
                for i, vi in enumerate(value):
                    self.barsets[name].replace(i, vi)

        for name in list(self.barsets.keys()):
            if name not in values:
                old_bs = self.barsets.pop(name)
                new_barsets.popleft().setColor(old_bs.color())
                self.series.remove(old_bs)
        self._update_yrange(values.values())


class SplineChart(QChart):
    def __init__(self, title: str) -> None:
        super().__init__(
            QChart.ChartType.ChartTypeCartesian,
            None,  # type: ignore
            Qt.WindowFlags(),  # type: ignore
        )
        self._series = QSplineSeries(self)
        self._titles = []
        self._axis_x = QValueAxis()
        self._axis_y = QValueAxis()

        self.addSeries(self._series)
        self.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        self._series.attachAxis(self._axis_x)
        self._series.attachAxis(self._axis_y)
        self._x = 0
        self._n_scrolled = 0
        self._axis_x.setRange(0, 100)
        self._axis_x.setTickCount(6)
        self._ymin = -1
        self._ymax = 1
        self._axis_y.setRange(self._ymin, self._ymax)
        self._prev_index = None
        self._title = title
        self.setTitle(title)
        self.legend().hide()
        self.setAnimationOptions(QChart.AnimationOption.GridAxisAnimations)
        self.setAnimationDuration(100)
        self._initial_plot_area = None

    @Slot(float)
    def appendValue(self, value: float) -> None:
        if self._initial_plot_area is None:
            self._initial_plot_area = self.plotArea()
        if value < self._ymin:
            self._ymin = value * 1.5
            self._axis_y.setRange(self._ymin, self._ymax)
        elif value > self._ymax:
            self._ymax = value * 1.5
            self._axis_y.setRange(self._ymin, self._ymax)
        self._series.append(self._x, value)
        self._x += 1
        if self._x >= 90 and (self._x % 20) == 0:
            w = int(self._initial_plot_area.width() * 20 / 100)
            self.scroll(w, 0)
            self._n_scrolled += 1

    @Slot(int)
    def reset(self, index: int) -> None:
        self._series.clear()
        self._axis_x.setRange(0, 100)
        self._x = 0
        self._n_scrolled = 0
        if self._prev_index != index and index != -1:
            self.setTitle(f"{self._title} of {index}")
            self._prev_index = index
