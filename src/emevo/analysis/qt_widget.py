"""Qt widget with moderngl visualizer for advanced visualization.
"""
from __future__ import annotations

import dataclasses
import sys
from collections import deque
from collections.abc import Iterable
from functools import partial

import jax
import matplotlib as mpl
import matplotlib.colors as mc
import moderngl
import numpy as np
import pyarrow as pa
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtWidgets
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QValueAxis,
)
from PySide6.QtCore import QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPainter, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from emevo.environments.circle_foraging import CircleForaging
from emevo.environments.moderngl_vis import MglRenderer
from emevo.environments.phyjax2d import StateDict
from emevo.exp_utils import SavedPhysicsState
from emevo.plotting import CBarRenderer


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


class MglWidget(QOpenGLWidget):
    selectionChanged = Signal(int)

    def __init__(
        self,
        *,
        timer: QTimer,
        env: CircleForaging,
        saved_physics: SavedPhysicsState,
        figsize: tuple[float, float],
        start: int = 0,
        end: int | None = None,
        log_offset: int = 0,
        log_table: pa.Table | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # Set default format
        QSurfaceFormat.setDefaultFormat(_mgl_qsurface_fmt())
        super().__init__(parent)
        # init renderer
        self._env_state, _ = env.reset(jax.random.PRNGKey(0))
        self._figsize = int(figsize[0]), int(figsize[1])
        x_range, y_range = env._x_range, env._y_range
        self._scaling = x_range / figsize[0], y_range / figsize[1]
        self._phys_state = saved_physics
        self._make_renderer = partial(
            MglRenderer,
            screen_width=self._figsize[0],
            screen_height=self._figsize[1],
            x_range=x_range,
            y_range=y_range,
            space=env._physics,
            stated=self._get_stated(0),
            sensor_fn=env._get_sensors,
        )
        self._log_offset = log_offset
        self._log_table = log_table
        self._index = start
        self._end_index = self._phys_state.circle_axy.shape[0] if end is None else end
        self._paused = False
        self._initialized = False
        self._overlay_fns = []

        # Set timer
        self._timer = timer
        self._timer.timeout.connect(self.update)

        self.setFixedSize(*self._figsize)
        self.setMouseTracking(True)

    def _scale_position(self, position: QPointF) -> tuple[float, float]:
        return (
            position.x() * self._scaling[0],
            (self._figsize[1] - position.y()) * self._scaling[1],
        )

    def _get_stated(self, index: int) -> StateDict:
        return self._phys_state.set_by_index(index, self._env_state.physics)

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
        if not self._paused and self._index < self._end_index - 1:
            self._index += 1
        self._render(self._get_stated(self._index))

    def _render(self, stated: StateDict) -> None:
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(stated)  # type: ignore

    def _emit_selected(self, index: int | None) -> None:
        if index is None:
            self.selectionChanged.emit(-1)
        else:
            self.selectionChanged.emit(index)

    def mousePressEvent(self, evt: QMouseEvent) -> None:  # type: ignore
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

    def mouseReleaseEvent(self, evt: QMouseEvent) -> None:  # type: ignore
        pass

    @Slot()
    def pause(self) -> None:
        self._paused = True

    @Slot()
    def play(self) -> None:
        self._paused = False


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


class CFEnvReplayWidget(QtWidgets.QWidget):
    energyUpdated = Signal(float)
    rewardUpdated = Signal(dict)
    foodrankUpdated = Signal(dict)
    valueUpdated = Signal(float)

    def __init__(
        self,
        xlim: int,
        ylim: int,
        env: CircleForaging,
        saved_physics: SavedPhysicsState,
        start: int = 0,
        end: int | None = None,
        log_offset: int = 0,
        log_table: pa.Table | None = None,
        profile_and_reward: pa.Table | None = None,
    ) -> None:
        super().__init__()

        timer = QTimer()
        # Environment
        self._mgl_widget = MglWidget(
            timer=timer,
            env=env,
            saved_physics=saved_physics,
            figsize=(xlim * 2, ylim * 2),
            start=start,
            end=end,
            log_offset=log_offset,
            log_table=log_table,
        )
        # Pause/Play
        self._pause_button = QtWidgets.QPushButton("⏸️")
        self._pause_button.clicked.connect(self._mgl_widget.pause)
        self._play_button = QtWidgets.QPushButton("▶️")
        self._play_button.clicked.connect(self._mgl_widget.play)
        self._cbar_select_button = QtWidgets.QPushButton("Switch Value/Energy")
        self._cbar_select_button.clicked.connect(self.change_cbar)
        # Colorbar
        self._cbar_renderer = CBarRenderer(xlim * 2, ylim // 4)
        self._showing_energy = True
        self._cbar_changed = True
        self._cbar_canvas = FigureCanvasQTAgg(self._cbar_renderer._fig)
        self._value_cm = mpl.colormaps["YlOrRd"]
        self._energy_cm = mpl.colormaps["YlGnBu"]
        self._norm = mc.Normalize(vmin=0.0, vmax=1.0)
        if profile_and_reward is not None:
            self._reward_widget = BarChart(
                next(iter(self._rewards.values())).to_pydict()
            )
        # Layout buttons
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._pause_button)
        buttons.addWidget(self._play_button)
        buttons.addWidget(self._cbar_select_button)
        # Total layout
        total_layout = QtWidgets.QVBoxLayout()
        total_layout.addLayout(buttons)
        total_layout.addWidget(self._cbar_canvas)
        if profile_and_reward is None:
            total_layout.addWidget(self._mgl_widget)
        else:
            env_and_reward_layout = QtWidgets.QHBoxLayout()
            env_and_reward_layout.addWidget(self._mgl_widget)
            env_and_reward_layout.addWidget(self._reward_widget)
            total_layout.addLayout(env_and_reward_layout)
        self.setLayout(total_layout)
        timer.start(30)  # 40fps
        self._arrow_cached = None
        self._obs_cached = {}
        # Signals
        self._mgl_widget.selectionChanged.connect(self.updateRewards)
        if profile_and_reward is not None:
            self.rewardUpdated.connect(self._reward_widget.updateValues)
        # Initial size
        self.resize(xlim * 3, int(ylim * 2.4))

    @Slot(int)
    def updateRewards(self, body_index: int) -> None:
        pass
        # if self._rewards is None or body_index == -1:
        #     return
        # self.rewardUpdated.emit(self._rewards[body_index].to_pydict())

    @Slot()
    def change_cbar(self) -> None:
        self._showing_energy = not self._showing_energy
        self._cbar_changed = True


def start_widget(widget_cls: type[QtWidgets.QtWidget], **kwargs) -> None:
    app = QtWidgets.QApplication([])
    widget = widget_cls(**kwargs)
    widget.show()
    sys.exit(app.exec())
