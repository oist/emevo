"""Qt widget with moderngl visualizer for advanced visualization.
"""
from __future__ import annotations

import enum
import functools
import sys
import warnings
from collections import deque
from collections.abc import Iterable
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.colors as mc
import moderngl
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from numpy.typing import NDArray
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
from emevo.environments.phyjax2d import Circle, State, StateDict
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


N_MAX_SCAN: int = 10000


@jax.jit
def _overlap(p: jax.Array, circle: Circle, state: State) -> jax.Array:
    dist = jnp.linalg.norm(p.reshape(1, 2) - state.p.xy, axis=1)
    return dist < circle.radius


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
        get_colors: Callable[[int], NDArray] | None = None,
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
        self._env = env
        self._get_colors = get_colors
        self._index = start
        self._end_index = self._phys_state.circle_axy.shape[0] if end is None else end
        self._paused = False
        self._initialized = False
        self._overlay_fns = []
        self._showing_energy = False

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
                warnings.warn(f"The qfollowing error occured: {self._ctx.error}")
            self._fbo = self._ctx.detect_framebuffer()
            self._renderer = self._make_renderer(self._ctx)
            self._initialized = True
        if not self._paused and self._index < self._end_index - 1:
            self._index += 1
        stated = self._get_stated(self._index)
        if self._get_colors is None:
            circle_colors = None
        else:
            circle_colors = self._get_colors(self._index)
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(stated, circle_colors=circle_colors)  # type: ignore

    def mousePressEvent(self, evt: QMouseEvent) -> None:  # type: ignore
        position = self._scale_position(evt.position())
        circle = self._get_stated(self._index).circle
        overlap = _overlap(
            jnp.array(position),
            self._env._physics.shaped.circle,
            circle,
        )
        (selected,) = jnp.nonzero(overlap)
        if 0 < selected.shape[0]:
            self.selectionChanged.emit(selected[0].item())

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
            warnings.warn(f"Invalid value for barset: {value}")
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
                warnings.warn(f"Invalid value for barset {value}")

        for name in list(self.barsets.keys()):
            if name not in values:
                old_bs = self.barsets.pop(name)
                new_barsets.popleft().setColor(old_bs.color())
                self.series.remove(old_bs)
        self._update_yrange(values.values())


class CBarState(enum.Enum):
    AGE = 1
    ENERGY = 2
    N_CHILDREN = 3


class CFEnvReplayWidget(QtWidgets.QWidget):
    energyUpdated = Signal(float)
    rewardUpdated = Signal(dict)

    def __init__(
        self,
        xlim: int,
        ylim: int,
        env: CircleForaging,
        saved_physics: SavedPhysicsState,
        start: int = 0,
        end: int | None = None,
        log_offset: int = 0,
        log_ds: ds.Dataset | None = None,
        profile_and_rewards: pa.Table | None = None,
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
            get_colors=None if log_ds is None else self._get_colors,
        )
        self._n_max_agents = env.n_max_agents
        # Log
        self._log_offset = log_offset
        self._log_ds = log_ds
        self._log_cached = []
        # Pause/Play
        pause_button = QtWidgets.QPushButton("⏸️")
        pause_button.clicked.connect(self._mgl_widget.pause)
        play_button = QtWidgets.QPushButton("▶️")
        play_button.clicked.connect(self._mgl_widget.play)
        # Colorbar
        radiobutton_1 = QtWidgets.QRadioButton("Age")
        radiobutton_2 = QtWidgets.QRadioButton("Energy")
        radiobutton_3 = QtWidgets.QRadioButton("Num. Children")
        radiobutton_1.setChecked(True)
        radiobutton_1.toggled.connect(self.cbarAge)
        radiobutton_2.toggled.connect(self.cbarEnergy)
        radiobutton_3.toggled.connect(self.cbarNChildren)
        self._cbar_state = CBarState.AGE
        self._cbar_renderer = CBarRenderer(xlim * 2, ylim // 4)
        self._showing_energy = True
        self._cbar_changed = True
        self._cbar_canvas = FigureCanvasQTAgg(self._cbar_renderer._fig)
        self._value_cm = mpl.colormaps["YlOrRd"]
        self._energy_cm = mpl.colormaps["YlGnBu"]
        self._n_children_cm = mpl.colormaps["PuBuGn"]
        self._norm = mc.Normalize(vmin=0.0, vmax=1.0)
        if profile_and_rewards is not None:
            self._profile_and_rewards = profile_and_rewards
            self._reward_widget = BarChart(self._get_rewards(1))
        # Layout buttons
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(pause_button)
        buttons.addWidget(play_button)
        cbar_selector = QtWidgets.QVBoxLayout()
        cbar_selector.addWidget(radiobutton_1)
        cbar_selector.addWidget(radiobutton_2)
        cbar_selector.addWidget(radiobutton_3)
        buttons.addLayout(cbar_selector)
        # Total layout
        total_layout = QtWidgets.QVBoxLayout()
        total_layout.addLayout(buttons)
        total_layout.addWidget(self._cbar_canvas)
        if profile_and_rewards is None:
            total_layout.addWidget(self._mgl_widget)
        else:
            env_and_reward_layout = QtWidgets.QHBoxLayout()
            env_and_reward_layout.addWidget(self._mgl_widget)
            env_and_reward_layout.addWidget(self._reward_widget)
            total_layout.addLayout(env_and_reward_layout)
        self.setLayout(total_layout)
        timer.start(30)  # 40fps
        # Signals
        self._mgl_widget.selectionChanged.connect(self.updateRewards)
        if profile_and_rewards is not None:
            self.rewardUpdated.connect(self._reward_widget.updateValues)
        # Initial size
        if profile_and_rewards is None:
            self.resize(xlim * 3, ylim * 3)
        else:
            self.resize(xlim * 4, ylim * 3)

    def _get_rewards(self, unique_id: int) -> dict[str, float]:
        filtered = self._profile_and_rewards.filter(pc.field("unique_id") == unique_id)
        return filtered.drop(["birthtime", "parent", "unique_id"]).to_pydict()

    @functools.cache
    def _get_n_children(self, unique_id: int) -> int:
        if self._profile_and_rewards is None:
            warnings.warn("N children requires profile_an_rewards.parquet")
            return 0
        if unique_id == 0:
            return 0
        return len(self._profile_and_rewards.filter(pc.field("parent") == unique_id))

    def _get_colors(self, index: int) -> NDArray:
        assert self._log_ds is not None
        step = self._log_offset + index
        if len(self._log_cached) == 0:
            scanner = self._log_ds.scanner(
                columns=["age", "energy", "step", "slots", "unique_id"],
                filter=(
                    (step <= pc.field("step")) & (pc.field("step") <= step + N_MAX_SCAN)
                ),
            )
            table = scanner.to_table()
            self._log_cached = [
                table.filter(pc.field("step") == i).to_pydict()
                for i in reversed(range(step, step + N_MAX_SCAN))
            ]
        log = self._log_cached.pop()
        slots = np.array(log["slots"])
        if self._cbar_state is CBarState.AGE:
            title = "Age"
            cm = self._value_cm
            age = np.array(log["age"])
            value = np.ones(self._n_max_agents) * np.min(age)
            value[slots] = age
        elif self._cbar_state is CBarState.ENERGY:
            title = "Energy"
            cm = self._energy_cm
            energy = np.array(log["energy"])
            value = np.ones(self._n_max_agents) * np.min(energy)
            value[slots] = energy
        elif self._cbar_state is CBarState.N_CHILDREN:
            title = "Num. Children"
            cm = self._n_children_cm
            value = np.zeros(self._n_max_agents)
            for slot, uid in zip(log["slots"], log["unique_id"]):
                value[slot] = self._get_n_children(uid)
        else:
            warnings.warn(f"Invalid cbar state {self._cbar_state}")
            return np.zeros((self._n_max_agents, 4))
        self._norm.vmin = np.amin(value)  # type: ignore
        self._norm.vmax = np.amax(value)  # type: ignore
        if self._cbar_changed:
            self._cbar_renderer.render(self._norm, cm, title)
            self._cbar_changed = False
            self._cbar_canvas.draw()
        return cm(self._norm(value))

    @Slot(int)
    def updateRewards(self, selected_slot: int) -> None:
        if self._profile_and_rewards is None or selected_slot == -1:
            return

        if len(self._log_cached) == 0:
            return
        last_log = self._log_cached[-1]
        for slot, uid in zip(last_log["slots"], last_log["unique_id"]):
            if slot == selected_slot:
                self.rewardUpdated.emit(self._get_rewards(uid))
                return

    @Slot(bool)
    def cbarAge(self, checked: bool) -> None:
        if checked:
            self._cbar_state = CBarState.AGE
            self._cbar_changed = True

    @Slot(bool)
    def cbarEnergy(self, checked: bool) -> None:
        if checked:
            self._cbar_state = CBarState.ENERGY
            self._cbar_changed = True

    @Slot(bool)
    def cbarNChildren(self, checked: bool) -> None:
        if checked:
            self._cbar_state = CBarState.N_CHILDREN
            self._cbar_changed = True


def start_widget(widget_cls: type[QtWidgets.QWidget], **kwargs) -> None:
    app = QtWidgets.QApplication([])
    widget = widget_cls(**kwargs)
    widget.show()
    sys.exit(app.exec())
