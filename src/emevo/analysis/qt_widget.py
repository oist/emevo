"""Qt widget with moderngl visualizer for advanced visualization.
"""

from __future__ import annotations

import enum
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
from phyjax2d import Circle, State, StateDict
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


N_MAX_SCAN: int = 20480
N_MAX_CACHED_LOG: int = 100


@jax.jit
def _overlap(p: jax.Array, circle: Circle, state: State) -> jax.Array:
    dist = jnp.linalg.norm(p.reshape(1, 2) - state.p.xy, axis=1)
    return dist < circle.radius


class MglWidget(QOpenGLWidget):
    selectionChanged = Signal(int, int)
    stepChanged = Signal(int)

    def __init__(
        self,
        *,
        timer: QTimer,
        env: CircleForaging,
        saved_physics: SavedPhysicsState,
        figsize: tuple[float, float],
        start: int = 0,
        slider_offset: int = 0,
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
            sc_color_opt=env._food_color,
        )
        self._env = env
        self._get_colors = get_colors
        self._index = start
        self._end_index = self._phys_state.circle_axy.shape[0] if end is None else end
        self._paused = False
        self._initialized = False
        self._overlay_fns = []
        self._showing_energy = False
        self._slider_offset = slider_offset

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
                warnings.warn(
                    f"The qfollowing error occured: {self._ctx.error}",
                    stacklevel=1,
                )
            self._fbo = self._ctx.detect_framebuffer()
            self._renderer = self._make_renderer(self._ctx)
            self._initialized = True
        if not self._paused and self._index < self._end_index - 1:
            self._index += 1
            self.stepChanged.emit(self._index)
        stated = self._get_stated(self._index)
        if self._get_colors is None:
            circle_colors = None
        else:
            circle_colors = self._get_colors(self._index)
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(stated, circle_colors=circle_colors)  # type: ignore

    def exitable(self) -> bool:
        return self._end_index - 1 <= self._index

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
            self.selectionChanged.emit(selected[0].item(), self._index)

    @Slot()
    def pause(self) -> None:
        self._paused = True

    @Slot()
    def play(self) -> None:
        self._paused = False

    @Slot(int)
    def sliderChanged(self, slider_index: int) -> None:
        self._index = slider_index - self._slider_offset


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
            warnings.warn(f"Invalid value for barset: {value}", stacklevel=1)
        self.barsets[name] = barset
        self.series.append(barset)
        if "_" in name:  # Shorten name
            us_ind = name.index("_")
            barset.setLabel(f"{name[0]}_{name[us_ind + 1: us_ind + 4]}")
        return barset

    def _update_yrange(self, values: Iterable[float | list[float]]) -> None:
        values_arr = np.array(list(values))
        if self._yrange_min is None:
            yrange_min = np.min(values_arr)
        else:
            yrange_min = min(self._yrange_min, np.min(values_arr))
        self.axis_y.setRange(yrange_min, np.max(values_arr))

    @Slot(dict)
    def updateValues(self, title: str, values: dict[str, float | list[float]]) -> None:
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
                warnings.warn(f"Invalid value for barset {value}", stacklevel=1)

        for name in list(self.barsets.keys()):
            if name not in values:
                old_bs = self.barsets.pop(name)
                new_barsets.popleft().setColor(old_bs.color())
                self.series.remove(old_bs)
        self._update_yrange(values.values())
        self.chart.setTitle(title)


class CBarState(str, enum.Enum):
    ENERGY = "energy"
    N_CHILDREN = "n-children"
    FOOD_REWARD = "food-reward"
    FOOD_REWARD2 = "food-reward2"  # Poison or poor foods


class CFEnvReplayWidget(QtWidgets.QWidget):
    energyUpdated = Signal(float)
    rewardUpdated = Signal(str, dict)

    def __init__(
        self,
        xlim: int,
        ylim: int,
        env: CircleForaging,
        saved_physics: SavedPhysicsState,
        start: int = 0,
        self_terminate: bool = False,
        end: int | None = None,
        step_offset: int = 0,
        log_ds: ds.Dataset | None = None,
        profile_and_rewards: pa.Table | None = None,
        cm_fixed_minmax: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()

        timer = QTimer()
        timer.timeout.connect(self._check_exit)
        # Environment
        self._mgl_widget = MglWidget(
            timer=timer,
            env=env,
            saved_physics=saved_physics,
            figsize=(xlim * 2, ylim * 2),
            start=start,
            end=end,
            slider_offset=step_offset,
            get_colors=None if log_ds is None else self._get_colors,
        )
        self._n_max_agents = env.n_max_agents
        # cache
        self._cached_rewards = {}
        self._cached_n_children = {}
        # Log / step
        self._log_ds = log_ds
        self._log_cached = {}
        self._step_offset = step_offset
        self._start = start
        self._end = end
        # Slider
        self._slider = QtWidgets.QSlider(Qt.Horizontal)  # type: ignore
        self._slider.setSingleStep(1)
        self._slider.setMinimum(start + step_offset)
        self._slider.setMaximum(self._mgl_widget._end_index + step_offset - 1)
        self._slider.setValue(start + step_offset)
        self._slider_label = QtWidgets.QLabel(f"Step {start + step_offset}")
        # Pause/Play
        pause_button = QtWidgets.QPushButton("⏸️")
        pause_button.clicked.connect(self._mgl_widget.pause)
        play_button = QtWidgets.QPushButton("▶️")
        play_button.clicked.connect(self._mgl_widget.play)
        # Colorbar
        radiobutton_1 = QtWidgets.QRadioButton("Energy")
        radiobutton_2 = QtWidgets.QRadioButton("Num. Children")
        radiobutton_3 = QtWidgets.QRadioButton("Food Reward")
        radiobutton_4 = QtWidgets.QRadioButton("Another Food Reward")
        radiobutton_1.setChecked(True)
        radiobutton_1.toggled.connect(self.cbarEnergy)
        radiobutton_2.toggled.connect(self.cbarNChildren)
        radiobutton_3.toggled.connect(self.cbarFood)
        radiobutton_4.toggled.connect(self.cbarFood2)
        self._cbar_state = CBarState.ENERGY
        self._cbar_renderer = CBarRenderer(int(xlim * 2), int(ylim * 0.4))
        self._showing_energy = True
        self._cbar_changed = True
        self._cbar_canvas = FigureCanvasQTAgg(self._cbar_renderer._fig)
        self._value_cm = mpl.colormaps["YlOrRd"]
        self._energy_cm = mpl.colormaps["YlGnBu"]
        self._n_children_cm = mpl.colormaps["PuBuGn"]
        self._food_cm = mpl.colormaps["plasma"]
        self._norm = mc.Normalize(vmin=0.0, vmax=1.0)
        self._cm_fixed_minmax = {} if cm_fixed_minmax is None else cm_fixed_minmax
        if profile_and_rewards is not None:
            self._profile_and_rewards = profile_and_rewards
            self._reward_widget = BarChart(self._get_rewards(1))  # type: ignore
        # Layout buttons
        left_control = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(pause_button)
        buttons.addWidget(play_button)
        left_control.addLayout(buttons)
        left_control.addWidget(self._slider_label)
        left_control.addWidget(self._slider)
        cbar_selector = QtWidgets.QVBoxLayout()
        cbar_selector.addWidget(radiobutton_1)
        cbar_selector.addWidget(radiobutton_2)
        cbar_selector.addWidget(radiobutton_3)
        cbar_selector.addWidget(radiobutton_4)
        control = QtWidgets.QHBoxLayout()
        control.addLayout(left_control)
        control.addLayout(cbar_selector)
        # Total layout
        total_layout = QtWidgets.QVBoxLayout()
        total_layout.addLayout(control)
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
        self._mgl_widget.stepChanged.connect(self.updateStep)
        self._slider.sliderMoved.connect(self._mgl_widget.sliderChanged)
        self._slider.sliderMoved.connect(self.updateSliderLabel)
        if profile_and_rewards is not None:
            self.rewardUpdated.connect(self._reward_widget.updateValues)
        # Initial size
        if profile_and_rewards is None:
            self.resize(xlim * 3, ylim * 3)
        else:
            self.resize(xlim * 4, ylim * 3)
        self._self_terminate = self_terminate

    def _check_exit(self) -> None:
        if self._mgl_widget.exitable() and self._self_terminate:
            print("Safely exited app because it reached the final frame")
            self.close()

    def _get_rewards(self, unique_id: int) -> dict[str, float]:
        if unique_id in self._cached_rewards:
            return self._cached_rewards[unique_id]
        filtered = self._profile_and_rewards.filter(pc.field("unique_id") == unique_id)
        rd = filtered.drop(["birthtime", "parent", "unique_id"]).to_pydict()
        rd = {k: v[0] for k, v in rd.items()}
        self._cached_rewards[unique_id] = rd
        return rd

    def _get_n_children(self, unique_id: int) -> int:
        if self._profile_and_rewards is None:
            warnings.warn(
                "N children requires profile_an_rewards.parquet",
                stacklevel=1,
            )
            return 0
        if unique_id == 0:
            return 0
        if unique_id in self._cached_n_children:
            return self._cached_n_children[unique_id]
        nc = len(self._profile_and_rewards.filter(pc.field("parent") == unique_id))
        self._cached_n_children[unique_id] = nc
        return nc

    def _get_log(self, step: int) -> dict[str, NDArray]:
        assert self._log_ds is not None
        log_key = step // N_MAX_SCAN
        if log_key not in self._log_cached:
            log_key = step // N_MAX_SCAN
            scanner = self._log_ds.scanner(
                columns=["energy", "step", "slots", "unique_id"],
                filter=(
                    (step <= pc.field("step")) & (pc.field("step") <= step + N_MAX_SCAN)
                ),
            )
            table = scanner.to_table()
            if len(self._log_cached) > N_MAX_CACHED_LOG:
                self._log_cached.clear()
            self._log_cached[log_key] = [
                table.filter(pc.field("step") == i).to_pydict()
                for i in reversed(range(step, step + N_MAX_SCAN))
            ]
        return self._log_cached[log_key][step % N_MAX_SCAN]

    def _get_colors(self, step_index: int) -> NDArray:
        assert self._log_ds is not None
        log = self._get_log(self._step_offset + step_index)
        slots = np.array(log["slots"])
        if self._cbar_state is CBarState.ENERGY:
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
        elif self._cbar_state is CBarState.FOOD_REWARD:
            title = "Food Reward"
            cm = self._food_cm
            value = np.zeros(self._n_max_agents)
            for slot, uid in zip(log["slots"], log["unique_id"]):
                rew = self._get_rewards(uid)
                if "scale_food" in rew:
                    rew_food = rew["w_food"] * (10 ** rew["scale_food"])
                elif "food" in rew:
                    rew_food = rew["food"]
                elif "food_1" in rew:
                    rew_food = rew["food_1"]
                else:
                    warnings.warn("Unsupported reward", stacklevel=1)
                    rew_food = 0.0
                value[slot] = rew_food
        elif self._cbar_state is CBarState.FOOD_REWARD2:
            title = "Food Reward"
            cm = self._food_cm
            value = np.zeros(self._n_max_agents)
            for slot, uid in zip(log["slots"], log["unique_id"]):
                rew = self._get_rewards(uid)
                if "food_2" in rew:
                    rew_food = rew["food_2"]
                else:
                    warnings.warn("Unsupported reward", stacklevel=1)
                    rew_food = 0.0
                value[slot] = rew_food
        else:
            warnings.warn(f"Invalid cbar state {self._cbar_state}", stacklevel=1)
            return np.zeros((self._n_max_agents, 4))
        if self._cbar_state.value in self._cm_fixed_minmax:
            self._norm.vmin, self._norm.vmax = self._cm_fixed_minmax[
                self._cbar_state.value
            ]
        else:
            self._norm.vmin = float(np.amin(value))
            self._norm.vmax = float(np.amax(value))
        if self._cbar_changed:
            self._cbar_renderer.render(self._norm, cm, title)
            self._cbar_changed = False
            self._cbar_canvas.draw()
        return cm(self._norm(value))

    @Slot(int)
    def updateStep(self, step_index: int) -> None:
        step = self._step_offset + step_index
        self._slider.setValue(step)
        self._slider_label.setText(f"Step {step}")

    @Slot(int)
    def updateSliderLabel(self, slider_value: int) -> None:
        self._slider_label.setText(f"Step {slider_value}")

    @Slot(int, int)
    def updateRewards(self, selected_slot: int, step_index: int) -> None:
        if self._profile_and_rewards is None or selected_slot == -1:
            return

        log = self._get_log(self._step_offset + step_index)
        for slot, uid in zip(log["slots"], log["unique_id"]):
            if slot == selected_slot:
                self.rewardUpdated.emit(
                    f"Reward function of {uid}",
                    self._get_rewards(uid),
                )
                return

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

    @Slot(bool)
    def cbarFood(self, checked: bool) -> None:
        if checked:
            self._cbar_state = CBarState.FOOD_REWARD
            self._cbar_changed = True

    @Slot(bool)
    def cbarFood2(self, checked: bool) -> None:
        if checked:
            self._cbar_state = CBarState.FOOD_REWARD2
            self._cbar_changed = True


def start_widget(widget_cls: type[QtWidgets.QWidget], **kwargs) -> None:
    app = QtWidgets.QApplication([])
    widget = widget_cls(**kwargs)
    widget.show()
    sys.exit(app.exec())
