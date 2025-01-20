from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import moderngl
import numpy as np
from numpy.typing import NDArray
from phyjax2d import Circle, State, StateDict, Vec2d
from PySide6 import QtWidgets
from PySide6.QtCore import QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColorSpace, QGuiApplication, QMouseEvent, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from emevo.environments.circle_foraging import CircleForaging
from emevo.environments.moderngl_vis import MglRenderer
from emevo.exp_utils import SavedPhysicsState


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setVersion(4, 1)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setColorSpace(QColorSpace.SRgb)
    return fmt


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
        self._index = start
        self._dragged_state = None
        self._make_renderer = partial(
            MglRenderer,
            screen_width=self._figsize[0],
            screen_height=self._figsize[1],
            x_range=x_range,
            y_range=y_range,
            space=env._physics,
            stated=self._get_stated(),
            sc_color_opt=env._food_color,
            sensor_color=np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32),
            sensor_fn=self._sensor_fn,
        )
        self._env = env
        self._get_colors = get_colors
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
        self._ctx, self._fbo = None, None
        # For dragging
        self._last_mouse_pos = None
        self._dragging_agent = False
        self._dragging_food = False
        self._xy_max = jnp.expand_dims(jnp.array([x_range, y_range]), axis=0)
        self._selected_slot = 0
        self._selected_food_slot = 0

    def _sensor_fn(self, stated: StateDict) -> tuple[jax.Array, jax.Array]:
        return self._env._get_selected_sensor(stated, self._selected_slot)

    def _scale_position(self, position: QPointF) -> tuple[float, float]:
        return (
            position.x() * self._scaling[0],
            (self._figsize[1] - position.y()) * self._scaling[1],
        )

    def _get_stated(self) -> StateDict:
        if self._dragged_state is not None:
            return self._dragged_state
        else:
            return self._phys_state.set_by_index(self._index, self._env_state.physics)

    def _set_default_viewport(self) -> None:
        self._ctx.viewport = 0, 0, *self._figsize
        self._fbo.viewport = 0, 0, *self._figsize

    def paintGL(self) -> None:
        if not self._initialized:
            if QGuiApplication.platformName() == "eglfs":
                self._ctx = moderngl.create_context(
                    require=410,
                    share=True,
                    backend="egl",  # type: ignore
                )
            else:
                self._ctx = moderngl.create_context(require=410)
            if self._ctx.error != "GL_NO_ERROR":
                warnings.warn(
                    f"The following error occured: {self._ctx.error}",
                    stacklevel=1,
                )
            self._fbo = self._ctx.detect_framebuffer()
            self._renderer = self._make_renderer(self._ctx)
            self._initialized = True
        if not self._paused and self._index < self._end_index - 1:
            self._index += 1
            self.stepChanged.emit(self._index)
        stated = self._get_stated()
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
        if evt.button() != Qt.LeftButton:
            return
        position = self._scale_position(evt.position())
        sd = self._get_stated()
        posarray = jnp.array(position)

        def _get_selected(state: State, shape: Circle) -> int | None:
            overlap = _overlap(posarray, shape, state)
            (selected,) = jnp.nonzero(overlap)
            if 0 < selected.shape[0]:
                return selected[0].item()
            else:
                return None

        selected = _get_selected(
            sd.circle,
            self._env._physics.shaped.circle,
        )
        if selected is not None:
            self._selected_slot = selected
            self.selectionChanged.emit(self._selected_slot, self._index)

            # Initialize dragging
            if self._paused:
                self._last_mouse_pos = Vec2d(*position)
                self._dragging_agent = True

        selected = _get_selected(
            sd.static_circle,
            self._env._physics.shaped.static_circle,
        )
        if selected is not None and self._paused:
            self._selected_food_slot = selected
            self._last_mouse_pos = Vec2d(*position)
            self._dragging_food = True

    def mouseReleaseEvent(self, evt: QMouseEvent) -> None:
        if evt.button() == Qt.LeftButton:
            self._last_mouse_pos = None
            self._dragging_food = False
            self._dragging_agent = False

    def mouseMoveEvent(self, evt: QMouseEvent) -> None:
        current_pos = Vec2d(*self._scale_position(evt.position()))

        dragging = self._dragging_agent or self._dragging_food
        if self._last_mouse_pos is not None and dragging:
            # Compute dx/dy
            dxy = current_pos - self._last_mouse_pos

            # Update the physics state
            stated = self._get_stated()
            if self._dragging_agent:
                circle = stated.circle
                xy = jnp.clip(
                    circle.p.xy.at[self._selected_slot].add(jnp.array(dxy)),
                    min=self._env._agent_radius,
                    max=self._xy_max - self._env._agent_radius,
                )
                self._dragged_state = stated.nested_replace("circle.p.xy", xy)
            elif self._dragging_food:
                static_circle = stated.static_circle
                xy = jnp.clip(
                    static_circle.p.xy.at[self._selected_food_slot].add(jnp.array(dxy)),
                    min=self._env._food_radius,
                    max=self._xy_max - self._env._food_radius,
                )
                self._dragged_state = stated.nested_replace("static_circle.p.xy", xy)

            self._last_mouse_pos = current_pos
            self.update()

    @Slot()
    def pause(self) -> None:
        self._paused = True

    @Slot()
    def play(self) -> None:
        self._paused = False
        self._dragged_state = None

    @Slot(int)
    def sliderChanged(self, slider_index: int) -> None:
        self._index = slider_index - self._slider_offset
