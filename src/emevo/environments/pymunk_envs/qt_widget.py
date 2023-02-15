"""Qt widget with moderngl visualizer for advanced visualization.
"""


import dataclasses
from functools import partial
from typing import Any, Callable

import moderngl
import pymunk
from pymunk.vec2d import Vec2d
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QGuiApplication, QMouseEvent, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

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
    point: Vec2d = dataclasses.field(default_factory=Vec2d.zero)

    def start_drag(self, point: Vec2d, shape: pymunk.Shape) -> None:
        shape.color = shape.color._replace(a=100)  # type: ignore
        self.shape = shape
        self.body = shape.body
        self.point = point

    def dragging(self, point: Vec2d) -> None:
        if self.body is not None:
            delta = point - self.point
            self.point = point
            self.body.position = self.body.position + delta
            if self.body.space is not None:
                self.body.space.reindex_shapes_for_body(self.body)

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
    changed: bool = False
    pantool: PanTool = dataclasses.field(default_factory=PanTool)
    paused: bool = False
    paused_before: bool = False


def _do_nothing(_env: PymunkEnv, _app_state: AppState) -> None:
    pass


class PymunkMglWidget(QOpenGLWidget):
    def __init__(
        self,
        *,
        env: PymunkEnv,
        timer: QTimer,
        app_state: AppState | None = None,
        figsize: tuple[float, float] | None = None,
        step_fn: Callable[[PymunkEnv, AppState], None] = _do_nothing,
        overlay_fn: Callable[
            [PymunkEnv, AppState],
            tuple[str, Any] | None,
        ] = _do_nothing,
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
        self._overlay_fn = overlay_fn
        self._env = env
        self._state = AppState() if app_state is None else app_state
        self._initialized = False
        self._timer = timer
        self._timer.timeout.connect(self.update)  # type: ignore

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
        self._step_fn(self._env, self._state)
        overlay = self._overlay_fn(self._env, self._state)
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(self._env)  # type: ignore
        if overlay is not None:
            self._renderer.overlay(*overlay)
        self._state.changed = False

    def _scale_position(self, position: QPointF) -> Vec2d:
        return Vec2d(
            position.x() * self._scaling[0],
            (self._figsize[1] - position.y()) * self._scaling[1],
        )

    def mousePressEvent(self, evt: QMouseEvent) -> None:
        position = self._scale_position(evt.position())
        query = self._env.get_space().point_query(
            position,
            0.0,
            shape_filter=make_filter(CollisionType.AGENT, CollisionType.FOOD),
        )
        if len(query) == 1:
            self._state.pantool.start_drag(position, query[0].shape)  # type: ignore
            self._paused_before = self._state.paused
            self._state.paused = True
            self._timer.stop()
            self.update()

    def mouseMoveEvent(self, evt: QMouseEvent) -> None:
        self._state.pantool.dragging(self._scale_position(evt.position()))
        self.update()

    def mouseReleaseEvent(self, evt: QMouseEvent) -> None:
        if self._state.pantool.is_dragging:
            self._state.pantool.stop_drag(self._scale_position(evt.position()))
            self._state.paused = self._state.paused_before
            self._timer.start()
            self.update()
