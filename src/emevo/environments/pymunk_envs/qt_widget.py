from functools import partial
from typing import Callable

import moderngl
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from emevo.environments.pymunk_envs.moderngl_vis import MglRenderer
from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv


def _do_nothing(_env: PymunkEnv) -> None:
    pass


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    return fmt


class PymunkMglWidget(QOpenGLWidget):
    def __init__(
        self,
        env: PymunkEnv,
        figsize: tuple[float, float] | None = None,
        step_fn: Callable[[PymunkEnv], None] = _do_nothing,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # init renderer
        xlim, ylim = env.get_coordinate().bbox()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        self._figsize = int(figsize[0]), int(figsize[1])
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
        self._paused = False
        self._initialized = False

        self.setFixedSize(*self._figsize)
        self.setMouseTracking(True)

    def _set_default_viewport(self) -> None:
        self._ctx.viewport = 0, 0, *self._figsize
        self._fbo.viewport = 0, 0, *self._figsize

    def paintGL(self) -> None:
        if not self._initialized:
            self.context().setFormat(_mgl_qsurface_fmt())
            self._ctx = moderngl.create_context(require=410, share=True, backend="egl")
            if self._ctx.error != "GL_NO_ERROR":
                raise RuntimeError(f"The following error occured: {self._ctx.error}")
            self._fbo = self._ctx.detect_framebuffer()
            self._renderer = self._make_renderer(self._ctx)
            self._initialized = True
        self.render()

    def render(self) -> None:
        self._step_fn(self._env)
        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)
        self._renderer.render(self._env)
