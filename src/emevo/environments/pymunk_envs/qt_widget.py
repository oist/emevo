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


class PymunkMglWidget(QOpenGLWidget):
    def __init__(
        self,
        env: PymunkEnv,
        figsize: tuple[float, float] | None = None,
        step_fn: Callable[[PymunkEnv], None] = _do_nothing,
        voffsets: tuple[int, ...] = (),
        hoffsets: tuple[int, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        # Init widget
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)
        # init renderer
        xlim, ylim = env.get_coordinate().bbox()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        w, h = int(figsize[0]), int(figsize[1])
        self._figsize = w + int(sum(hoffsets)), h + int(sum(voffsets))
        self._make_renderer = partial(
            MglRenderer,
            screen_width=w,
            screen_height=h,
            x_range=x_range,
            y_range=y_range,
            env=env,
            voffsets=voffsets,
            hoffsets=hoffsets,
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
            self._ctx = moderngl.create_context(require=410)
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
