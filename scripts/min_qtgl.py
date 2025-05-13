"""To debug Qt OpenGL app?"""

from __future__ import annotations

import sys
import warnings

import moderngl
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget


def _mgl_qsurface_fmt() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(4, 1)
    return fmt


class MglWidget(QOpenGLWidget):
    selectionChanged = Signal(int, int)
    stepChanged = Signal(int)

    def __init__(
        self,
        *,
        timer: QTimer,
        figsize: tuple[float, float],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # Set default format
        QSurfaceFormat.setDefaultFormat(_mgl_qsurface_fmt())
        super().__init__(parent)
        self._timer = timer
        self._timer.timeout.connect(self.update)
        self._figsize = figsize

        self.setFixedSize(*self._figsize)
        self.setMouseTracking(True)
        self._ctx, self._fbo = None, None

    def paintGL(self) -> None:
        if self._ctx is None:
            self._ctx = moderngl.create_context()
            if self._ctx.error != "GL_NO_ERROR":
                warnings.warn(
                    f"The following error occured: {self._ctx.error}",
                    stacklevel=1,
                )
            self._fbo = self._ctx.detect_framebuffer()

        self._fbo.use()
        self._ctx.clear(1.0, 1.0, 1.0)


def main() -> None:
    app = QtWidgets.QApplication([])
    timer = QTimer()
    widget = MglWidget(timer=timer, figsize=(600, 600))
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
