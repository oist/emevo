from __future__ import annotations

from os import PathLike
from typing import Any, Protocol, TypeVar

from numpy.typing import NDArray

ENV = TypeVar("ENV", contravariant=True)


class Visualizer(Protocol[ENV]):
    pix_fmt: str

    def close(self) -> None:
        """Close this visualizer"""

    def get_image(self) -> NDArray:
        ...

    def render(self, env: ENV) -> Any:
        """Render image"""
        ...

    def show(self) -> None:
        """Open a GUI window"""


class SaveVideoWrapper(Visualizer[ENV]):
    def __init__(
        self,
        visualizer: Visualizer[ENV],
        filename: PathLike,
        **kwargs,
    ) -> None:
        self.unwrapped = visualizer
        self.pix_fmt = self.unwrapped.pix_fmt
        self._path = filename
        self._writer = None
        self._iio_kwargs = kwargs

    def close(self) -> None:
        self.unwrapped.close()
        if self._writer is not None:
            self._writer.close()

    def get_image(self) -> NDArray:
        return self.unwrapped.get_image()

    def render(self, env: ENV) -> Any:
        ret = self.unwrapped.render(env)
        image = self.unwrapped.get_image()
        if self._writer is None:
            from imageio_ffmpeg import write_frames

            self._writer = write_frames(
                self._path,
                image.shape[:2],
                pix_fmt_in="rgba",
                **self._iio_kwargs,
            )
            self._writer.send(None)  # seed the generator
        self._writer.send(image)
        return ret

    def show(self) -> None:
        self.unwrapped.show()
