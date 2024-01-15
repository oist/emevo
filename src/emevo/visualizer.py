from __future__ import annotations

from os import PathLike
from typing import Protocol, TypeVar

from numpy.typing import NDArray

STATE = TypeVar("STATE", contravariant=True)


class Visualizer(Protocol[STATE]):
    def close(self) -> None:
        """Close this visualizer"""
        ...

    def get_image(self) -> NDArray:
        ...

    def render(self, state: STATE) -> None:
        """Render image"""
        ...

    def show(self) -> None:
        """Open a GUI window"""
        ...


class VisWrapper(Visualizer[STATE], Protocol):
    unwrapped: Visualizer[STATE]

    def close(self) -> None:
        self.unwrapped.close()

    def get_image(self) -> NDArray:
        return self.unwrapped.get_image()

    def render(self, state: STATE) -> None:
        self.unwrapped.render(state)

    def show(self) -> None:
        self.unwrapped.show()


class SaveVideoWrapper(VisWrapper[STATE]):
    def __init__(
        self,
        visualizer: Visualizer[STATE],
        filename: PathLike,
        **kwargs,
    ) -> None:
        self.unwrapped = visualizer
        self._path = filename
        self._writer = None
        self._iio_kwargs = kwargs
        self._count = 0

    def close(self) -> None:
        self.unwrapped.close()
        if self._writer is not None:
            self._writer.close()

    def show(self) -> None:
        self._count += 1
        image = self.unwrapped.get_image()
        if self._writer is None:
            h, w = image.shape[:2]
            from imageio_ffmpeg import write_frames

            self._writer = write_frames(
                self._path,
                (w, h),
                pix_fmt_in="rgb24" if image.shape[2] == 3 else "rgba",
                **self._iio_kwargs,
            )
            self._writer.send(None)  # seed the generator
        self._writer.send(image.tobytes())  # seed the generator
        self.unwrapped.show()
