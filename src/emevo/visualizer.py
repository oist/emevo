from __future__ import annotations

from os import PathLike
from typing import Any, Protocol, TypeVar

from numpy.typing import NDArray

STATE = TypeVar("STATE", contravariant=True)


class Visualizer(Protocol[STATE]):
    def close(self) -> None:
        """Close this visualizer"""
        ...

    def get_image(self) -> NDArray:
        ...

    def render(self, state: STATE) -> Any:
        """Render image"""
        ...

    def show(self, *args, **kwargs) -> None:
        """Open a GUI window"""
        ...

    def overlay(self, name: str, _value: Any) -> Any:
        """Render additional value as an overlay"""
        raise ValueError(f"Unsupported overlay: {name}")


class VisWrapper(Visualizer[STATE], Protocol):
    unwrapped: Visualizer[STATE]

    def close(self) -> None:
        self.unwrapped.close()

    def get_image(self) -> NDArray:
        return self.unwrapped.get_image()

    def render(self, state: STATE) -> Any:
        return self.unwrapped.render(state)

    def show(self, *args, **kwargs) -> None:
        self.unwrapped.show()

    def overlay(self, name: str, value: Any) -> Any:
        return self.unwrapped.overlay(name, value)


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

    def show(self, *args, **kwargs) -> None:
        del args, kwargs
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
