from __future__ import annotations

from typing import Any, Iterable, Protocol, TypeVar

from numpy.typing import NDArray


class Image(Protocol):
    def __array__(self) -> NDArray:
        ...


ENV = TypeVar("ENV", contravariant=True)
IMAGE = TypeVar("IMAGE", covariant=True, bound=Image)


class Visualizer(Protocol[ENV, IMAGE]):
    def close(self) -> None:
        """Close this visualizer"""

    def get_image(self) -> IMAGE:
        ...

    def render(self, env: ENV) -> Any:
        """Render image"""
        ...

    def show(self) -> None:
        """Open a GUI window"""


def save_video(filename: str, size: tuple[int, int], frames: Iterable[Image]) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(filename, size)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()


def save_gif(filename: str, size: tuple[int, int], frames: Iterable[Image]) -> None:
    import imageio

    writer = imageio_ffmpeg.write_frames(filename, size)
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()
