from __future__ import annotations

from os import PathLike
from typing import Any, Callable, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

ENV = TypeVar("ENV", contravariant=True)


class Visualizer(Protocol[ENV]):
    def close(self) -> None:
        """Close this visualizer"""
        ...

    def get_image(self) -> NDArray:
        ...

    def render(self, env: ENV) -> Any:
        """Render image"""
        ...

    def show(self) -> None:
        """Open a GUI window"""
        ...

    def overlay(self, name: str, _value: Any) -> Any:
        """Render additional value as an overlay"""
        raise ValueError(f"Unsupported overlay: {name}")


class VisWrapper(Visualizer[ENV], Protocol):
    unwrapped: Visualizer[ENV]

    def close(self) -> None:
        self.unwrapped.close()

    def get_image(self) -> NDArray:
        return self.unwrapped.get_image()

    def render(self, env: ENV) -> Any:
        return self.unwrapped.render(env)

    def show(self) -> None:
        self.unwrapped.show()

    def overlay(self, name: str, value: Any) -> Any:
        return self.unwrapped.overlay(name, value)


class ImageStackWrapper(VisWrapper[ENV]):
    def __init__(
        self,
        visualizer: Visualizer[ENV],
        image_source: Callable[[], NDArray],
        vstack: bool = False,
        **kwargs,
    ) -> None:
        self.unwrapped = visualizer
        self._image_source = image_source
        self._vstack = vstack

    def get_image(self) -> NDArray:
        orig_image = self.unwrapped.get_image()
        additional_image = self._image_source()
        if self._vstack:
            new_image = np.append(orig_image, additional_image, axis=0)
        else:
            new_image = np.append(orig_image, additional_image, axis=1)
        return new_image


class SaveVideoWrapper(VisWrapper[ENV]):
    def __init__(
        self,
        visualizer: Visualizer[ENV],
        filename: PathLike,
        **kwargs,
    ) -> None:
        self.unwrapped = visualizer
        self._path = filename
        self._writer = None
        self._iio_kwargs = kwargs

    def close(self) -> None:
        self.unwrapped.close()
        if self._writer is not None:
            self._writer.close()

    def show(self) -> None:
        image = self.unwrapped.get_image()
        if self._writer is None:
            from imageio_ffmpeg import write_frames

            self._writer = write_frames(
                self._path,
                image.shape[:2],
                pix_fmt_in="rgb" if image.shape[2] == 3 else "rgba",
                **self._iio_kwargs,
            )
            self._writer.send(None)  # seed the generator
        self._writer.send(image)
        self.unwrapped.show()
