from __future__ import annotations

from os import PathLike
from typing import Any, Callable, Iterable, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

ENV = TypeVar("ENV", contravariant=True)


class Visualizer(Protocol[ENV]):
    pix_fmt: str

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


class VisWrapper(Visualizer[ENV], Protocol):
    unwrapped: Visualizer[ENV]

    def close(self) -> None:
        self.unwrapped.close()

    def get_image(self) -> NDArray:
        return self.unwrapped.get_image()

    def render(self, env: ENV) -> Any:
        return self.unwrapped.render(env)

    def show(self) -> None:
        return self.unwrapped.show()


ImageSource = Callable[[], NDArray]


class ImageComposeWrapper(VisWrapper[ENV]):
    def __init__(
        self,
        visualizer: Visualizer[ENV],
        image_sources: Iterable[ImageSource],
        compose_rules: Iterable[Literal["h", "v"]],
        **kwargs,
    ) -> None:
        self.unwrapped = visualizer
        self.pix_fmt = self.unwrapped.pix_fmt
        self._image_sources = image_sources
        self._compose_rules = compose_rules

    def get_image(self) -> NDArray:
        images = [self.unwrapped.get_image()]
        orig_image = images[0]
        total_w, total_h = orig_image.shape[:2]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for rule, source in zip(self._compose_rules, self._image_sources):
            image = source()
            images.append(image)
            w, h = image.shape[:2]
            if rule == "h":
                offsets.append((0, total_h))
                total_w = max(w, total_w)
                total_h += h
            elif rule == "v":
                offsets.append((total_w, 0))
                total_w += w
                total_h = max(h, total_h)
            else:
                raise ValueError(f"Invalid image composition rule: {rule}")
        new_image = np.zeros(
            (total_w, total_h, *orig_image.shape[2:]),
            dtype=orig_image.dtype,
        )
        for (w_offset, h_offset), image in zip(offsets, images):
            w, h = image.shape[:2]
            new_image[w_offset : w_offset + w, h_offset : h_offset + h] = image
        return new_image


class SaveVideoWrapper(VisWrapper[ENV]):
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
