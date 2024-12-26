from __future__ import annotations

from os import PathLike
from typing import Protocol, TypeVar

from numpy.typing import NDArray

STATE = TypeVar("STATE", contravariant=True)


class Visualizer(Protocol[STATE]):
    def close(self) -> None:
        """Close this visualizer"""
        ...

    def get_image(self) -> NDArray: ...

    def render(self, state: STATE, **kwargs) -> None:
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

    def render(self, state: STATE, **kwargs) -> None:
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
        self._container = None
        self._stream = None
        self._pyav_kwargs = kwargs
        self._count = 0

    def close(self) -> None:
        self.unwrapped.close()
        if self._container is not None:
            self._container.close()

    def show(self) -> None:
        import av

        self.unwrapped.show()
        self._count += 1
        image = self.unwrapped.get_image()
        if self._container is None:
            codec = self._pyav_kwargs.pop("codec", "h264")
            self._container = av.open(self._path, mode="w")
            self._stream = self._container.add_stream(codec, **self._pyav_kwargs)
            self._stream.bit_rate = 8000000
        # Encode frame
        frame = av.VideoFrame.from_ndarray(image[:, :, :3], format="rgb24")
        packet = self._stream.encode(frame)
        self._container.mux(packet)
        self.unwrapped.show()
