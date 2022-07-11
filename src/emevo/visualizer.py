from typing import Any, Protocol, TypeVar

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
