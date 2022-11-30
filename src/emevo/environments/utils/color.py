from typing import NamedTuple


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int = 255

    @staticmethod
    def from_float(r: float, g: float, b: float, a: float = 1.0) -> "Color":
        return Color(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
