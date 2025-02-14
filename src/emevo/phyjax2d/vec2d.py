# ----------------------------------------------------------------------------
# pymunk
# Copyright (c) 2007-2020 Victor Blomqvist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""This module contain the Vec2d class that is used in all of pymunk when a
vector is needed.

The Vec2d class is used almost everywhere in pymunk for 2d coordinates and
vectors, for example to define gravity vector in a space. However, pymunk is
smart enough to convert tuples or tuple like objects to Vec2ds so you usually
do not need to explicitly do conversions if you happen to have a tuple::

    >>> import pymunk
    >>> space = pymunk.Space()
    >>> space.gravity
    Vec2d(0.0, 0.0)
    >>> space.gravity = 3,5
    >>> space.gravity
    Vec2d(3.0, 5.0)
    >>> space.gravity += 2,6
    >>> space.gravity
    Vec2d(5.0, 11.0)

More examples::

    >>> from pymunk.vec2d import Vec2d
    >>> Vec2d(7.3, 4.2)
    Vec2d(7.3, 4.2)
    >>> Vec2d(7.3, 4.2) + Vec2d(1, 2)
    Vec2d(8.3, 6.2)

"""
from __future__ import annotations

__docformat__ = "reStructuredText"

import math
import numbers
import operator
from typing import Any, NamedTuple

__all__ = ["Vec2d"]

Self = Any


class Vec2d(NamedTuple):
    """2d vector class, supports vector and scalar operators, and also
    provides some high level functions.
    """

    x: float
    y: float

    # String representaion (for debugging)
    def __repr__(self) -> str:
        return f"Vec2d({self.x}, {self.y})"

    # Addition
    def __add__(self, other: tuple[float, float]) -> Self:  # type: ignore
        """Add a Vec2d with another Vec2d or tuple of size 2

        >>> Vec2d(3,4) + Vec2d(1,2)
        Vec2d(4, 6)
        >>> Vec2d(3,4) + (1,2)
        Vec2d(4, 6)
        """
        assert (
            len(other) == 2
        ), f"{other} not supported. Only Vec2d and Sequence of length 2 is supported."

        return Vec2d(self.x + other[0], self.y + other[1])

    def __radd__(self, other: tuple[float, float]) -> Self:
        """Add a tuple of size 2 with a Vec2d

        >>> (1,2) + Vec2d(3,4)
        Vec2d(4, 6)
        """
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other: tuple[float, float]) -> Self:
        """Subtract a Vec2d with another Vec2d or tuple of size 2

        >>> Vec2d(3,4) - Vec2d(1,2)
        Vec2d(2, 2)
        >>> Vec2d(3,4) - (1,2)
        Vec2d(2, 2)
        """
        return Vec2d(self.x - other[0], self.y - other[1])

    def __rsub__(self, other: tuple[float, float]) -> Self:
        """Subtract a tuple of size 2 with a Vec2d

        >>> (1,2) - Vec2d(3,4)
        Vec2d(-2, -2)
        """
        assert (
            len(other) == 2
        ), f"{other} not supported. Only Vec2d and Sequence of length 2 is supported."
        return Vec2d(other[0] - self.x, other[1] - self.y)

    # Multiplication
    def __mul__(self, other: float) -> Self:  # type: ignore[override]
        """Multiply with a float

        >>> Vec2d(3,6) * 2.5
        Vec2d(7.5, 15.0)
        """
        assert isinstance(other, numbers.Real)
        return Vec2d(self.x * other, self.y * other)

    def __rmul__(self, other: float) -> Self:  # type: ignore[override]
        """Multiply a float with a Vec2d

        >>> 2.5 * Vec2d(3,6)
        Vec2d(7.5, 15.0)
        """
        return self.__mul__(other)

    # Division
    def __floordiv__(self, other: float) -> Self:
        """Floor division by a float (also known as integer division)

        >>> Vec2d(3,6) // 2.0
        Vec2d(1.0, 3.0)
        """
        assert isinstance(other, numbers.Real)
        return Vec2d(self.x // other, self.y // other)

    def __truediv__(self, other: float) -> Self:
        """Division by a float

        >>> Vec2d(3,6) / 2.0
        Vec2d(1.5, 3.0)
        """
        assert isinstance(other, numbers.Real)
        return Vec2d(self.x / other, self.y / other)

    # Unary operations
    def __neg__(self) -> Self:
        """Return the negated version of the Vec2d

        >>> -Vec2d(1,-2)
        Vec2d(-1, 2)
        """
        return Vec2d(operator.neg(self.x), operator.neg(self.y))

    def __pos__(self) -> Self:
        """Return the unary pos of the Vec2d.

        >>> +Vec2d(1,-2)
        Vec2d(1, -2)
        """
        return Vec2d(operator.pos(self.x), operator.pos(self.y))

    def __abs__(self) -> float:
        """Return the length of the Vec2d

        >>> abs(Vec2d(3,4))
        5.0
        """
        return self.length

    # vectory functions
    def get_length_sqrd(self) -> float:
        """Get the squared length of the vector.
        If the squared length is enough it is more efficient to use this method
        instead of first calling get_length() or access .length and then do a
        x**2.

        >>> v = Vec2d(3,4)
        >>> v.get_length_sqrd() == v.length**2
        True

        :return: The squared length
        """
        return self.x**2 + self.y**2

    @property
    def length(self) -> float:
        """Get the length of the vector.

        >>> Vec2d(10, 0).length
        10.0
        >>> '%.2f' % Vec2d(10, 20).length
        '22.36'

        :return: The length
        """
        return math.sqrt(self.x**2 + self.y**2)

    def scale_to_length(self, length: float) -> Self:
        """Return a copy of this vector scaled to the given length.

        >>> '%.2f, %.2f' % Vec2d(10, 20).scale_to_length(20)
        '8.94, 17.89'
        """
        old_length = self.length
        return Vec2d(self.x * length / old_length, self.y * length / old_length)

    def rotated(self, angle_radians: float) -> Self:
        """Create and return a new vector by rotating this vector by
        angle_radians radians.

        :return: Rotated vector
        """
        cos = math.cos(angle_radians)
        sin = math.sin(angle_radians)
        x = self.x * cos - self.y * sin
        y = self.x * sin + self.y * cos
        return Vec2d(x, y)

    def rotated_degrees(self, angle_degrees: float) -> Self:
        """Create and return a new vector by rotating this vector by
        angle_degrees degrees.

        :return: Rotade vector
        """
        return self.rotated(math.radians(angle_degrees))

    @property
    def angle(self) -> float:
        """The angle (in radians) of the vector"""
        if self.get_length_sqrd() == 0:
            return 0
        return math.atan2(self.y, self.x)

    @property
    def angle_degrees(self) -> float:
        """Gets the angle (in degrees) of a vector"""
        return math.degrees(self.angle)

    def get_angle_between(self, other: tuple[float, float]) -> float:
        """Get the angle between the vector and the other in radians

        :return: The angle
        """
        assert len(other) == 2
        cross = self.x * other[1] - self.y * other[0]
        dot = self.x * other[0] + self.y * other[1]
        return math.atan2(cross, dot)

    def get_angle_degrees_between(self, other: Self) -> float:
        """Get the angle between the vector and the other in degrees

        :return: The angle (in degrees)
        """
        return math.degrees(self.get_angle_between(other))

    def normalized(self) -> Self:
        """Get a normalized copy of the vector
        Note: This function will return 0 if the length of the vector is 0.

        :return: A normalized vector
        """
        length = self.length
        if length != 0:
            return self / length
        return Vec2d(0, 0)

    def normalized_and_length(self) -> tuple[Self, float]:
        """Normalize the vector and return its length before the normalization

        :return: The length before the normalization
        """
        length = self.length
        if length != 0:
            return self / length, length
        return Vec2d(0, 0), 0

    def perpendicular_left(self) -> Self:
        return Vec2d(-self.y, self.x)

    def perpendicular_right(self) -> Self:
        return Vec2d(self.y, -self.x)

    def dot(self, other: tuple[float, float]) -> float:
        """The dot product between the vector and other vector
            v1.dot(v2) -> v1.x*v2.x + v1.y*v2.y

        :return: The dot product
        """
        assert len(other) == 2
        return float(self.x * other[0] + self.y * other[1])

    def get_distance(self, other: tuple[float, float]) -> float:
        """The distance between the vector and other vector

        :return: The distance
        """
        assert len(other) == 2
        return math.sqrt((self.x - other[0]) ** 2 + (self.y - other[1]) ** 2)

    def get_dist_sqrd(self, other: tuple[float, float]) -> float:
        """The squared distance between the vector and other vector
        It is more efficent to use this method than to call get_distance()
        first and then do a sqrt() on the result.

        :return: The squared distance
        """
        assert len(other) == 2
        return (self.x - other[0]) ** 2 + (self.y - other[1]) ** 2

    def projection(self, other: tuple[float, float]) -> Self:
        """Project this vector on top of other vector"""
        assert len(other) == 2
        other_length_sqrd = other[0] * other[0] + other[1] * other[1]
        if other_length_sqrd == 0.0:
            return Vec2d(0, 0)
        projected_length_times_other_length = self.dot(other)
        new_length = projected_length_times_other_length / other_length_sqrd
        return Vec2d(other[0] * new_length, other[1] * new_length)

    def cross(self, other: tuple[float, float]) -> float:
        """The cross product between the vector and other vector
            v1.cross(v2) -> v1.x*v2.y - v1.y*v2.x

        :return: The cross product
        """
        assert len(other) == 2
        return self.x * other[1] - self.y * other[0]

    def interpolate_to(self, other: tuple[float, float], range: float) -> Self:
        assert len(other) == 2
        return Vec2d(
            self.x + (other[0] - self.x) * range, self.y + (other[1] - self.y) * range
        )

    def convert_to_basis(
        self,
        x_vector: tuple[float, float],
        y_vector: tuple[float, float],
    ) -> Self:
        assert len(x_vector) == 2
        assert len(y_vector) == 2
        x = self.dot(x_vector) / Vec2d(*x_vector).get_length_sqrd()
        y = self.dot(y_vector) / Vec2d(*y_vector).get_length_sqrd()
        return Vec2d(x, y)

    @property
    def int_tuple(self) -> tuple[int, int]:
        """The x and y values of this vector as a tuple of ints.
        Uses round() to round to closest int.

        >>> Vec2d(0.9, 2.4).int_tuple
        (1, 2)
        """
        return round(self.x), round(self.y)

    @staticmethod
    def zero() -> Self:
        """A vector of zero length.

        >>> Vec2d.zero()
        Vec2d(0, 0)
        """
        return Vec2d(0, 0)

    @staticmethod
    def unit() -> Self:
        """A unit vector pointing up

        >>> Vec2d.unit()
        Vec2d(0, 1)
        """
        return Vec2d(0, 1)

    @staticmethod
    def ones() -> Self:
        """A vector where both x and y is 1

        >>> Vec2d.ones()
        Vec2d(1, 1)
        """
        return Vec2d(1, 1)
