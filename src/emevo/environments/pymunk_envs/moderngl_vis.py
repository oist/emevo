"""
A simple but fast visualizer based on moderngl.
Currently, only supports circles and lines.
"""
from __future__ import annotations

from typing import ClassVar, Protocol

import moderngl as mgl
import moderngl_window as mglw
import numpy as np
import pymunk
from numpy.typing import NDArray

from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv

_CIRCLE_VERTEX_SHADER = """
#version 330
uniform mat4 proj;
in vec2 in_position;
in float in_scale;
in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0) * proj;
    gl_PointSize = in_scale;
    v_color = vec3(in_color);
}
"""

# Smoothing by fwidth is based on: https://rubendv.be/posts/fwidth/
_CIRCLE_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    float dist = length(gl_PointCoord.xy - vec2(0.5));
    float delta = fwidth(dist);
    float alpha = smoothstep(0.45, 0.45 - delta, dist);
    f_color = vec4(v_color * alpha, alpha);
}
"""


class Renderable(Protocol):
    MODE: ClassVar[int]
    vertex_array: mgl.VertexArray

    def render(self) -> None:
        self.vertex_array.render(mode=self.MODE)


class CircleVA:
    MODE = mgl.POINTS

    def __init__(
        self,
        ctx: mgl.Context,
        program: mgl.Program,
        points: NDArray,
        scales: NDArray,
        colors: NDArray,
    ) -> None:
        self._ctx = ctx
        self._n_points = points.shape[0]
        if colors.ndim == 1:
            colors = np.tile(colors, (self._n_points, 1))

        self.points = ctx.buffer(points.astype(np.float32))
        self.scales = ctx.buffer(np.ones(self._n_points, dtype=np.float32) * scales)
        self.colors = ctx.buffer(colors.astype(np.float32))

        if colors.shape[1] == 4:
            color_type = "4f"
        else:
            color_type = "3f"

        self.vao = ctx.vertex_array(
            program,
            [
                (self.points, "2f", "in_position"),
                (self.scales, "f", "in_scale"),
                (self.colors, color_type, "in_color"),
            ],
        )


def _accumulate_circles(shapes: list[pymunk.Shape]):
    positions = []
    scales = []
    colors = []
    for circles in filter(lambda shape: isinstance(shape, pymunk.Circle), shapes):
        pass


class MglVisualizer:
    def __init__(
        self,
        x_range: float,
        y_range: float,
        figsize: tuple[int, int] | None = None,
        title: str = "Pymunk Env",
    ) -> None:
        if figsize is None:
            figsize = 600, 600
        self._window = _make_window(title, figsize)
        scale = figsize[0] / x_range, figsize[1] / y_range
        circle_program = _make_gl_program(
            self._window.ctx,
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
            proj_scale=scale,
        )
        self._circle_va = CircleVA(
            ctx=self._window.ctx,
            program=circle_program,
        )

    def close(self) -> None:
        pass

    def get_image(self) -> NDArray:
        pass

    def render(self, env: PymunkEnv) -> None:
        pass

    def show(self) -> None:
        pass


def _make_window(
    title: str,
    size: tuple[int, int],
    **kwargs,
) -> mglw.BaseWindow:
    window_cls = mglw.get_window_cls("moderngl_window.context.pyglet.Window")
    window = window_cls(title=title, gl_version=(4, 1), size=size, **kwargs)
    mglw.activate_context(ctx=window.ctx)
    return window


def _make_gl_program(
    ctx: mgl.Context,
    vertex_shader: str,
    fragment_shader: str,
    proj_scale: tuple[float, float] = (1.0, 1.0),
) -> mgl.Program:
    ctx.enable(mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
    ctx.blend_func = mgl.DEFAULT_BLENDING
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    proj = _make_projection_matrix(proj_scale)
    prog["proj"].write(proj)
    return prog


def _make_projection_matrix(window_size: tuple[float, float]) -> NDArray:
    w, h = window_size
    scale_mat = np.array(
        [[2 / w, 0, 0, 0], [0, 2 / h, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    trans_mat = np.array(
        [[1, 0, 0, -w / 2], [0, 1, 0, -h / 2], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    scale = np.dot(np.eye(4, dtype=np.float32), scale_mat)
    return np.dot(scale, trans_mat)
