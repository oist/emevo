"""
A simple but fast visualizer based on moderngl.
Currently, only supports circles and lines.
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterable

import moderngl as mgl
import moderngl_window as mglw
import numpy as np
import pymunk
from moderngl_window.context import headless
from numpy.typing import NDArray

from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv

_CIRCLE_VERTEX_SHADER = """
#version 330
uniform mat4 proj;
in vec2 in_position;
in float in_scale;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0) * proj;
    gl_PointSize = in_scale;
    v_color = in_color;
}
"""

# Smoothing by fwidth is based on: https://rubendv.be/posts/fwidth/
_CIRCLE_FRAGMENT_SHADER = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    float dist = length(gl_PointCoord.xy - vec2(0.5));
    float delta = fwidth(dist);
    float alpha = smoothstep(0.45, 0.45 - delta, dist);
    f_color = v_color * alpha;
}
"""

_LINE_VERTEX_SHADER = """
#version 330
in vec2 in_position;
uniform mat4 proj;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_LINE_GEOMETRY_SHADER = """
#version 330
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;
uniform mat4 proj;
uniform float width;
void main() {
    vec2 a = gl_in[0].gl_Position.xy;
    vec2 b = gl_in[1].gl_Position.xy;
    vec2 a2b = b - a;
    vec2 a2left = vec2(-a2b.y, a2b.x) / length(a2b) * width;

    vec4 positions[4] = vec4[4](
        vec4(a - a2left, 0.0, 1.0),
        vec4(a + a2left, 0.0, 1.0),
        vec4(b - a2left, 0.0, 1.0),
        vec4(b + a2left, 0.0, 1.0)
    );
    for (int i = 0; i < 4; ++i) {
        gl_Position = positions[i] * proj;
        EmitVertex();
    }
    EndPrimitive();
}
"""

_LINE_FRAGMENT_SHADER = """
#version 330
out vec4 f_color;
uniform vec4 color;
void main() {
    f_color = color;
}
"""


_ARROW_GEOMETRY_SHADER = """
#version 330
layout (lines) in;
layout (triangle_strip, max_vertices = 7) out;
uniform mat4 proj;
void main() {
    vec2 a = gl_in[0].gl_Position.xy;
    vec2 b = gl_in[1].gl_Position.xy;
    vec2 a2b = b - a;
    float a2b_len = length(a2b);
    float width = min(0.004, a2b_len * 0.12);
    vec2 a2left = vec2(-a2b.y, a2b.x) / length(a2b) * width;
    vec2 c = a + a2b * 0.5;
    vec2 c2head = a2left * 2.5;

    vec4 positions[7] = vec4[7](
        vec4(a - a2left, 0.0, 1.0),
        vec4(a + a2left, 0.0, 1.0),
        vec4(c - a2left, 0.0, 1.0),
        vec4(c + a2left, 0.0, 1.0),
        vec4(c - c2head, 0.0, 1.0),
        vec4(b, 0.0, 1.0),
        vec4(c + c2head, 0.0, 1.0)
    );
    for (int i = 0; i < 7; ++i) {
        gl_Position = positions[i] * proj;
        EmitVertex();
    }
    EndPrimitive();
}
"""


class Renderable:
    MODE: ClassVar[int]
    vertex_array: mgl.VertexArray

    def render(self) -> None:
        self.vertex_array.render(mode=self.MODE)


class CircleVA(Renderable):
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
        self._length = points.shape[0]
        self._points = ctx.buffer(points)
        self._scales = ctx.buffer(scales)
        self._colors = ctx.buffer(colors)

        self.vertex_array = ctx.vertex_array(
            program,
            [
                (self._points, "2f", "in_position"),
                (self._scales, "f", "in_scale"),
                (self._colors, "4f", "in_color"),
            ],
        )

    def update(self, points: NDArray, scales: NDArray, colors: NDArray) -> bool:
        length = points.shape[0]
        if self._length != length:
            self._length = length
            self._points.orphan(length * 4 * 2)
            self._scales.orphan(length * 4)
            self._colors.orphan(length * 4 * 4)
        self._points.write(points)
        self._scales.write(scales)
        self._colors.write(colors)
        return length > 0


class SegmentVA(Renderable):
    MODE = mgl.LINES

    def __init__(
        self,
        ctx: mgl.Context,
        program: mgl.Program,
        segments: NDArray,
    ) -> None:
        self._ctx = ctx
        self._length = segments.shape[0]
        self._segments = ctx.buffer(segments)

        self.vertex_array = ctx.vertex_array(
            program,
            [(self._segments, "2f", "in_position")],
        )

    def update(self, segments: NDArray) -> bool:
        length = segments.shape[0]
        if self._length != length:
            self._length = length
            self._segments.orphan(length * 4 * 2)
        self._segments.write(segments)
        return length > 0


def _collect_circles(
    shapes: list[pymunk.Shape],
    pos_scaling: tuple[float, float],
    size_scaling: float,
) -> tuple[NDArray, NDArray, NDArray]:
    points = []
    scales = []
    colors = []
    for circle in filter(lambda shape: isinstance(shape, pymunk.Circle), shapes):
        x, y = circle.body.position + circle.offset
        points.append([x * pos_scaling[0], y * pos_scaling[1]])
        scales.append(circle.radius * size_scaling)
        colors.append(circle.color)
    return (
        np.array(points, dtype=np.float32),
        np.array(scales, dtype=np.float32),
        np.array(colors, dtype=np.float32) / 255.0,
    )


def _collect_sensors(
    shapes: list[pymunk.Shape],
    pos_scaling: tuple[float, float],
) -> NDArray:
    points = []
    for segment in filter(lambda shape: isinstance(shape, pymunk.Segment), shapes):
        body = segment.body
        if body.body_type == pymunk.Body.STATIC:
            continue
        pos = segment.body.position
        angle = segment.body.angle
        ax, ay = segment.a.rotated(angle) + pos
        bx, by = segment.b.rotated(angle) + pos
        points.append([ax * pos_scaling[0], ay * pos_scaling[1]])
        points.append([bx * pos_scaling[0], by * pos_scaling[1]])
    return np.array(points, dtype=np.float32)


def _collect_heads(
    shapes: list[pymunk.Shape],
    pos_scaling: tuple[float, float],
) -> NDArray:
    points = []
    for circle in filter(lambda shape: isinstance(shape, pymunk.Circle), shapes):
        pos = circle.body.position + circle.offset
        angle = circle.body.angle
        p1 = pymunk.Vec2d(0.0, circle.radius * 0.9).rotated(angle) + pos
        p2 = pymunk.Vec2d(0.0, circle.radius * 1.2).rotated(angle) + pos
        points.append([p1.x * pos_scaling[0], p1.y * pos_scaling[1]])
        points.append([p2.x * pos_scaling[0], p2.y * pos_scaling[1]])
    return np.array(points, dtype=np.float32)


def _collect_policies(
    bodies_and_policies: Iterable[tuple[pymunk.Body, float, float]],
    pos_scaling: tuple[float, float],
    max_arrow_length: float,
) -> NDArray:
    max_policy = max(
        map(lambda pp: np.sqrt(pp[1] ** 2 + pp[2] ** 2), bodies_and_policies)
    )
    policy_scaling = max_arrow_length / max_policy
    points = []
    for body, px, py in bodies_and_policies:
        a = body.position
        policy = pymunk.Vec2d(px * policy_scaling, py * policy_scaling)
        b = a + policy.rotated(body.angle)
        points.append([a.x * pos_scaling[0], a.y * pos_scaling[1]])
        points.append([b.x * pos_scaling[0], b.y * pos_scaling[1]])
    return np.array(points, dtype=np.float32)


class MglVisualizer:
    def __init__(
        self,
        x_range: float,
        y_range: float,
        env: PymunkEnv,
        figsize: tuple[float, float] | None = None,
        vsync: bool = False,
        backend: str = "pyglet",
        title: str = "Pymunk Env",
    ) -> None:
        self.pix_fmt = "rgba"

        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        self._window = _make_window(
            title=title,
            size=figsize,
            backend=backend,
            vsync=vsync,
        )
        self._figsize = int(figsize[0]), int(figsize[1])
        self._pos_scaling = 1.0 / x_range, 1.0 / y_range
        self._range_min = min(x_range, y_range)
        self._size_scaling = figsize[0] / x_range * 2
        circle_program = _make_gl_program(
            self._window.ctx,
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
        )
        shapes = env.get_space().shapes
        points, scales, colors = _collect_circles(
            shapes,
            self._pos_scaling,
            self._size_scaling,
        )
        self._circles = CircleVA(
            ctx=self._window.ctx,
            program=circle_program,
            points=points,
            scales=scales,
            colors=colors,
        )
        segment_program = _make_gl_program(
            self._window.ctx,
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
        )
        segment_program["color"].write(np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32))
        segment_program["width"].write(np.array([0.002], dtype=np.float32))
        self._sensors = SegmentVA(
            ctx=self._window.ctx,
            program=segment_program,
            segments=_collect_sensors(shapes, self._pos_scaling),
        )
        head_program = _make_gl_program(
            self._window.ctx,
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
        )
        head_program["color"].write(np.array([0.5, 0.0, 1.0, 1.0], dtype=np.float32))
        head_program["width"].write(np.array([0.004], dtype=np.float32))
        self._heads = SegmentVA(
            ctx=self._window.ctx,
            program=head_program,
            segments=_collect_heads(shapes, self._pos_scaling),
        )
        self._overlays = {}

    def close(self) -> None:
        self._window.close()

    def get_image(self) -> NDArray:
        output = np.frombuffer(
            self._window.fbo.read(components=4, dtype="f4"),
            dtype=np.float32,
        )
        output = output.reshape(*self._figsize, 4)
        output = np.flip(output, axis=0)  # Reverse image
        return np.multiply(output, 255).astype(np.uint8)

    def render(self, env: PymunkEnv) -> None:
        self._window.clear(1.0, 1.0, 1.0)
        self._window.use()
        shapes = env.get_space().shapes
        points, scales, colors = _collect_circles(
            shapes,
            self._pos_scaling,
            self._size_scaling,
        )
        if self._circles.update(points, scales, colors):
            self._circles.render()
        if self._sensors.update(_collect_sensors(shapes, self._pos_scaling)):
            self._sensors.render()
        if self._heads.update(_collect_heads(shapes, self._pos_scaling)):
            self._heads.render()

    def overlay(self, name: str, value: Any) -> Any:
        """Render additional value as an overlay"""
        if name.lower() == "arrow":
            segments = _collect_policies(
                value,
                self._pos_scaling,
                self._range_min * 0.1,
            )
            if "arrow" in self._overlays:
                do_render = self._overlays["arrow"].update(segments)
            else:
                arrow_program = _make_gl_program(
                    self._window.ctx,
                    vertex_shader=_LINE_VERTEX_SHADER,
                    geometry_shader=_ARROW_GEOMETRY_SHADER,
                    fragment_shader=_LINE_FRAGMENT_SHADER,
                )
                arrow_program["color"].write(
                    np.array([0.98, 0.45, 0.45, 1.0], dtype=np.float32)
                )
                self._overlays["arrow"] = SegmentVA(
                    ctx=self._window.ctx,
                    program=arrow_program,
                    segments=segments,
                )
                do_render = True
            if do_render:
                self._overlays["arrow"].render()
        else:
            raise ValueError(f"Unsupported overlay in moderngl visualizer: {name}")

    def show(self) -> None:
        self._window.swap_buffers()


class _EglHeadlessWindow(headless.Window):
    name = "egl-headless"

    def init_mgl_context(self) -> None:
        """Create an standalone context and framebuffer"""
        self._ctx = mgl.create_standalone_context(
            require=self.gl_version_code,
            backend="egl",
        )
        self._fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(self.size, 4, samples=self._samples),
            depth_attachment=self.ctx.depth_texture(self.size, samples=self._samples),
        )
        self.use()


def _make_window(
    *,
    title: str,
    size: tuple[float, float],
    backend: str,
    **kwargs,
) -> mglw.BaseWindow:
    if backend == "headless":
        window_cls = _EglHeadlessWindow
    else:
        window_cls = mglw.get_window_cls(f"moderngl_window.context.{backend}.Window")
    window = window_cls(title=title, gl_version=(4, 1), size=size, **kwargs)
    mglw.activate_context(ctx=window.ctx)
    return window


def _make_gl_program(
    ctx: mgl.Context,
    *,
    vertex_shader: str,
    geometry_shader: str | None = None,
    fragment_shader: str | None = None,
    proj_scale: tuple[float, float] = (1.0, 1.0),
) -> mgl.Program:
    ctx.enable(mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
    ctx.blend_func = mgl.DEFAULT_BLENDING
    prog = ctx.program(
        vertex_shader=vertex_shader,
        geometry_shader=geometry_shader,
        fragment_shader=fragment_shader,
    )
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
