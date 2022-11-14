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
    gl_Position = proj * vec4(in_position, 0.0, 1.0);
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
    gl_Position = proj * vec4(in_position, 0.0, 1.0);
}
"""

_LINE_GEOMETRY_SHADER = """
#version 330
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;
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
        gl_Position = positions[i];
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
        gl_Position = positions[i];
        EmitVertex();
    }
    EndPrimitive();
}
"""

_TEXTURE_VERTEX_SHADER = """
#version 330
uniform mat4 proj;
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = proj * vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}
"""

_TEXTURE_FRAGMENT_SHADER = """
#version 330
uniform sampler2D image;
in vec2 uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(image, uv).rgb, 1.0);
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
        self._points = ctx.buffer(reserve=len(points) * 4 * 2 * 10)
        self._scales = ctx.buffer(reserve=len(scales) * 4 * 10)
        self._colors = ctx.buffer(reserve=len(colors) * 4 * 4 * 10)

        self.vertex_array = ctx.vertex_array(
            program,
            [
                (self._points, "2f", "in_position"),
                (self._scales, "f", "in_scale"),
                (self._colors, "4f", "in_color"),
            ],
        )
        self.update(points, scales, colors)

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
        self._segments = ctx.buffer(reserve=len(segments) * 4 * 2 * 10)

        self.vertex_array = ctx.vertex_array(
            program,
            [(self._segments, "2f", "in_position")],
        )
        self.update(segments)

    def update(self, segments: NDArray) -> bool:
        length = segments.shape[0]
        if self._length != length:
            self._length = length
            self._segments.orphan(length * 4 * 2)
        self._segments.write(segments)
        return length > 0


class TextureVA(Renderable):
    MODE = mgl.TRIANGLE_STRIP

    def __init__(
        self,
        ctx: mgl.Context,
        program: mgl.Program,
        texture: mgl.Texture,
    ) -> None:
        self._ctx = ctx
        self._texture = texture
        quad_mat = np.array(
            # x, y, u, v
            [
                [0, 1, 0, 1],  # upper left
                [0, 0, 0, 0],  # lower left
                [1, 1, 1, 1],  # upper right
                [1, 0, 1, 0],  # lower right
            ],
            dtype=np.float32,
        )
        quad_mat_buffer = ctx.buffer(data=quad_mat)
        self.vertex_array = ctx.vertex_array(
            program,
            [(quad_mat_buffer, "2f 2f", "in_position", "in_uv")],
        )

    def update(self, image: bytes) -> None:
        self._texture.write(image)
        self._texture.use()


def _collect_circles(
    shapes: list[pymunk.Shape],
    size_scaling: float,
) -> tuple[NDArray, NDArray, NDArray]:
    points = []
    scales = []
    colors = []
    for circle in filter(lambda shape: isinstance(shape, pymunk.Circle), shapes):
        points.append(circle.body.position + circle.offset)
        scales.append(circle.radius * size_scaling)
        colors.append(circle.color)
    return (
        np.array(points, dtype=np.float32),
        np.array(scales, dtype=np.float32),
        np.array(colors, dtype=np.float32) / 255.0,
    )


def _collect_static_lines(shapes: list[pymunk.Shape]) -> NDArray:
    points = []
    for segment in filter(lambda shape: isinstance(shape, pymunk.Segment), shapes):
        body = segment.body
        if body.body_type != pymunk.Body.STATIC:
            continue
        points.append(segment.a)
        points.append(segment.b)
    return np.array(points, dtype=np.float32)


def _collect_sensors(shapes: list[pymunk.Shape]) -> NDArray:
    points = []
    for segment in filter(lambda shape: isinstance(shape, pymunk.Segment), shapes):
        body = segment.body
        if body.body_type == pymunk.Body.STATIC:
            continue
        pos = segment.body.position
        angle = segment.body.angle
        points.append(segment.a.rotated(angle) + pos)
        points.append(segment.b.rotated(angle) + pos)
    return np.array(points, dtype=np.float32)


def _collect_heads(shapes: list[pymunk.Shape]) -> NDArray:
    points = []
    for circle in filter(lambda shape: isinstance(shape, pymunk.Circle), shapes):
        pos = circle.body.position + circle.offset
        angle = circle.body.angle
        points.append(pymunk.Vec2d(0.0, circle.radius * 0.8).rotated(angle) + pos)
        points.append(pymunk.Vec2d(0.0, circle.radius * 1.2).rotated(angle) + pos)
    return np.array(points, dtype=np.float32)


def _collect_policies(
    bodies_and_policies: Iterable[tuple[pymunk.Body, NDArray]],
    max_arrow_length: float,
) -> NDArray:
    max_policy = max(map(lambda bp: np.linalg.norm(bp[1]), bodies_and_policies))  # type: ignore
    policy_scaling = max_arrow_length / max_policy
    points = []
    for body, policy in bodies_and_policies:
        a = body.position
        policy = pymunk.Vec2d(*(policy * policy_scaling))
        points.append(a)
        points.append(a + policy.rotated(body.angle))
    return np.array(points, dtype=np.float32)


def _get_clip_ranges(lengthes: list[float]) -> list[tuple[float, float]]:
    """Clip ranges to [-1, 1]"""
    total = sum(lengthes)
    left = -1.0
    res = []
    for length in lengthes:
        right = left + 2.0 * length / total
        res.append((left, right))
        left = right
    return res


class MglVisualizer:
    def __init__(
        self,
        x_range: float,
        y_range: float,
        env: PymunkEnv,
        figsize: tuple[float, float] | None = None,
        voffsets: tuple[int, ...] = (),
        hoffsets: tuple[int, ...] = (),
        vsync: bool = False,
        backend: str = "pyglet",
        title: str = "EmEvo PymunkEnv",
    ) -> None:
        self.pix_fmt = "rgba"

        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        w, h = int(figsize[0]), int(figsize[1])
        self._figsize = w + sum(hoffsets), h + sum(voffsets)
        self._screen_x = _get_clip_ranges([w, *hoffsets])
        self._screen_y = _get_clip_ranges([h, *voffsets])
        self._xrange, self._yrange = x_range, y_range
        self._range_min = min(x_range, y_range)
        if x_range < y_range:
            self._range_min = x_range
            self._size_scaling = figsize[0] / x_range * 2
        else:
            self._range_min = y_range
            self._size_scaling = figsize[1] / y_range * 2

        self._window = _make_window(
            title=title,
            size=self._figsize,
            backend=backend,
            vsync=vsync,
        )
        circle_program = self._make_gl_program(
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
        )
        shapes = env.get_space().shapes
        points, scales, colors = _collect_circles(shapes, self._size_scaling)
        self._circles = CircleVA(
            ctx=self._window.ctx,
            program=circle_program,
            points=points,
            scales=scales,
            colors=colors,
        )
        segment_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32),
            width=np.array([0.002], dtype=np.float32),
        )
        self._sensors = SegmentVA(
            ctx=self._window.ctx,
            program=segment_program,
            segments=_collect_sensors(shapes),
        )
        static_segment_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32),
            width=np.array([0.004], dtype=np.float32),
        )
        self._static_lines = SegmentVA(
            ctx=self._window.ctx,
            program=static_segment_program,
            segments=_collect_static_lines(shapes),
        )
        head_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.5, 0.0, 1.0, 1.0], dtype=np.float32),
            width=np.array([0.004], dtype=np.float32),
        )
        self._heads = SegmentVA(
            ctx=self._window.ctx,
            program=head_program,
            segments=_collect_heads(shapes),
        )
        self._overlays = {}

    def close(self) -> None:
        self._window.close()

    def get_image(self) -> NDArray:
        output = np.frombuffer(
            self._window.fbo.read(components=4, dtype="f1"),
            dtype=np.uint8,
        )
        w, h = self._figsize
        return np.ascontiguousarray(output.reshape(w, h, -1)[::-1])

    def render(self, env: PymunkEnv) -> None:
        self._window.clear(1.0, 1.0, 1.0)
        self._window.use()
        shapes = env.get_space().shapes
        if self._circles.update(*_collect_circles(shapes, self._size_scaling)):
            self._circles.render()
        if self._heads.update(_collect_heads(shapes)):
            self._heads.render()
        sensors = _collect_sensors(shapes)
        if self._sensors.update(sensors):
            self._sensors.render()
        self._static_lines.render()

    def overlay(self, name: str, value: Any) -> Any:
        """Render additional value as an overlay"""
        key = name.lower()
        if key == "arrow":
            segments = _collect_policies(value, self._range_min * 0.1)
            if "arrow" in self._overlays:
                do_render = self._overlays["arrow"].update(segments)
            else:
                arrow_program = self._make_gl_program(
                    vertex_shader=_LINE_VERTEX_SHADER,
                    geometry_shader=_ARROW_GEOMETRY_SHADER,
                    fragment_shader=_LINE_FRAGMENT_SHADER,
                    color=np.array([0.98, 0.45, 0.45, 1.0], dtype=np.float32),
                )
                self._overlays["arrow"] = SegmentVA(
                    ctx=self._window.ctx,
                    program=arrow_program,
                    segments=segments,
                )
                do_render = True
            if do_render:
                self._overlays["arrow"].render()
        elif key.startswith("stack"):
            xi, yi = map(int, key.split("-")[1:])
            image = value
            if key not in self._overlays:
                texture = self._window.ctx.texture(image.shape[:2], 3, image.tobytes())
                texture.build_mipmaps()
                program = self._make_gl_program(
                    vertex_shader=_TEXTURE_VERTEX_SHADER,
                    fragment_shader=_TEXTURE_FRAGMENT_SHADER,
                    screen_idx=(xi, yi),
                    game_x=(0.0, 1.0),
                    game_y=(0.0, 1.0),
                )
                self._overlays[key] = TextureVA(self._window.ctx, program, texture)
            self._overlays[key].update(image.tobytes())
            self._overlays[key].render()
        else:
            raise ValueError(f"Unsupported overlay in moderngl visualizer: {name}")

    def _make_gl_program(
        self,
        vertex_shader: str,
        geometry_shader: str | None = None,
        fragment_shader: str | None = None,
        screen_idx: tuple[int, int] = (0, 0),
        game_x: tuple[float, float] | None = None,
        game_y: tuple[float, float] | None = None,
        **kwargs: NDArray,
    ) -> mgl.Program:
        ctx = self._window.ctx
        ctx.enable(mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
        ctx.blend_func = mgl.DEFAULT_BLENDING
        prog = ctx.program(
            vertex_shader=vertex_shader,
            geometry_shader=geometry_shader,
            fragment_shader=fragment_shader,
        )
        proj = _make_projection_matrix(
            game_x=game_x or (0.0, self._xrange),
            game_y=game_y or (0.0, self._yrange),
            screen_x=self._screen_x[screen_idx[0]],
            screen_y=self._screen_y[screen_idx[1]],
        )
        prog["proj"].write(proj)
        for key, value in kwargs.items():
            prog[key].write(value)
        return prog

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
    size: tuple[int, int],
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


def _make_projection_matrix(
    game_x: tuple[float, float] = (0.0, 1.0),
    game_y: tuple[float, float] = (0.0, 1.0),
    screen_x: tuple[float, float] = (-1.0, 1.0),
    screen_y: tuple[float, float] = (-1.0, 1.0),
) -> NDArray:
    screen_width = screen_x[1] - screen_x[0]
    screen_height = screen_y[1] - screen_y[0]
    x_scale = screen_width / (game_x[1] - game_x[0])
    y_scale = screen_height / (game_y[1] - game_y[0])
    scale_mat = np.array(
        [
            [x_scale, 0, 0, 0],
            [0, y_scale, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    trans_mat = np.array(
        [
            [1, 0, 0, (sum(screen_x) - sum(game_x)) / screen_width],
            [0, 1, 0, (sum(screen_y) - sum(game_y)) / screen_height],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(np.dot(scale_mat, trans_mat).T)
