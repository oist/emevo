"""
A simple,  fast visualizer based on moderngl.
Currently, only supports circles and lines.
"""
from __future__ import annotations

from typing import Callable, ClassVar

import jax.numpy as jnp
import moderngl as mgl
import moderngl_window as mglw
import numpy as np
from moderngl_window.context import headless
from numpy.typing import NDArray

from emevo.environments.phyjax2d import Circle, Segment, Space, State, StateDict


NOWHERE: float = -1000.0


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
    float alpha = smoothstep(0.5, 0.5 - delta, dist);
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
    circle: Circle,
    state: State,
    circle_scaling: float,
) -> tuple[NDArray, NDArray, NDArray]:
    flag = np.array(state.is_active).reshape(-1, 1)
    points = np.where(flag, np.array(state.p.xy, dtype=np.float32), NOWHERE)
    scales = circle.radius * circle_scaling
    colors = np.array(circle.rgba, dtype=np.float32) / 255.0
    is_active = np.expand_dims(np.array(state.is_active), axis=1)
    colors = np.where(is_active, colors, np.ones_like(colors))
    return points, np.array(scales, dtype=np.float32), colors


def _collect_static_lines(segment: Segment, state: State) -> NDArray:
    a, b = segment.point1, segment.point2
    a = state.p.transform(a)
    b = state.p.transform(b)
    flag = np.repeat(np.array(state.is_active), 2).reshape(-1, 1)
    return np.where(flag, np.concatenate((a, b), axis=1).reshape(-1, 2), NOWHERE)


def _collect_heads(circle: Circle, state: State) -> NDArray:
    y = jnp.array(circle.radius)
    x = jnp.zeros_like(y)
    p1, p2 = jnp.stack((x, y * 0.8), axis=1), jnp.stack((x, y * 1.2), axis=1)
    p1, p2 = state.p.transform(p1), state.p.transform(p2)
    flag = np.repeat(np.array(state.is_active), 2).reshape(-1, 1)
    return np.where(flag, np.concatenate((p1, p2), axis=1).reshape(-1, 2), NOWHERE)


def _get_clip_ranges(lengthes: list[float]) -> list[tuple[float, float]]:
    """Clip ranges to [-1, 1]"""
    total = sum(lengthes)
    res = []
    left = -1.0
    for length in lengthes:
        right = left + 2.0 * length / total
        res.append((left, right))
        left = right
    return res


class MglRenderer:
    """Render pymunk environments to the given moderngl context."""

    def __init__(
        self,
        context: mgl.Context,
        screen_width: int,
        screen_height: int,
        x_range: float,
        y_range: float,
        space: Space,
        stated: StateDict,
        voffsets: tuple[int, ...] = (),
        hoffsets: tuple[int, ...] = (),
        sensor_fn: Callable[[StateDict], tuple[NDArray, NDArray]] | None = None,
    ) -> None:
        self._context = context

        self._screen_x = _get_clip_ranges([screen_width, *hoffsets])
        self._screen_y = _get_clip_ranges([screen_height, *voffsets])
        self._x_range, self._y_range = x_range, y_range
        self._range_min = min(x_range, y_range)

        if x_range < y_range:
            self._range_min = x_range
            self._circle_scaling = screen_width / x_range * 2
        else:
            self._range_min = y_range
            self._circle_scaling = screen_height / y_range * 2

        self._space = space
        circle_program = self._make_gl_program(
            vertex_shader=_CIRCLE_VERTEX_SHADER,
            fragment_shader=_CIRCLE_FRAGMENT_SHADER,
        )
        points, scales, colors = _collect_circles(
            space.shaped.circle,
            stated.circle,
            self._circle_scaling,
        )
        self._circles = CircleVA(
            ctx=context,
            program=circle_program,
            points=points,
            scales=scales,
            colors=colors,
        )
        points, scales, colors = _collect_circles(
            space.shaped.static_circle,
            stated.static_circle,
            self._circle_scaling,
        )
        self._static_circles = CircleVA(
            ctx=context,
            program=circle_program,
            points=points,
            scales=scales,
            colors=colors,
        )
        static_segment_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32),
            width=np.array([0.004], dtype=np.float32),
        )
        self._static_lines = SegmentVA(
            ctx=context,
            program=static_segment_program,
            segments=_collect_static_lines(space.shaped.segment, stated.segment),
        )
        if sensor_fn is not None:
            segment_program = self._make_gl_program(
                vertex_shader=_LINE_VERTEX_SHADER,
                geometry_shader=_LINE_GEOMETRY_SHADER,
                fragment_shader=_LINE_FRAGMENT_SHADER,
                color=np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32),
                width=np.array([0.002], dtype=np.float32),
            )

            def collect_sensors(stated: StateDict) -> NDArray:
                sensors = np.concatenate(
                    sensor_fn(stated=stated),  # type: ignore
                    axis=1,
                )
                sensors = sensors.reshape(-1, 2).astype(jnp.float32)
                flag = np.repeat(
                    np.array(stated.circle.is_active),
                    sensors.shape[0] // stated.circle.batch_size(),
                )
                return np.where(
                    flag.reshape(-1, 1),
                    sensors,
                    NOWHERE,
                )

            self._sensors = SegmentVA(
                ctx=context,
                program=segment_program,
                segments=collect_sensors(stated),
            )
            self._collect_sensors = collect_sensors
        else:
            self._sensors, self._collect_sensors = None, None

        head_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.5, 0.0, 1.0, 1.0], dtype=np.float32),
            width=np.array([0.004], dtype=np.float32),
        )
        self._heads = SegmentVA(
            ctx=context,
            program=head_program,
            segments=_collect_heads(space.shaped.circle, stated.circle),
        )

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
        self._context.enable(mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
        prog = self._context.program(
            vertex_shader=vertex_shader,
            geometry_shader=geometry_shader,
            fragment_shader=fragment_shader,
        )
        proj = _make_projection_matrix(
            game_x=game_x or (0, self._x_range),
            game_y=game_y or (0, self._y_range),
            screen_x=self._screen_x[screen_idx[0]],
            screen_y=self._screen_y[screen_idx[1]],
        )
        prog["proj"].write(proj)  # type: ignore
        for key, value in kwargs.items():
            prog[key].write(value)  # type: ignore
        return prog


    def render(self, stated: StateDict) -> None:
        circles = _collect_circles(
            self._space.shaped.circle,
            stated.circle,
            self._circle_scaling,
        )
        static_circles = _collect_circles(
            self._space.shaped.static_circle,
            stated.static_circle,
            self._circle_scaling,
        )
        if self._circles.update(*circles):
            self._circles.render()
        if self._static_circles.update(*static_circles):
            self._static_circles.render()
        if self._sensors is not None and self._collect_sensors is not None:
            if self._sensors.update(self._collect_sensors(stated)):
                self._sensors.render()
        if self._heads.update(_collect_heads(self._space.shaped.circle, stated.circle)):
            self._heads.render()
        self._static_lines.render()


class MglVisualizer:
    """
    Visualizer class that follows the `emevo.Visualizer` protocol.
    Considered as a main interface to use this visualizer.
    """

    def __init__(
        self,
        x_range: float,
        y_range: float,
        space: Space,
        stated: StateDict,
        figsize: tuple[float, float] | None = None,
        voffsets: tuple[int, ...] = (),
        hoffsets: tuple[int, ...] = (),
        vsync: bool = False,
        backend: str = "pyglet",
        sensor_fn: Callable[[StateDict], tuple[NDArray, NDArray]] | None = None,
        title: str = "EmEvo CircleForaging",
    ) -> None:
        self.pix_fmt = "rgba"

        if figsize is None:
            figsize = x_range * 3.0, y_range * 3.0
        w, h = int(figsize[0]), int(figsize[1])
        self._figsize = w + int(sum(hoffsets)), h + int(sum(voffsets))

        self._window = _make_window(
            title=title,
            size=self._figsize,
            backend=backend,
            vsync=vsync,
        )
        self._renderer = MglRenderer(
            context=self._window.ctx,
            screen_width=w,
            screen_height=h,
            x_range=x_range,
            y_range=y_range,
            space=space,
            stated=stated,
            voffsets=voffsets,
            hoffsets=hoffsets,
            sensor_fn=sensor_fn,
        )

    def close(self) -> None:
        self._window.close()

    def get_image(self) -> NDArray:
        output = np.frombuffer(
            self._window.fbo.read(components=4, dtype="f1"),
            dtype=np.uint8,
        )
        w, h = self._figsize
        return output.reshape(h, w, -1)[::-1]

    def render(self, state: StateDict) -> None:
        self._window.clear(1.0, 1.0, 1.0)
        self._window.use()
        self._renderer.render(stated=state)

    def show(self) -> None:
        self._window.swap_buffers()


class _EglHeadlessWindow(headless.Window):
    name = "egl-headless"

    def init_mgl_context(self) -> None:
        """Create an standalone context and framebuffer"""
        self._ctx = mgl.create_standalone_context(
            require=self.gl_version_code,
            backend="egl",  # type: ignore
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
