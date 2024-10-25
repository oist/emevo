"""
A simple,  fast visualizer based on moderngl.
Currently, only supports circles and lines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp
import moderngl as mgl
import moderngl_window as mglw
import numpy as np
from moderngl_window.context import headless
from numpy.typing import NDArray
from phyjax2d import Circle, Segment, Space, State, StateDict

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
out float g_len;
out float g_u;
out float g_v;
void main() {
    vec2 a = gl_in[0].gl_Position.xy;
    vec2 b = gl_in[1].gl_Position.xy;
    vec2 a2b = b - a;
    vec2 a2left = vec2(-a2b.y, a2b.x) / length(a2b) * width;
    float len = length(a2b) * 0.5;
    vec4 positions[4] = vec4[4](
        vec4(a + a2left, 0.0, 1.0),
        vec4(a - a2left, 0.0, 1.0),
        vec4(b + a2left, 0.0, 1.0),
        vec4(b - a2left, 0.0, 1.0)
    );
    float gus[4] = float[4](width, -width, width, -width);
    float gvs[4] = float[4](len, len, -len, -len);
    for (int i = 0; i < 4; ++i) {
        g_len = len;
        g_u = gus[i];
        g_v = gvs[i];
        gl_Position = positions[i];
        EmitVertex();
    }
    EmitVertex();
    EndPrimitive();
}
"""

_LINE_FRAGMENT_SHADER = """
#version 330
in float g_u;
in float g_v;
in float g_len;
out vec4 f_color;
uniform float width;
uniform float w_rad;
uniform float l_rad;
uniform vec4 color;
void main() {
    float aw = 1.0 - smoothstep(1.0 - ((2.0 * w_rad) / width), 1.0, abs(g_u / width));
    float al = 1.0 - smoothstep(1.0 - ((2.0 * l_rad) / g_len), 1.0, abs(g_v / g_len));
    f_color = color;
    f_color.a *= min(aw, al);
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


def _get_sc_color(colors: NDArray, state: State) -> NDArray:
    # Clip labels to make it work when less number of colors are provided
    label = np.clip(np.array(state.label), 0, len(colors) - 1)
    default_color = colors[label].astype(np.float32) / 255.0
    inactive_color = np.ones_like(default_color)
    is_active_expanded = np.expand_dims(state.is_active, axis=1)
    return np.where(is_active_expanded, default_color, inactive_color)


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
        sc_color_opt: NDArray | None = None,
        sensor_color: NDArray | None = None,
        sensor_width: float = 0.001,
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

        if sc_color_opt is None:
            self._sc_color = np.array([[254, 2, 162, 255]])
        else:
            self._sc_color = sc_color_opt

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
        points, scales, _ = _collect_circles(
            space.shaped.static_circle,
            stated.static_circle,
            self._circle_scaling,
        )
        self._static_circles = CircleVA(
            ctx=context,
            program=circle_program,
            points=points,
            scales=scales,
            colors=_get_sc_color(self._sc_color, stated.static_circle),
        )
        static_segment_program = self._make_gl_program(
            vertex_shader=_LINE_VERTEX_SHADER,
            geometry_shader=_LINE_GEOMETRY_SHADER,
            fragment_shader=_LINE_FRAGMENT_SHADER,
            color=np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32),
            width=np.array([0.004], dtype=np.float32),
            w_rad=np.array([0.001], dtype=np.float32),
            l_rad=np.array([0.001], dtype=np.float32),
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
                color=(
                    np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32)
                    if sensor_color is None
                    else sensor_color
                ),
                width=np.array([sensor_width], dtype=np.float32),
                w_rad=np.array([sensor_width / 4], dtype=np.float32),
                l_rad=np.array([sensor_width / 4], dtype=np.float32),
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

    @staticmethod
    def _get_colors(default_colors: NDArray, colors: NDArray | None) -> NDArray:
        if colors is None:
            return default_colors
        else:
            clen = colors.shape[0]
            if clen < default_colors.shape[0]:
                return np.concatenate(
                    (colors.astype(np.float32), default_colors[clen:]),
                    axis=0,
                )
            else:
                return colors.astype(np.float32)

    def render(
        self,
        stated: StateDict,
        circle_colors: NDArray | None = None,
        sc_colors: NDArray | None = None,
    ) -> None:
        circle_points, circle_scale, circle_colors_default = _collect_circles(
            self._space.shaped.circle,
            stated.circle,
            self._circle_scaling,
        )
        circle_colors = self._get_colors(circle_colors_default, circle_colors)
        if self._circles.update(circle_points, circle_scale, circle_colors):
            self._circles.render()
        sc_points, sc_scale, _ = _collect_circles(
            self._space.shaped.static_circle,
            stated.static_circle,
            self._circle_scaling,
        )
        sc_colors = self._get_colors(
            _get_sc_color(self._sc_color, stated.static_circle),
            sc_colors,
        )
        if self._static_circles.update(sc_points, sc_scale, sc_colors):
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
        food_color: NDArray,
        sensor_color: NDArray | None = None,
        figsize: tuple[float, float] | None = None,
        voffsets: tuple[int, ...] = (),
        hoffsets: tuple[int, ...] = (),
        vsync: bool = False,
        backend: str = "pyglet",
        sensor_fn: Callable[[StateDict], tuple[NDArray, NDArray]] | None = None,
        sensor_width: float = 0.001,
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
            sc_color_opt=food_color,
            sensor_color=sensor_color,
            sensor_width=sensor_width,
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

    def render(self, state: StateDict, **kwargs) -> None:
        self._window.clear(1.0, 1.0, 1.0)
        self._window.use()
        self._renderer.render(stated=state, **kwargs)

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
