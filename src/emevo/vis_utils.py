import abc
import typing as t

import numpy as np

try:
    import moderngl as mgl
    import moderngl_window as mglw
except ImportError as e:
    raise ImportError(
        "noderngl and moderngl-window is required for visualization"
    ) from e


CIRCLE_VERTEX_SHADER = """
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
CIRCLE_FRAGMENT_SHADER = """
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


class Renderable(abc.ABC):
    MODE: int

    @property
    @abc.abstractmethod
    def vertex_array(self) -> mgl.VertexArray:
        pass

    def render(self) -> None:
        self.vertex_array.render(mode=self.MODE)


class CircleBuffers(Renderable):
    MODE = mgl.POINTS

    def __init__(
        self,
        ctx: mgl.Context,
        program: mgl.Program,
        points: np.ndarray,
        scales: t.Union[float, np.ndarray],
        colors: np.ndarray,
    ) -> None:
        self._ctx = ctx
        self._n_points = points.shape[0]
        if isinstance(scales - 0.0, float):
            scales = np.ones(self._n_points) * scales
        if colors.ndim == 1:
            colors = np.tile(colors, (self._n_points, 1))

        self.points = ctx.buffer(points.astype(np.float32))
        self.scales = ctx.buffer(scales.astype(np.float32))
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

    @property
    def vertex_array(self) -> mgl.VertexArray:
        return self.vao


def make_window(
    title: str,
    size: t.Tuple[int, int],
    backend: str = "pyglet",
    **kwargs,
) -> mglw.BaseWindow:
    window_cls = mglw.get_window_cls(f"moderngl_window.context.{backend}.Window")
    window = window_cls(title=title, gl_version=(4, 1), size=size, **kwargs)
    mglw.activate_context(ctx=window.ctx)
    return window


def make_program(
    ctx: mgl.Context,
    vertex_shader: str = CIRCLE_VERTEX_SHADER,
    fragment_shader: str = CIRCLE_FRAGMENT_SHADER,
    proj_scale: t.Tuple[int, int] = (1.0, 1.0),
) -> mgl.Program:
    ctx.enable(mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
    ctx.blend_func = mgl.DEFAULT_BLENDING
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    prog["proj"].write(_proj(proj_scale))
    return prog


def _proj(window_size: t.Tuple[int, int]) -> np.ndarray:
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


if __name__ == "__main__":
    size = 640, 480
    window = make_window("Test", size)
    prog = make_program(window.ctx, proj_scale=(1.0, 1.0))
    # prog["color"].write(np.array([0.1, 1.0, 1.0], dtype=np.float32))
    red_circles = CircleBuffers(
        window.ctx,
        prog,
        np.random.uniform(0.0, 1.0, size=(20, 2)),
        10.0,
        np.array([0.9, 0.1, 0.1]),
    )
    blue_circles = CircleBuffers(
        window.ctx,
        prog,
        np.random.uniform(0.0, 1.0, size=(10, 2)),
        20.0,
        np.array([0.1, 0.1, 0.9]),
    )
    count = 0
    while not window.is_closing:
        count += 1
        if count % 100 == 0:
            red_circles.points.write(
                np.random.uniform(0.0, 1.0, size=(20, 2)).astype(np.float32)
            )
        window.clear(1.0, 1.0, 1.0)
        red_circles.render()
        blue_circles.render()
        # Render stuff here
        window.swap_buffers()
