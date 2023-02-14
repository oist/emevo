"""Example of using circle foraging environment"""
from __future__ import annotations

import sys

import numpy as np
import typer
from numpy.random import PCG64
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer

from emevo import make
from emevo.environments.pymunk_envs.qt_widget import PymunkMglWidget


def main(
    seed: int = 1,
    debug: bool = False,
    env_shape: str = "square",
) -> None:
    if debug:
        import loguru

        loguru.logger.enable("emevo")

    env = make(
        "CircleForaging-v0",
        env_shape=env_shape,
        sensor_range=(-60, 60),
        sensor_length=60,
        max_abs_angle=np.pi / 20,
        seed=seed,
    )
    bodies = env.bodies()
    gen = np.random.Generator(PCG64(seed=seed))

    def step_fn(env):
        actions = {body: body.act_space.sample(gen) for body in bodies}
        env.step(actions)

    app = QtWidgets.QApplication([])
    widget = PymunkMglWidget(env, (640, 640), step_fn=step_fn)  # type: ignore
    widget.setWindowTitle("Testing Qt widget")
    timer = QTimer(widget)
    timer.start(20)
    timer.timeout.connect(widget.update)  # type: ignore
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    typer.run(main)
