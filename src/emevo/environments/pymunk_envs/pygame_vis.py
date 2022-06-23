from __future__ import annotations

import pygame
import pymunk.pygame_util

from emevo.env import Visualizer
from emevo.environments.pymunk_envs.pymunk_env import PymunkEnv


class PygameVisualizer(Visualizer):
    def __init__(
        self,
        x_range: float,
        y_range: float,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        if figsize is None:
            self._figsize = 600, 600
        else:
            self._figsize = figsize
        pygame.display.init()
        self._background = pygame.Rect(0, 0, x_range, y_range)
        self._screen = pygame.display.set_mode(self._figsize)
        self._pymunk_surface = pygame.Surface((x_range, y_range))
        pymunk.pygame_util.positive_y_is_up = True
        self._draw_options = pymunk.pygame_util.DrawOptions(self._pymunk_surface)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def render(self, env: PymunkEnv) -> None:
        pygame.draw.rect(self._pymunk_surface, (255, 255, 255), self._background)
        space = env.get_space()
        space.debug_draw(self._draw_options)

    def show(self) -> None:
        transform = pygame.transform.smoothscale(self._pymunk_surface, self._figsize)
        self._screen.blit(transform, (0, 0))
        pygame.display.flip()
