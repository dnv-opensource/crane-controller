from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces

if TYPE_CHECKING:
    from py_crane.crane import Crane

# Type aliases for observations and actions
CraneObs = dict[str, npt.NDArray[np.int_]]


class Actions:
    def __init__(self, mode: str) -> None:
        self.mode = mode


class ControlledCraneEnv(gym.Env[CraneObs, int]):
    """Environment of the controlled py-crane based mobile crane.

    using the matplotlib-based animation module from py-crane.

    Args:
        crane (py_crane): the crane object to use as basis
        render_modes (str): 'animation' (use direct animation) or 'data' (return boom.end for all booms)
        size (int): The axis length in all directions, but -z


    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": ["animation", "data"], "render_fps": 4}  # type: ignore[assignment]

    def __init__(
        self,
        crane: Crane,
        mov_mode: str = "separate",
        render_mode: str | None = None,
        size: int = 10,
    ) -> None:
        self.crane = crane
        self.mode: int = 1 if mov_mode == "separate" else 2
        self.size = size
        self.figsize: tuple[int, int] = (15, 15)  # The matplotlib animation window

        # Observations are dictionaries with the agent's and the target's location.
        # Each boom.end is encoded as ndarray
        # The target (the landing place on the 'vessel deck') is also encoded as ndarray
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.integer),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.integer),
            }
        )

        # We have 4 acceleration actions which can each be min, zero or max, corresponding to acceleration of booms:
        # pedestal azimuthal
        # boom polar
        # boom length
        # wire length
        # Coded as integer if 'separate' and MultiDiscrete else.
        self.action_space = spaces.MultiDiscrete(np.array((3, 3, 3, 3), int), seed=42)  # type: ignore[assignment]  # MultiDiscrete is compatible with Space[int]

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction: dict[int, np.ndarray] = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]  # type: ignore[operator]  # metadata values are typed as object
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self._agent_location: npt.NDArray[np.int_] = np.zeros(2, dtype=int)
        self._target_location: npt.NDArray[np.int_] = np.zeros(2, dtype=int)

    def _get_obs(self) -> dict[str, npt.NDArray[np.int_]]:
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self) -> dict[str, float]:
        return {"distance": float(np.linalg.norm(self._agent_location - self._target_location, ord=1))}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, npt.NDArray[np.int_]], dict[str, float]]:
        # We need the following line to seed self.np_random
        _ = super().reset(seed=seed, options=options)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "animation":
            _ = self._render_frame()

        return observation, info

    def step(self, action: int) -> tuple[dict[str, npt.NDArray[np.int_]], int, bool, bool, dict[str, float]]:
        """Step in the environment according to the given action."""
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "animation":
            _ = self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> npt.NDArray[np.uint8] | None:  # type: ignore[override]  # NDArray is compatible with RenderFrame
        if self.render_mode == "data":
            return self._render_frame()
        return None

    def _render_frame(self) -> npt.NDArray[np.uint8] | None:
        if self.window is None and self.render_mode == "animation":
            _ = pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.size, self.size))
        if self.clock is None and self.render_mode == "animation":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.size, self.size))
        _ = canvas.fill((255, 255, 255))
        pix_square_size = self.size / self.size  # The size of a single grid square in pixels

        # First we draw the target
        _x, _y = pix_square_size * self._target_location
        _ = pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(_x, _y, pix_square_size, pix_square_size),
        )
        # Now we draw the agent
        _ = pygame.draw.circle(
            canvas,
            (0, 0, 255),
            tuple((self._agent_location + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            _ = pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.size, pix_square_size * x),
                width=3,
            )
            _ = pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.size),
                width=3,
            )

        if self.render_mode == "animation":
            # The following line copies our drawings from `canvas` to the visible window
            assert self.window is not None
            _ = self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])  # type: ignore[arg-type]  # metadata value is float at runtime
            return None

        # data mode
        if self.render_mode == "data":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

        return None

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
