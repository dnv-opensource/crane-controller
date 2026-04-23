import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class DiscreteActions(gym.ActionWrapper[object, int, np.ndarray]):
    """Gymnasium action wrapper mapping discrete indices to continuous arrays."""

    def __init__(self, env: gym.Env[object, object], disc_to_cont: list[np.ndarray]) -> None:
        """Initialize the discrete-to-continuous action wrapper.

        Parameters
        ----------
        env : gym.Env[object, object]
            The environment to wrap.
        disc_to_cont : list[np.ndarray]
            Lookup table mapping discrete action indices to continuous arrays.
        """
        super().__init__(env)  # type: ignore[arg-type]  # gymnasium Env type params are invariant
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, action: int) -> np.ndarray:
        """Map a discrete action index to a continuous action array.

        Parameters
        ----------
        action : int
            Discrete action index.

        Returns
        -------
        np.ndarray
            Corresponding continuous action from the lookup table.
        """
        return self.disc_to_cont[action]
