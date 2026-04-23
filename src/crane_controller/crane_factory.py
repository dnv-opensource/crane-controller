import numpy as np
from py_crane.crane import Crane


def build_crane(length: float = 10.0, mass: float = 1.0, q_factor: float = 50.0) -> Crane:
    crane = Crane()
    _ = crane.add_boom(
        "pedestal",
        description="A simple pole with same length as the wire",
        mass=100.0,
        boom=(length, 0.0, 0.0),
    )
    _ = crane.add_boom(
        "wire",
        description="The wire fixed to the pole. Flexible connection",
        mass=mass,
        mass_center=1.0,
        boom=(length, np.pi, 0.0),
        q_factor=q_factor,
    )
    crane.calc_statics_dynamics(None)
    return crane
