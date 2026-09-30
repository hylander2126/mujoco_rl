"""Use the shared fixed press-pull controller; orientation search is retired."""
from dataclasses import dataclass
from parameter_estimation.controllers.press_pull_fsm import (
    PressPullConfig as SharedPressPullConfig, PressPullFSM, STATE_IDS,
)


@dataclass
class PressPullConfig(SharedPressPullConfig):
    finger_pitch_deg: float = 0.0

    def __post_init__(self):
        if self.finger_pitch_deg != 0:
            raise ValueError('Contact selection uses zero initial pitch; orientation search is retired')
