"""The shared press-pull controller, plus a loader for controller configs saved in datasets."""
from parameter_estimation.controllers.press_pull_fsm import PressPullConfig, PressPullFSM, STATE_IDS  # noqa: F401

# Older datasets recorded wrist options that no longer exist: the wrist always
# holds a constant orientation. Accept those keys only at their no-op values.
_RETIRED = {'finger_pitch_deg': 0.0, 'rotate_with_arc': False}


def config_from_saved(saved: dict) -> PressPullConfig:
    for key, inert in _RETIRED.items():
        if saved.get(key, inert) != inert:
            raise ValueError(f'Saved run used {key}={saved[key]}; only constant wrist orientation is supported')
    return PressPullConfig(**{k: v for k, v in saved.items() if k not in _RETIRED})
