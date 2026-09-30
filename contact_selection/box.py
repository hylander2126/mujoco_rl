"""Centered box preset using the existing shared demo's physical setup."""
from dataclasses import asdict, dataclass, replace
import mujoco
import numpy as np
from parameter_estimation.press_pull_demo import (
    BoxDemoConfig as SharedBoxConfig, prepare_box as prepare_shared_box,
)
from contact_selection.controller import PressPullConfig
from contact_selection.scene import disable_adapter_object_collisions


@dataclass(frozen=True)
class BoxDemoConfig(SharedBoxConfig):
    object_y_m: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.object_y_m):
            raise ValueError('Object Y must be finite')


def prepare_box(config: BoxDemoConfig, verbose: bool = True):
    shared = SharedBoxConfig(**{key: value for key, value in asdict(config).items()
                               if key != 'object_y_m'})
    model, data, candidate, base_cfg, metadata = prepare_shared_box(shared, verbose)
    payload = int(model.site_bodyid[model.site('site:obj_frame').id])
    adr = int(model.jnt_qposadr[int(model.body_jntadr[payload])])
    shift_y = config.object_y_m - data.qpos[adr + 1]
    model.body_pos[payload, 1] = config.object_y_m
    model.qpos0[adr + 1] = model.qpos_spring[adr + 1] = config.object_y_m
    data.qpos[adr + 1] = config.object_y_m
    disable_adapter_object_collisions(model)
    mujoco.mj_forward(model, data)
    point = np.asarray(candidate.position).copy()
    point[1] += shift_y
    candidate = replace(candidate, position=point.tolist())
    cfg = PressPullConfig(**asdict(base_cfg))
    metadata.update(preset=asdict(config), controller=asdict(cfg), candidate=candidate.to_dict(),
                    pivot_world_m=data.site_xpos[model.site('site:obj_frame').id].copy(),
                    collision_policy='adapter_object_disabled',
                    geom_contype=model.geom_contype.copy(), geom_conaffinity=model.geom_conaffinity.copy())
    return model, data, candidate, cfg, metadata
