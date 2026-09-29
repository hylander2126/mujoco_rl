"""Sample exposed upper collision surfaces compatible with the fixed XZ arc.

MuJoCo collides mesh geoms as convex hulls. Sampling these hulls (rather than
STL visual triangles or bounding-box faces) matches that representation.
"""
from dataclasses import asdict, dataclass
from itertools import product

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class Candidate:
    index: int
    position: list[float]  # world surface point, not ball centre
    normal: list[float]
    object_position: list[float]  # payload body frame
    press_offset_xy: list[float]
    approach_error_m: float = 0.0
    joint_margin_rad: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def collision_hulls(model, data, body_id: int) -> list[ConvexHull]:
    """World hulls of box/mesh collision geoms; fail explicitly on other types."""
    hulls = []
    for gid in np.flatnonzero(model.geom_bodyid == body_id):
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        kind = model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.array(list(product([-1, 1], repeat=3))) * model.geom_size[gid]
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = model.mesh_vert[start:start + count]
        else:
            raise ValueError(f"Unsupported payload geom type: {kind}")
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        hulls.append(ConvexHull(world))
    if not hulls:
        raise ValueError("Payload has no supported collision geometry")
    return hulls


def upper_surface(hulls, xy, margin: float, min_normal: float):
    """Vertical ray intersection with the union of convex collision hulls."""
    hits = []
    for hull in hulls:
        planes = hull.equations
        upper = planes[:, 2] > 1e-8
        z = -(planes[upper, :2] @ xy + planes[upper, 3]) / planes[upper, 2]
        k = int(np.argmin(z))
        p = np.array([*xy, z[k]])
        if np.max(planes[:, :3] @ p + planes[:, 3]) > 1e-7:
            continue
        normal = planes[upper][k, :3]
        # Clearance from silhouette: every nearby XY probe must still hit hull.
        inset_ok = True
        for offset in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)):
            qxy = np.asarray(xy) + offset
            qz = np.min(-(planes[upper, :2] @ qxy + planes[upper, 3]) / planes[upper, 2])
            if np.max(planes[:, :3] @ np.array([*qxy, qz]) + planes[:, 3]) > 1e-7:
                inset_ok = False
        hits.append((p, normal, inset_ok))
    if not hits:
        return None
    p, normal, inset_ok = max(hits, key=lambda h: h[0][2])
    return (p, normal) if inset_ok and normal[2] >= min_normal else None


def unexpected_contacts(model, data, payload_id: int, ball_id: int) -> set[tuple[int, int]]:
    """Robot collisions except the intended ball/payload pair.

Robot bodies are identified by ancestry of joint_1's body, avoiding geom-name
heuristics. Static base contacts are outside that moving subtree.
"""
    root = int(model.jnt_bodyid[model.joint('joint_1').id])
    robot = {root}
    for bid in range(root + 1, model.nbody):
        if int(model.body_parentid[bid]) in robot:
            robot.add(bid)
    pairs = set()
    for contact in data.contact:
        g0, g1 = map(int, contact.geom)
        b0, b1 = map(int, model.geom_bodyid[[g0, g1]])
        if b0 not in robot and b1 not in robot:
            continue
        if (g0 == ball_id and b1 == payload_id) or (g1 == ball_id and b0 == payload_id):
            continue
        pairs.add(tuple(sorted((g0, g1))))
    return pairs


def generate_candidates(model, initial_data, count: int, geometry: dict, controller_config,
                        reference_points_xy: list[list[float]] | None = None) -> tuple[list[Candidate], dict]:
    """Generate a deterministic spread, filtering IK, collisions and pivot mismatch.

    Returns fewer than requested if geometry cannot supply enough valid points;
    rejected proposals and invalid scene geometry are reported separately.
    Optional reference points are proposed first, subject to every normal filter.
    """
    from dataclasses import replace
    from mujoco_irb120.robot.controllers.robot import controller
    from parameter_estimation.controllers.press_pull_fsm import PressPullFSM

    if count < 1:
        raise ValueError("Candidate count must be positive")
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, initial_data)
    irb = controller(model, data)
    fsm = PressPullFSM(irb, model, data, controller_config)
    top = fsm.object_top_center()
    hulls = collision_hulls(model, data, irb.payload_body_id)
    vertices = np.concatenate([h.points for h in hulls])
    lo, hi = vertices.min(0), vertices.max(0)
    pivot = data.site_xpos[irb.obj_frame_site].copy()
    if controller_config.arc_center_xz is not None:
        pivot[[0, 2]] = controller_config.arc_center_xz
        pivot[1] = (lo[1] + hi[1]) / 2
    report = {"requested": count, "rejected": {}, "bounds": [lo.tolist(), hi.tolist()],
              "pivot": pivot.tolist(), "hulls": [h.points[h.vertices].tolist() for h in hulls]}
    # This controller requires a world-Y support edge at the near-X side.
    bottom = vertices[vertices[:, 2] <= lo[2] + geometry['pivot_tolerance_m']]
    near = bottom[bottom[:, 0] <= bottom[:, 0].min() + geometry['pivot_tolerance_m']]
    valid_pivot = (abs(pivot[0] - near[:, 0].min()) <= geometry['pivot_tolerance_m']
                   and abs(pivot[2] - lo[2]) <= geometry['pivot_tolerance_m']
                   and near[:, 1].max() - near[:, 1].min() > 2 * geometry['edge_margin_m'])
    if not valid_pivot:
        report['scene_rejection'] = 'pivot_site_incompatible_with_near_x_support_edge'
        return [], report
    side = int(np.ceil(np.sqrt(count * geometry['proposal_multiplier'])))
    margin = geometry['edge_margin_m']
    if np.any(hi[:2] - lo[:2] <= 2 * margin):
        report['scene_rejection'] = 'surface_too_small'
        return [], report
    references = reference_points_xy or []
    if any(np.asarray(p).shape != (2,) or not np.isfinite(p).all() for p in references):
        raise ValueError("Reference points must be finite XY coordinates")
    proposals = [np.asarray(p, dtype=float) for p in references] + [np.array(top[:2])]
    proposals.extend(np.array(p) for p in product(*[
        np.linspace(lo[d] + margin, hi[d] - margin, side) for d in range(2)]))
    # Farthest-point order spreads even a small requested set across the surface.
    pool = np.asarray(proposals)
    order, distances = list(range(len(references) + 1)), np.full(len(pool), np.inf)
    for index in order[:-1]:
        distances = np.minimum(distances, np.linalg.norm(pool - pool[index], axis=1))
    for _ in range(len(pool) - len(order)):
        distances = np.minimum(distances, np.linalg.norm(pool - pool[order[-1]], axis=1))
        distances[order] = -1
        order.append(int(np.argmax(distances)))
    candidates = []
    for j in order:
        hit = upper_surface(hulls, pool[j], margin, geometry['min_upward_normal'])
        reason = None
        if hit is None:
            reason = 'surface_or_edge_clearance'
        else:
            p, normal = hit
            if p[0] <= pivot[0] or not (near[:, 1].min() + margin <= p[1] <= near[:, 1].max() - margin):
                reason = 'outside_pivot_span'
            elif (top[2] + controller_config.approach_clearance_m - p[2]
                  > controller_config.descend_speed * controller_config.speed_scale * controller_config.squash_timeout_sec):
                reason = 'surface_below_descent_range'
        if reason is None:
            mujoco.mj_copyData(data, model, initial_data)
            irb = controller(model, data)
            cfg = replace(controller_config, press_offset_xy=tuple(p[:2] - top[:2]))
            fsm = PressPullFSM(irb, model, data, cfg)
            try:
                fsm.move_to_pre_squash()
            except RuntimeError:
                reason = 'unreachable'
            else:
                mujoco.mj_forward(model, data)
                target = np.array([*p[:2], top[2] + cfg.approach_clearance_m])
                error = float(np.linalg.norm(data.site_xpos[irb.ball_site] - target))
                q = data.qpos[irb.joint_idx]
                joint_margin = float(np.min(np.minimum(q - irb.q_min, irb.q_max - q)))
                if error > geometry['approach_tolerance_m'] or joint_margin < 0:
                    reason = 'unreachable'
                elif unexpected_contacts(model, data, irb.payload_body_id, irb.ball_geom_id):
                    reason = 'approach_collision'
        if reason:
            report['rejected'][reason] = report['rejected'].get(reason, 0) + 1
            continue
        if any(np.linalg.norm(p - np.array(c.position)) < 1e-7 for c in candidates):
            continue
        local = initial_data.xmat[irb.payload_body_id].reshape(3, 3).T @ (p - initial_data.xpos[irb.payload_body_id])
        candidates.append(Candidate(len(candidates), p.tolist(), normal.tolist(), local.tolist(),
                                    list(cfg.press_offset_xy), error, joint_margin))
        if len(candidates) == count:
            break
    report['generated'] = len(candidates)
    return candidates, report
