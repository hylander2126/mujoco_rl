"""Geometry-only logistic baseline on aligned, object-grouped contact outcomes.

Each candidate's target is success in every supplied physical scenario for its
object. Scores are sigmoid outputs, not calibrated probabilities of hardware
success. Whole objects, never neighboring contacts, define the data splits.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from contact_selection.dataset import read_records, write_json
from contact_selection.features import FEATURE_NAMES, extract_features

# Physical floors stop near-constant training dimensions (for example box and
# heart height) from amplifying tiny mesh differences by millions at inference.
SCALE_FLOORS = {
    **{name: 0.02 for name in ('local_x_m', 'local_y_m', 'local_z_m',
                               'pivot_dx_m', 'pivot_dz_m', 'arc_radius_m',
                               'width_m', 'depth_m', 'height_m')},
    **{name: 0.1 for name in ('normal_x', 'normal_y', 'normal_z')},
    'normalized_height': 0.05,
    'approach_error_m': 0.002,
    'joint_margin_rad': 0.05,
    'pivot_ray_angle_rad': 0.02,
}


def load_robust_contacts(directories: list[Path]) -> tuple[list[dict], dict]:
    """Align candidates across saved scenarios and take a conservative AND label."""
    by_object = defaultdict(list)
    for directory in directories:
        rows = read_records(directory / 'rollouts.jsonl')
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row['candidate_set_id']].append(row)
        manifests = list(directory.glob('*/scene.json'))
        if not manifests:
            raise ValueError(f'No scene manifests in {directory}')
        manifest_ids = set()
        for path in manifests:
            manifest = json.loads(path.read_text())
            scene_id = manifest['candidate_set_id']
            manifest_ids.add(scene_id)
            if len(by_scene.get(scene_id, [])) != len(manifest['candidates']):
                raise ValueError(f'Incomplete candidate set in {directory}/{scene_id}')
            if not manifest['candidates']:
                raise ValueError(f'Empty candidate set in {directory}/{scene_id}; report coverage separately')
        if set(by_scene) != manifest_ids:
            raise ValueError(f'Rollout scenes and manifests differ in {directory}')
        for scene_id, scene_rows in by_scene.items():
            scene_rows.sort(key=lambda row: row['candidate']['index'])
            indices = [row['candidate']['index'] for row in scene_rows]
            if indices != list(range(len(scene_rows))):
                raise ValueError(f'Incomplete or duplicate candidate indices in {directory}/{scene_id}')
            first = scene_rows[0]
            if any(row['object_name'] != first['object_name'] or
                   row['object_split'] != first['object_split'] for row in scene_rows):
                raise ValueError('Mixed object identities inside a candidate set')
            by_object[first['object_name']].append((directory, scene_id, scene_rows))
    if not by_object:
        raise ValueError('No rollout records supplied')
    contacts = []
    scenarios = {}
    for name, groups in sorted(by_object.items()):
        reference = groups[0][2]
        signature = [row['candidate'] for row in reference]
        split = reference[0]['object_split']
        for directory, scene_id, rows in groups:
            if [row['candidate'] for row in rows] != signature:
                raise ValueError(f'Candidate geometry/order differs across {name} scenarios')
            if any(row['object_split'] != split for row in rows):
                raise ValueError(f'Object {name} appears in multiple data splits')
            if any(row['features'] != ref['features'] for row, ref in zip(rows, reference)):
                raise ValueError(f'Pre-action features differ across {name} scenarios')
        scenarios[name] = [{'directory': str(directory), 'candidate_set_id': scene_id,
                            'config_id': rows[0]['config_id']}
                           for directory, scene_id, rows in groups]
        for index, row in enumerate(reference):
            contacts.append({'object': name, 'split': split, 'candidate': row['candidate'],
                             'features': row['features'],
                             'robust_feasible': all(rows[index]['feasible'] for _, _, rows in groups),
                             'scenario_count': len(groups)})
    return contacts, scenarios


def feature_matrix(contacts: list[dict], names: list[str] = FEATURE_NAMES) -> np.ndarray:
    matrix = np.asarray([[row['features'][key] for key in names] for row in contacts], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Candidate features must be finite')
    return matrix


def fit_logistic(contacts: list[dict], l2: float = 0.1,
                 feature_names: list[str] = FEATURE_NAMES) -> dict:
    """Fit on a subset of FEATURE_NAMES (default all); ablations use older subsets."""
    _check_feature_names(feature_names)
    train = [row for row in contacts if row['split'] == 'train']
    if len({row['object'] for row in train}) < 2:
        raise ValueError('Training requires at least two distinct object geometries')
    y = np.asarray([row['robust_feasible'] for row in train], dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError('Training requires both feasibility classes')
    x = feature_matrix(train, feature_names)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), [SCALE_FLOORS[key] for key in feature_names])
    x = (x - mean) / scale
    counts = defaultdict(int)
    for row in train:
        counts[row['object']] += 1
    weights = np.asarray([1 / counts[row['object']] for row in train])
    weights /= weights.sum()

    def objective(theta):
        logits = x @ theta[:-1] + theta[-1]
        loss = np.dot(weights, np.logaddexp(0, logits) - y * logits) + 0.5 * l2 * np.dot(theta[:-1], theta[:-1])
        residual = weights * (expit(logits) - y)
        gradient = np.r_[x.T @ residual + l2 * theta[:-1], residual.sum()]
        return float(loss), gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'Logistic fit failed: {result.message}')
    return {'feature_names': list(feature_names), 'mean': mean.tolist(), 'scale': scale.tolist(),
            'weights': result.x[:-1].tolist(), 'intercept': float(result.x[-1]),
            'l2': l2, 'train_objects': sorted(counts), 'score_calibrated': False,
            'max_feature_distance': 10.0,
            'target': 'feasible in every supplied scenario for this object'}


def _check_feature_names(names: list[str]) -> None:
    # Order and uniqueness matter: weights are stored positionally.
    if not names or len(set(names)) != len(names) or not set(names) <= set(FEATURE_NAMES):
        raise ValueError(f'Feature schema must be distinct names from FEATURE_NAMES: {names}')


def score_contacts(model: dict, contacts: list[dict]) -> np.ndarray:
    _check_feature_names(model['feature_names'])
    x = (feature_matrix(contacts, model['feature_names']) - model['mean']) / model['scale']
    return expit(x @ np.asarray(model['weights']) + model['intercept'])


def feature_distances(model: dict, contacts: list[dict]) -> np.ndarray:
    """Largest standardized feature deviation from the training mean."""
    x = (feature_matrix(contacts, model['feature_names']) - model['mean']) / model['scale']
    return np.max(np.abs(x), axis=1)


def select_contact(contacts: list[dict], scores: np.ndarray, threshold: float = 0.5,
                   distances: np.ndarray | None = None, max_distance: float = 10.0) -> int | None:
    """Return the best in-range candidate above threshold, or abstain."""
    if len(contacts) != len(scores) or not 0 <= threshold <= 1 or max_distance <= 0:
        raise ValueError('Invalid scores, selection threshold or distance limit')
    eligible = np.asarray(scores) >= threshold
    if distances is not None:
        if len(distances) != len(contacts):
            raise ValueError('Feature distances must match candidates')
        eligible &= np.asarray(distances) <= max_distance
    eligible = np.flatnonzero(eligible)
    return int(eligible[np.argmax(np.asarray(scores)[eligible])]) if len(eligible) else None


def predict_and_select(model: dict, candidates: list, geometry: dict,
                       threshold: float = 0.5) -> dict:
    """Score pre-action candidate geometry and select one, or abstain.

    This API accepts the candidate generator's outputs and never reads rollout
    labels or simulator ground-truth physical parameters. Scores are uncalibrated.
    """
    if not candidates:
        return {'selected_candidate': None, 'reason': 'no_valid_candidates', 'scores': []}
    rows = [{'candidate': candidate.to_dict(), 'features': extract_features(candidate, geometry)}
            for candidate in candidates]
    scores = score_contacts(model, rows)
    distances = feature_distances(model, rows)
    selected = select_contact(rows, scores, threshold, distances, model['max_feature_distance'])
    reason = None if selected is not None else (
        'geometry_out_of_range' if np.all(distances > model['max_feature_distance'])
        else 'low_score')
    return {'selected_candidate': rows[selected]['candidate'] if selected is not None else None,
            'reason': reason,
            'scores': [{'candidate_index': row['candidate']['index'], 'score': float(score),
                        'feature_distance': float(distance),
                        'eligible': bool(score >= threshold and distance <= model['max_feature_distance'])}
                       for row, score, distance in zip(rows, scores, distances)]}


def evaluate(model: dict, contacts: list[dict], threshold: float = 0.5) -> dict:
    by_object = defaultdict(list)
    for row in contacts:
        by_object[row['object']].append(row)
    scenes = []
    for name, rows in sorted(by_object.items()):
        scores = score_contacts(model, rows)
        distances = feature_distances(model, rows)
        labels = np.asarray([row['robust_feasible'] for row in rows], dtype=bool)
        selected = select_contact(rows, scores, threshold, distances, model['max_feature_distance'])
        abstain_reason = None if selected is not None else (
            'geometry_out_of_range' if np.all(distances > model['max_feature_distance'])
            else 'low_score')
        center = int(np.argmin([np.linalg.norm(row['candidate']['press_offset_xy']) for row in rows]))
        scenes.append({'object': name, 'split': rows[0]['split'], 'candidates': len(rows),
                       'robust_feasible': int(labels.sum()), 'random_expected_success': float(labels.mean()),
                       'oracle_success': bool(labels.any()), 'center_success': bool(labels[center]),
                       'selected_index': rows[selected]['candidate']['index'] if selected is not None else None,
                       'selected_success': bool(labels[selected]) if selected is not None else None,
                       'selected_score': float(scores[selected]) if selected is not None else None,
                       'abstain_reason': abstain_reason,
                       'brier_score': float(np.mean((scores - labels.astype(float)) ** 2)),
                       'candidate_scores': [{'index': row['candidate']['index'], 'score': float(score),
                                             'feature_distance': float(distance),
                                             'robust_feasible': bool(label)}
                                            for row, score, distance, label in zip(rows, scores, distances, labels)]})
    return {'threshold': threshold, 'score_calibrated': False, 'scenes': scenes}


def train_and_save(directories: list[Path], output: Path, threshold: float = 0.5,
                   feature_names: list[str] = FEATURE_NAMES) -> dict:
    contacts, scenarios = load_robust_contacts(directories)
    model = fit_logistic(contacts, feature_names=feature_names)
    model['scenarios_by_object'] = scenarios
    report = evaluate(model, contacts, threshold)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'model.json', model)
    write_json(output / 'evaluation.json', report)
    return report
