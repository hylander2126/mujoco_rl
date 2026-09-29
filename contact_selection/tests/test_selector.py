"""Protect scenario alignment, held-out grouping and abstention semantics."""
from pathlib import Path

import numpy as np
import pytest

from contact_selection.candidate_generator import Candidate
from contact_selection.dataset import append_record, write_json
from contact_selection.features import FEATURE_NAMES
from contact_selection.selector import (fit_logistic, load_robust_contacts,
                                        predict_and_select, score_contacts, select_contact)


def _record(name, split, scene, config, index, feasible):
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    features['pivot_dx_m'] = 0.01 + index * 0.05
    features['width_m'] = 0.1
    return {'object_name': name, 'object_split': split, 'candidate_set_id': scene,
            'config_id': config, 'candidate': {'index': index, 'press_offset_xy': [index, 0]},
            'features': features, 'feasible': feasible}


def _scene(root: Path, name: str, config: str, labels: list[bool]):
    scene = f'{name}_0'
    rows = [_record(name, 'train', scene, config, index, label)
            for index, label in enumerate(labels)]
    path = root / scene
    path.mkdir(parents=True)
    write_json(path / 'scene.json', {'candidate_set_id': scene,
                                     'candidates': [row['candidate'] for row in rows]})
    for row in rows:
        append_record(root / 'rollouts.jsonl', row)


def test_robust_labels_align_scenarios_and_fit(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    _scene(first, 'box', 'a', [True, True])
    _scene(first, 'heart', 'a', [True, False])
    _scene(second, 'box', 'b', [True, False])
    contacts, scenarios = load_robust_contacts([first, second])
    box = [row for row in contacts if row['object'] == 'box']
    assert [row['robust_feasible'] for row in box] == [True, False]
    assert all(row['scenario_count'] == 2 for row in box)
    assert len(scenarios['box']) == 2
    model = fit_logistic(contacts)
    scores = score_contacts(model, box)
    assert np.isfinite(scores).all()
    assert select_contact(box, scores, 0.0, np.array([11.0, 12.0]), 10.0) is None


def test_incomplete_scene_is_rejected(tmp_path):
    tmp_path.joinpath('run').mkdir()
    root = tmp_path / 'run'
    _scene(root, 'box', 'a', [True, False])
    (root / 'rollouts.jsonl').write_text((root / 'rollouts.jsonl').read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='Incomplete candidate set'):
        load_robust_contacts([root])


def test_preaction_api_selects_or_abstains_without_outcomes():
    model = {'feature_names': FEATURE_NAMES, 'mean': [0.0] * len(FEATURE_NAMES),
             'scale': [1.0] * len(FEATURE_NAMES), 'weights': [0.0] * len(FEATURE_NAMES),
             'intercept': 0.0, 'max_feature_distance': 10.0}
    candidate = Candidate(0, [0.01, 0.0, 1.0], [0.0, 0.0, 1.0],
                          [0.0, 0.0, 0.0], [0.0, 0.0])
    geometry = {'bounds': [[0, -0.05, 0], [0.1, 0.05, 1]], 'pivot': [0, 0, 0]}
    chosen = predict_and_select(model, [candidate], geometry, threshold=0.4)
    assert chosen['selected_candidate']['index'] == 0
    assert chosen['scores'][0]['eligible']
    skipped = predict_and_select(model, [candidate], geometry, threshold=0.6)
    assert skipped['selected_candidate'] is None and skipped['reason'] == 'low_score'
    assert predict_and_select(model, [], geometry)['reason'] == 'no_valid_candidates'
