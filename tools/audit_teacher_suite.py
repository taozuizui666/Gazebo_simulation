#!/usr/bin/env python3
"""Audit saved teacher trajectories against a frozen suite protocol; no simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_maps import centerline
from rl_training.privileged.observation import GroundTruthMap


def track_metrics(trace, points, geometry):
    points = np.asarray(points)
    vector = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vector, axis=1)
    starts = np.r_[0., np.cumsum(lengths)[:-1]]
    # Sample between control ticks so a traversed bin cannot be skipped merely
    # because it lies between two recorded positions.
    positions = np.concatenate([
        np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/.01)) + 1))
        for a, b in zip(trace[:-1, 1:3], trace[1:, 1:3])])
    phases, offsets = [], []
    for chunk in np.array_split(positions, max(1, len(positions)//500)):
        delta = chunk[:, None, :] - points
        t = np.clip((delta * vector).sum(axis=2) / lengths**2, 0, 1)
        distances = np.linalg.norm(delta - t[:, :, None] * vector, axis=2)
        index = distances.argmin(axis=1)
        rows = np.arange(len(chunk))
        phases.extend((starts[index] + t[rows, index]*lengths[index])/lengths.sum())
        offsets.extend(distances[rows, index])
    phases = np.asarray(phases)
    bins = np.unique(np.minimum((phases * 100).astype(int), 99))
    unwrapped = np.unwrap(phases * 2*np.pi)/(2*np.pi)
    clearance = geometry.signed_distance(positions) - geometry.settings.body_clearance_radius_m
    return dict(coverage_bins=len(bins), net_laps=float(unwrapped[-1]-unwrapped[0]),
                min_clearance_m=float(clearance.min()), max_centerline_offset_m=float(max(offsets)))


def audit(root):
    protocol = json.loads((root/'protocol.json').read_text())
    expected = {(w['sha256'], s) for w in protocol['worlds'] for s in protocol['seeds']}
    seen, rows, errors = set(), [], []
    for report in sorted(root.glob('*/evaluation_*/evaluation.json')):
        data = json.loads(report.read_text())
        key = (data['world_sha256'], data['seed'])
        if key not in expected:
            continue
        if key in seen:
            errors.append(f'Duplicate report: {report}')
        seen.add(key)
        world = Path(data['world'])
        if hashlib.sha256(world.read_bytes()).hexdigest() != key[0]:
            errors.append(f'World changed: {world}')
        if data['model_sha256'] != protocol['model_sha256']:
            errors.append(f'Model mismatch: {report}')
        meta = json.loads(world.with_suffix('.json').read_text())
        p = meta['parameters']
        points = meta.get('reference_centerline')
        if points is None:
            if meta['template'] == 's_bend':
                raise ValueError('Legacy convex S map cannot establish S-bend competence')
            points = [xy for xy, normal in centerline(meta['template'], p['width_m'], p['height_m'], p['corner_radius_m'])]
        geometry = GroundTruthMap(world)
        traces = np.load(report.with_name('trajectories.npz'))
        if len(data['episodes']) != protocol['episodes_per_report'] or {e['scenario'] for e in data['episodes']} != set(range(8)):
            errors.append(f'Incomplete scenarios: {report}')
        report_metrics = []
        for i, entry in enumerate(data['episodes']):
            trace = traces[f'episode_{i}']
            metrics = track_metrics(trace, points, geometry)
            element_ok = True
            lobe_switches = 0
            if meta['template'] == 'rectangle':
                corners = np.array([[sx*p['width_m']/2, sy*p['height_m']/2] for sx in (-1, 1) for sy in (-1, 1)])
                element_ok = bool((np.linalg.norm(trace[:, None, 1:3]-corners, axis=2).min(axis=0) < .4).all())
            elif meta['template'] == 'figure_eight':
                lobes = np.where(trace[:, 2] > p['height_m']*.2, 1,
                                 np.where(trace[:, 2] < -p['height_m']*.2, -1, 0))
                lobes = lobes[lobes != 0]
                lobe_switches = int(np.count_nonzero(np.diff(lobes)))
                element_ok = bool(trace[:, 2].min() < -p['height_m']*.3 and trace[:, 2].max() > p['height_m']*.3
                                  and (np.linalg.norm(trace[:, 1:3], axis=1)<.3).any()
                                  and lobe_switches >= protocol['min_figure_eight_lobe_switches'])
                if protocol.get('require_figure_eight_route', False):
                    element_ok = (element_ok
                                  and data.get('task_reward_settings', {}).get('figure_eight_route_enabled') is True
                                  and not entry.get('route_failed', True)
                                  and entry.get('route_switches', 0) >= protocol['min_figure_eight_lobe_switches'])
            flags = ('collided', 'stuck', 'looping', 'slow', 'out_of_bounds')
            passed = (entry['success'] and entry['reason']=='time_limit' and not any(entry[f] for f in flags)
                      and entry['sim_duration_sec'] >= protocol['seconds_per_episode']-1e-6
                      and entry['speed_compliance'] >= protocol['min_speed_compliance']
                      and len(trace)==entry['steps']+1 and np.isfinite(trace).all()
                      and metrics['coverage_bins']>=protocol['min_centerline_coverage_bins']
                      and metrics['min_clearance_m']>=protocol['min_conservative_clearance_m']
                      and (meta['template']=='figure_eight' or abs(metrics['net_laps'])>=protocol['min_net_laps_non_crossing'])
                      and element_ok)
            row = dict(map=world.parent.name+'/'+world.stem, seed=data['seed'], scenario=entry['scenario'],
                       passed=bool(passed), **metrics, elements_passed=element_ok, lobe_switches=lobe_switches,
                       route_switches=entry.get('route_switches', 0),
                       sim_seconds=entry['sim_duration_sec'], wall_seconds=entry['wall_duration_sec'],
                       speed_compliance=entry['speed_compliance'], report=str(report.resolve()))
            rows.append(row); report_metrics.append(row)
        report.with_name('trajectory_metrics.json').write_text(json.dumps(report_metrics, indent=2)+'\n')
    missing = expected-seen
    result = dict(passed=not errors and not missing and bool(rows) and all(r['passed'] for r in rows),
                  completed_reports=len(seen), expected_reports=len(expected), episodes=len(rows),
                  passed_episodes=sum(r['passed'] for r in rows), missing=list(missing), errors=errors,
                  criteria=protocol, episodes_detail=rows)
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    if rows:
        with (root/'audit_episodes.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes_detail', 'criteria')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.root)['passed'] else 1)
