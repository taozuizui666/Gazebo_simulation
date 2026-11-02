"""Regression checks for RF acceptance false positives; no Gazebo startup."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_rf_acceptance as acceptance


class AcceptanceTests(unittest.TestCase):
    def test_real_s_map_uses_its_sibling_metadata(self):
        path = acceptance.ROOT / acceptance.S_MAP
        meta, points = acceptance.centerline_geometry(path)
        self.assertEqual(meta['template'], 's_bend')
        np.testing.assert_array_equal(points, meta['reference_centerline'])

    def test_legacy_convex_s_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '真正 S 弯'):
            acceptance.centerline_geometry(acceptance.ROOT / 'assets/worlds/simple/s_bend_002.sdf')

    def test_wall_between_ticks_is_detected(self):
        class Geometry:
            class settings:
                body_clearance_radius_m = .085

            def signed_distance(self, points):
                return np.abs(points[:, 0])

        trace = np.array([[0, -1, 0, 0], [.2, 1, 0, 0]])
        metrics = acceptance.route_metrics(
            acceptance.ROOT / 'assets/worlds/simple/rectangle_001.sdf', trace, Geometry(), .2)
        self.assertLess(metrics['min_clearance_m'], 0)

    def test_time_limit_alone_does_not_pass(self):
        metrics = dict(min_clearance_m=.02, route_qualified=True)
        info = dict(episode_sim_time_sec=300, speed_compliance=1.)
        self.assertTrue(acceptance.qualification(metrics, info, True)['fully_qualified'])
        for field, value in [('episode_sim_time_sec', 299), ('speed_compliance', .94)]:
            with self.subTest(field=field):
                self.assertFalse(acceptance.qualification(metrics, dict(info, **{field: value}), True)['fully_qualified'])
        self.assertFalse(acceptance.qualification(dict(metrics, route_qualified=False), info, True)['fully_qualified'])
        self.assertFalse(acceptance.qualification(dict(metrics, min_clearance_m=-.001), info, True)['fully_qualified'])

    def test_failed_or_incomplete_suite_returns_failure(self):
        self.assertEqual(acceptance.acceptance_exit_code([dict(fully_qualified=True)], 1), 0)
        for episodes, expected in [([], 0), ([], 8), ([dict(fully_qualified=True)], 8),
                                   ([dict(fully_qualified=False)], 1)]:
            self.assertEqual(acceptance.acceptance_exit_code(episodes, expected), 1)


if __name__ == '__main__':
    unittest.main()
