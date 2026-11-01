"""Geometric route feedback must accept both directions and reject shortcut loops."""
import math
from pathlib import Path
import unittest

import numpy as np

from rl_training.common.config import RewardSettings
from rl_training.common.figure_eight_route import FigureEightRoute

WORLD = Path(__file__).resolve().parents[1]/'assets/worlds/simple/figure_eight_005.sdf'


class RouteTests(unittest.TestCase):
    def route(self):
        return FigureEightRoute(WORLD, RewardSettings(figure_eight_route_enabled=True), .2)

    def test_ideal_route_both_directions_and_starting_lobes(self):
        for direction in (-1, 1):
            for start in (0., .2, 1., math.pi, 3.5, 5.):
                r = self.route()
                t = np.linspace(start, start+direction*4*math.pi, 5001)
                xy = np.column_stack((r.a*np.sin(2*t), r.b*np.sin(t)))
                yaw = math.atan2(direction*r.b*math.cos(start), direction*2*r.a*math.cos(2*start))
                r.reset(xy[0], yaw)
                seconds = 0.
                for a, b in zip(xy, xy[1:]):
                    seconds += np.linalg.norm(b-a)/.2
                    state = r.update(seconds, b)
                    self.assertFalse(state['route_failed'], (direction, start, b, state))
                self.assertGreaterEqual(r.switches, 3)

    def test_turning_into_side_exit_fails(self):
        r = self.route();r.reset((-.7, .5), -.65)
        for i, p in enumerate([(-.4,.3),(-.1,.1),(-.3,-.2),(-.6,-.4)]):
            state = r.update(i+1, p)
        self.assertEqual(state['route_reason'], 'route_wrong_exit')
        self.assertLess(state['route_reward'], 0)

    def test_no_progress_timeout(self):
        r = self.route();r.reset((1.6,1.7), 1.5)
        state = r.update(r.timeout+.1, (1.6,1.7))
        self.assertEqual(state['route_reason'], 'route_no_progress')

    def test_short_time_limit_without_switch_is_not_success(self):
        r = self.route();r.reset((1.6,1.7), 1.5)
        state = r.update(25., (1.6,1.7), final=True)
        self.assertEqual(state['route_reason'], 'route_incomplete')
        self.assertLess(state['route_reward'], 0)

    def test_crossing_jitter_cannot_repeatedly_collect_switch_bonus(self):
        r = self.route();r.reset((-.7,.5), -.65)
        for i in range(100):
            state = r.update(i*.2, ((-.1 if i%2 else .1), .01))
            self.assertEqual(state['route_switches'], 0)

    def test_far_gate_bonus_requires_a_new_crossing(self):
        r = self.route();r.reset((-.7,.5), -.65)
        for i, p in enumerate([(-.4,.3),(0.,0.),(.6,-.45),(1.,-1.5)]):
            state = r.update(i+1, p)
        self.assertEqual(state['route_switches'], 1)
        for i in range(10):
            state = r.update(i+5, (1., -1.4 if i%2 else -1.5))
            self.assertEqual(state['route_switches'], 1)
            self.assertEqual(state['route_reward'], 0.)

    def test_opt_in_and_map_scope(self):
        self.assertIsNone(FigureEightRoute.for_world(WORLD, RewardSettings(), .2))
        self.assertIsNone(FigureEightRoute.for_world(WORLD.with_name('rectangle_001.sdf'),
                         RewardSettings(figure_eight_route_enabled=True), .2))


if __name__ == '__main__':
    unittest.main()
