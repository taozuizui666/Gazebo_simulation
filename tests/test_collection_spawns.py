"""Collection starts must be reproducible, diverse and clear of map obstacles."""

import unittest
from unittest.mock import patch

from my_gazebo_launch.config import load_config
from my_gazebo_launch.world_config import load_world_settings
from rl_training.privileged import collect_data as collector
from rl_training.privileged.collection_spawns import build_spawn_plan
from rl_training.privileged.observation import GroundTruthMap


class CollectionSpawnTests(unittest.TestCase):
    def setUp(self):
        self.world = load_world_settings(collector.WORLDS[0])
        self.geometry = GroundTruthMap(self.world.world)
        self.vehicle = load_config(collector.MODEL.resolve().parent / "vehicle_config.h")
        self.settings = collector.spawn_settings(0.2)

    def plan(self, **changes):
        return build_spawn_plan(self.world, self.geometry, self.vehicle, 20, 42,
                                {**self.settings, **changes})

    def test_four_maps_have_safe_diverse_starts_and_exact_recovery_count(self):
        for index, path in enumerate(collector.WORLDS):
            world = load_world_settings(path)
            geometry = GroundTruthMap(path)
            plan = build_spawn_plan(world, geometry, self.vehicle, 20, 42 + index, self.settings)
            self.assertEqual(sum(s["kind"] == "near_wall" for s in plan), 5)
            self.assertEqual(len({tuple(s["pose"]) for s in plan}), 20)
            for entry in plan:
                x, y, z, yaw = entry["pose"]
                radius = geometry.settings.body_clearance_radius_m + self.settings["margin_m"]
                self.assertGreater(float(geometry.signed_distance([[x, y]])[0]), radius)
                xmin, xmax, ymin, ymax = world.workspace_bounds
                self.assertTrue(xmin + radius < x < xmax - radius)
                self.assertTrue(ymin + radius < y < ymax - radius)
                if entry["kind"] == "near_wall":
                    self.assertTrue(0.04 <= entry["clearance_m"] <= 0.08)
                    self.assertTrue(0.15 <= entry["front_clearance_m"] <= 0.4)
                else:
                    base = world.eval_spawn_poses[entry["base_spawn_index"]]
                    self.assertLessEqual(abs(x - base[0]), 0.03)
                    self.assertLessEqual(abs(y - base[1]), 0.03)

    def test_seed_replays_exactly_and_isolated_from_other_random_calls(self):
        import random
        first = self.plan()
        random.seed(999)
        random.random()
        self.assertEqual(first, self.plan())
        other = build_spawn_plan(self.world, self.geometry, self.vehicle, 20, 43, self.settings)
        self.assertNotEqual(first, other)

    def test_both_switches_off_restore_original_poses(self):
        with patch.object(collector, "SPAWN_JITTER_ENABLED", False), \
                patch.object(collector, "RECOVERY_ENABLED", False):
            settings = collector.spawn_settings(0.2)
        plan = build_spawn_plan(self.world, self.geometry, self.vehicle, 20, 42, settings)
        for index, entry in enumerate(plan):
            self.assertEqual(entry["pose"], list(self.world.eval_spawn_poses[index % len(self.world.eval_spawn_poses)]))
            self.assertEqual(entry["kind"], "fixed")

    def test_impossible_near_wall_range_fails_without_fixed_spawn_fallback(self):
        with self.assertRaisesRegex(ValueError, "无法生成近墙"):
            self.plan(recovery_fraction=1.0, recovery_clearance_m=[5.0, 6.0], attempts=3)

    def test_invalid_parameters_fail_before_sampling(self):
        for change in ({"recovery_fraction": 1.1}, {"position_jitter_m": float("nan")},
                       {"attempts": 0}, {"recovery_yaw_deg": [10, 90]},
                       {"recovery_clearance_m": [0.01, 0.08]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(**change)


if __name__ == "__main__":
    unittest.main()
