import unittest

from aleph.qgeom.core import LayerGeometryReport, RunGeometryReport
from aleph.qgeom.policy import build_baseline_policy, score_correction_points


def make_report(layer_idx, score_seed):
    return LayerGeometryReport(
        layer_index=layer_idx,
        linear_error_norm=1.0 + score_seed,
        nonlinear_error_norm=0.1,
        saturation_count=0,
        relu_flip_rate=0.01,
        survive_rate=0.7,
        dead_rate=0.29,
        anisotropy_ratio=1.2,
        entropy_proxy=0.5 + score_seed,
        volume_proxy=0.2,
        correctability_score=0.8,
    )


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.run_report = RunGeometryReport(
            model_name="m",
            task_name="t",
            bit_width=4,
            layer_reports=[
                make_report(0, 0.1),
                make_report(1, 0.3),
                make_report(2, 0.2),
                make_report(3, 0.4),
            ],
        )

    def test_budget_respected(self):
        policy = score_correction_points(self.run_report, budget=2)
        self.assertEqual(len(policy["selected_points"]), 2)

    def test_ranking_deterministic(self):
        p1 = score_correction_points(self.run_report, budget=3)
        p2 = score_correction_points(self.run_report, budget=3)
        self.assertEqual(p1["selected_points"], p2["selected_points"])
        self.assertEqual(p1["ranking"], p2["ranking"])

    def test_min_gap_constraint(self):
        policy = score_correction_points(self.run_report, budget=3, constraints={"min_gap": 2})
        selected = policy["selected_points"]
        for i in range(1, len(selected)):
            self.assertGreaterEqual(abs(selected[i] - selected[i - 1]), 2)

    def test_baseline_policy_even_spacing(self):
        baseline = build_baseline_policy(self.run_report, budget=2)
        self.assertEqual(len(baseline["selected_points"]), 2)
        self.assertEqual(sorted(baseline["selected_points"]), baseline["selected_points"])


if __name__ == "__main__":
    unittest.main()
