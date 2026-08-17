"""Golden tests: the scoring pipeline must keep producing the same numbers.

These exist to make the planned refactor safe. Splitting `task.py` and breaking
up the runner should move code without changing a single reported figure, and
the only way to know that is to pin the figures first.

Two levels:

  synthetic   self-contained, deterministic, no dataset needed. Runs everywhere.
  real data   scores a saved checkpoint against the report it originally
              produced. Skipped when the fixtures are absent.
"""

import json
import os
import pickle
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_ueba import scoring  # noqa: E402

CPU = torch.device("cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HalfModel(torch.nn.Module):
    """Reconstructs half the input, so the error is analytic and stable.

    A trained network would tie the expected values to a torch version and its
    RNG; this keeps the fixture about the pipeline rather than about torch.
    """

    def forward(self, x):
        return x * 0.5


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


def synthetic_frame():
    """80 users, 30 days, 4 features. The first 10 carry a mild anomaly.

    Anomaly strength rises across those users so the scores spread out and the
    metrics land mid-range; a saturated fixture would pass for any implementation
    that is roughly correct.
    """
    rng = np.random.RandomState(11)
    rows = []
    for user in range(80):
        anomalous = user < 10
        strength = 0.15 + 0.05 * user if anomalous else 0.0
        for day in range(30):
            values = rng.rand(4) * 0.6
            if anomalous and 10 <= day < 13:
                values += strength
            rows.append({
                "user": float(user), "day": day,
                "insider": 1 if (anomalous and day == 11) else 0,
                **{f"f{i}": values[i] for i in range(4)},
            })
    return pd.DataFrame(rows)


SYNTHETIC_CONFIG = scoring.ScoringConfig(
    top_k_features=2, persistence_window=3, diversity_threshold=1.0,
    scan_stride=1, inference_batch_size=64)

# Captured 2026-07-29 from the pipeline as it stands. A diff here after a
# refactor means behaviour changed, not just structure.
EXPECTED = {
    "score_sum": 70.8714102149,
    "first_user_score": 0.7917517029,
    "last_user_score": 0.5421494498,
    "pr_auc_all": 0.9284615385,
    "pr_auc_val": 1.0,
    "pr_auc_test": 0.8769230769,
    "f1": 0.8,
    "precision": 0.8,
    "recall": 0.8,
    "threshold": 1.3131220542,
    "tp": 4, "fp": 1, "tn": 34, "fn": 1,
}


class TestSyntheticGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = synthetic_frame()
        features = [f"f{i}" for i in range(4)]
        scorer = scoring.Scorer(
            features=features, scaler=IdentityScaler(),
            ref_mean=np.full(4, 0.02), ref_std=np.full(4, 0.01),
            cfg=SYNTHETIC_CONFIG, window_size=14, device=CPU)
        cls.results = scorer.scan(HalfModel(), df, sorted(df["user"].unique()))
        cls.metrics = scoring.evaluate_scores(cls.results, seed=42)

    def test_per_user_scores_are_unchanged(self):
        self.assertAlmostEqual(float(self.results["max_z_score"].sum()),
                               EXPECTED["score_sum"], places=6)
        self.assertAlmostEqual(float(self.results["max_z_score"].iloc[0]),
                               EXPECTED["first_user_score"], places=6)
        self.assertAlmostEqual(float(self.results["max_z_score"].iloc[-1]),
                               EXPECTED["last_user_score"], places=6)

    def test_reported_metrics_are_unchanged(self):
        for key in ("pr_auc_all", "pr_auc_val", "pr_auc_test",
                    "f1", "precision", "recall", "threshold"):
            self.assertAlmostEqual(self.metrics[key], EXPECTED[key], places=6,
                                   msg=f"{key} drifted")

    def test_confusion_matrix_is_unchanged(self):
        for key in ("tp", "fp", "tn", "fn"):
            self.assertEqual(self.metrics[key], EXPECTED[key], f"{key} drifted")

    def test_fixture_is_not_saturated(self):
        """A fixture scoring 1.0 everywhere would not detect a regression."""
        self.assertLess(self.metrics["pr_auc_test"], 0.95)
        self.assertGreater(self.metrics["pr_auc_test"], 0.5)


BASELINE_WEIGHTS = os.path.join(ROOT, "model_pickle", "baseline")
BASELINE_SCALERS = os.path.join(ROOT, "scaler_data", "baseline")
ABLATION_REPORT = os.path.join(ROOT, "federated_evaluation_reports",
                               "ablation-full", "federated_rounds_comparison.csv")
DATA_PATH = os.path.join(ROOT, "ExtractedData", "dayr4.2-percentile30.csv")

_FIXTURES = all(os.path.exists(p) for p in
                (BASELINE_WEIGHTS, BASELINE_SCALERS, ABLATION_REPORT, DATA_PATH))


@unittest.skipUnless(_FIXTURES, "archived baseline run not present")
class TestRealDataRegression(unittest.TestCase):
    """Rescore a saved checkpoint and compare against its original report.

    The stored report predates the selection fix, so its `PR-AUC` column is the
    full-set figure that is now called `pr_auc_all`. `Max-F1` and the threshold
    are unaffected by that change and must match exactly.

    Slow: it loads the full dataset and scores all 1000 users.
    """

    ROUND = 16

    @classmethod
    def setUpClass(cls):
        import federated_ueba.task as task

        with open(os.path.join(BASELINE_SCALERS, "scaler_client_0.pkl"), "rb") as f:
            cls.scaler = pickle.load(f)
        cls.features = list(cls.scaler.feature_names_in_)

        with open(os.path.join(BASELINE_SCALERS, "error_stats_client_0.pkl"), "rb") as f:
            calibration = pickle.load(f)

        model = task.LSTMAutoencoder(input_dim=len(cls.features), hidden_dim=128).to(CPU)
        with open(os.path.join(BASELINE_WEIGHTS,
                               f"parameters_round_{cls.ROUND}.pkl"), "rb") as f:
            weights = pickle.load(f)["global_parameters"]
        model.load_state_dict({k: torch.tensor(w) for k, w
                               in zip(model.state_dict().keys(), weights)})

        # Pinned explicitly rather than read from the config singleton. The
        # singleton is global mutable state that other tests in the same process
        # reassign, which silently rescored this fixture with a different
        # pipeline. A golden test must not depend on ambient configuration.
        pipeline = scoring.ScoringConfig(
            top_k_features=5, persistence_window=3, diversity_threshold=2.0,
            scan_stride=1, inference_batch_size=128)

        df = pd.read_csv(DATA_PATH, low_memory=False)
        scorer = scoring.Scorer(
            features=cls.features, scaler=cls.scaler,
            ref_mean=calibration["mean_per_feature"],
            ref_std=calibration["std_per_feature"],
            cfg=pipeline, window_size=task.WINDOW_SIZE, device=CPU)
        results = scorer.scan(model, df, sorted(df["user"].unique()))
        cls.metrics = scoring.evaluate_scores(results, seed=42)

        report = pd.read_csv(ABLATION_REPORT)
        cls.expected = report[report["Round"] == cls.ROUND].iloc[0]

    def test_full_set_pr_auc_matches_the_original_report(self):
        self.assertAlmostEqual(self.metrics["pr_auc_all"],
                               float(self.expected["PR-AUC"]), places=9)

    def test_threshold_and_f1_match_the_original_report(self):
        # The threshold is a raw score value, so it carries whatever the LSTM
        # forward pass produced. The stored report was generated on GPU and this
        # test runs on CPU, which shifts it around the fourth decimal; on the
        # same device the two agree exactly. Three places still catches any
        # change in the selection logic, which would move it far more than that.
        self.assertAlmostEqual(self.metrics["threshold"],
                               float(self.expected["Optimal-Threshold"]), places=3)
        # F1 is unaffected: no user crosses the threshold in that interval.
        self.assertAlmostEqual(self.metrics["f1"],
                               float(self.expected["Max-F1"]), places=9)

    def test_ranking_is_bit_identical_across_devices(self):
        """PR-AUC depends only on the ordering, so it must match exactly.

        This is the stronger of the two checks: a genuine change to the scoring
        arithmetic would reorder users and show up here even when the absolute
        scores are device-dependent.
        """
        self.assertAlmostEqual(self.metrics["pr_auc_all"],
                               float(self.expected["PR-AUC"]), places=12)

    def test_precision_and_recall_match_the_original_report(self):
        self.assertAlmostEqual(self.metrics["precision"],
                               float(self.expected["Precision"]), places=9)
        self.assertAlmostEqual(self.metrics["recall"],
                               float(self.expected["Recall"]), places=9)


class TestModelPayload(unittest.TestCase):
    """The parameter count is what every communication figure derives from."""

    def test_parameter_count_and_dense_payload(self):
        import federated_ueba.task as task
        model = task.LSTMAutoencoder(input_dim=50, hidden_dim=128)
        total = sum(p.numel() for p in model.parameters())

        self.assertEqual(total, 450_258)
        self.assertAlmostEqual(total * 4 / 1024 / 1024, 1.7176, places=4)
        self.assertAlmostEqual(total * 2 / 1024 / 1024, 0.8588, places=4)

    def _payload_mb(self, chain, codec, entropy):
        from federated_ueba.efficiency_plugins import PluginManager
        manager = PluginManager(chain, codec=codec, entropy=entropy)
        out = manager.apply_on_client(self.params)
        return manager.measure_transport_size(out) / 1024 / 1024

    @property
    def params(self):
        import federated_ueba.task as task
        model = task.LSTMAutoencoder(input_dim=50, hidden_dim=128)
        return [p.detach().numpy() for p in model.parameters()]

    def test_index_encoding_matches_the_figures_reported_in_the_thesis(self):
        """The four-bytes-per-position scheme the reported MB figures came from.

        Pinned explicitly rather than through the default, so switching the
        configured codec cannot silently rewrite what the thesis reports.
        """
        from federated_ueba.efficiency_plugins import (QuantizationPlugin,
                                                       WeightSparsificationPlugin)

        for ratio, expected_mb in ((0.1, 0.3434), (0.05, 0.1716)):
            mb = self._payload_mb([WeightSparsificationPlugin(ratio=ratio)],
                                  "index", "none")
            self.assertAlmostEqual(mb, expected_mb, places=3, msg=f"ratio {ratio}")

        mb = self._payload_mb([WeightSparsificationPlugin(ratio=0.1),
                               QuantizationPlugin()], "index", "none")
        self.assertAlmostEqual(mb, 0.2575, places=3)

    def test_lossless_codecs_only_move_the_byte_count(self):
        """Every codec is cheaper than `index` and none of them touches a weight.

        Measured on an untrained model, so the entropy coder has little structure
        to exploit and these are the pessimistic figures; a trained checkpoint
        compresses further (see analysis/analyze_payload_codecs.py).
        """
        from federated_ueba.efficiency_plugins import WeightSparsificationPlugin

        chain = [WeightSparsificationPlugin(ratio=0.1)]
        index_mb = self._payload_mb(chain, "index", "none")

        for codec, expected_mb in (("bitmask", 0.2254), ("delta_varint", 0.2146)):
            mb = self._payload_mb(chain, codec, "none")
            self.assertAlmostEqual(mb, expected_mb, places=3, msg=codec)
            self.assertLess(mb, index_mb, msg=codec)
            # zlib on top helps here because most of the blob is still zeros. It
            # is not guaranteed to: on incompressible input a deflate stream is a
            # few bytes larger, which is why the coder is configured rather than
            # always on.
            self.assertLess(self._payload_mb(chain, codec, "zlib"), mb, msg=codec)

    def test_dense_baseline_is_charged_the_same_entropy_coder(self):
        """Otherwise a saving would partly be the coder running on one side only."""
        dense_raw = self._payload_mb([], "bitmask", "none")
        dense_zlib = self._payload_mb([], "bitmask", "zlib")

        self.assertAlmostEqual(dense_raw, 1.7176, places=4)
        self.assertLess(dense_zlib, dense_raw)


if __name__ == "__main__":
    unittest.main()
