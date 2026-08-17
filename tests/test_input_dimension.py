"""Server and clients must size the model from the same number.

They did not. The server sized its initial global model from
`len(selected_features)`, which is 50, while each client sized its own from the
data after the variance filter, which is 46 under `drop_constant_features`. Every
client in the `features-filtered` experiment then rejected the broadcast:

    size mismatch for encoder.weight_ih_l0: copying a param with shape
    torch.Size([256, 50]), the shape in current model is torch.Size([256, 46])

every round, for the whole run, 15,002 times in one sweep. The experiment failed
and the sweep moved on, so the only visible symptom was a missing result.

The general shape of the bug is two independent answers to one question. These
tests pin that there is now one: `task.input_dimension`.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config_manager import config
from federated_ueba import task


def frame_with_dead_columns(n_users=6, n_days=30, dead=2):
    """A frame whose last `dead` feature columns are zero on every row."""
    rng = np.random.RandomState(0)
    live = [f"f{i}" for i in range(4)]
    dead_names = [f"dead{i}" for i in range(dead)]
    rows = []
    for user in range(n_users):
        for day in range(n_days):
            row = {"user": f"u{user}", "day": day, "insider": 0}
            for name in live:
                row[name] = float(rng.randint(0, 20))
            for name in dead_names:
                row[name] = 0.0
            rows.append(row)
    return pd.DataFrame(rows), live + dead_names


class TestTheFilterActuallyNarrowsTheInput(unittest.TestCase):
    def test_dead_columns_are_dropped_when_the_filter_is_on(self):
        df, features = frame_with_dead_columns(dead=2)
        kept = task.drop_constant_features(df, features)
        self.assertEqual(len(kept), len(features) - 2)
        self.assertNotIn("dead0", kept)

    def test_nothing_is_dropped_when_every_column_varies(self):
        df, features = frame_with_dead_columns(dead=0)
        self.assertEqual(task.drop_constant_features(df, features), features)


class TestOneAnswerForTheInputWidth(unittest.TestCase):
    """`input_dimension` must agree with what a client derives from the data."""

    def setUp(self):
        self._experiment = config._experiment_name

    def tearDown(self):
        config.set_experiment(self._experiment)

    def width_from_the_data(self, df):
        """What a client ends up with: select_features over the real frame."""
        return len(task.select_features(df))

    def test_the_configured_default_keeps_every_feature(self):
        config.set_experiment("baseline")
        self.assertFalse(config.get("data", "drop_constant_features"))

    def test_features_filtered_is_the_experiment_that_narrows_it(self):
        """If this ever stops being true the regression below tests nothing."""
        config.set_experiment("features-filtered")
        self.assertTrue(config.get("data", "drop_constant_features"))

    def test_the_server_width_matches_the_client_width(self):
        """The regression. Both sides go through select_features now.

        Run for both experiments, because the bug was invisible in the default
        configuration: with the filter off the two answers coincide by accident,
        which is why it survived until an experiment turned the filter on.
        """
        path = config.get("data", "processed_data_path")
        if not os.path.exists(path):
            self.skipTest(f"processed dataset not present at {path}")

        df = task.load_dataset(path)
        for experiment in ("baseline", "features-filtered"):
            with self.subTest(experiment=experiment):
                config.set_experiment(experiment)
                self.assertEqual(task.input_dimension(path),
                                 self.width_from_the_data(df))

    def test_the_filter_changes_the_width_it_is_supposed_to_change(self):
        """Otherwise the two experiments would be the same run under two names."""
        path = config.get("data", "processed_data_path")
        if not os.path.exists(path):
            self.skipTest(f"processed dataset not present at {path}")

        config.set_experiment("baseline")
        wide = task.input_dimension(path)
        config.set_experiment("features-filtered")
        narrow = task.input_dimension(path)

        self.assertLess(narrow, wide)
        # Four features are zero on every row of this dataset, which is also the
        # figure the thesis quotes for the variance filter.
        self.assertEqual(wide - narrow, 4)

    def test_a_model_built_from_it_accepts_the_configured_features(self):
        """Sizes must line up end to end, not just as integers."""
        import torch

        path = config.get("data", "processed_data_path")
        if not os.path.exists(path):
            self.skipTest(f"processed dataset not present at {path}")

        config.set_experiment("features-filtered")
        width = task.input_dimension(path)
        model = task.LSTMAutoencoder(input_dim=width)

        batch = torch.randn(2, config.get("model", "window_size"), width)
        self.assertEqual(tuple(model(batch).shape),
                         (2, config.get("model", "window_size"), width))


if __name__ == "__main__":
    unittest.main()
