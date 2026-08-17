"""Detection latency: the prospective replay that answers "when is it caught?".

The logic worth testing is not the model but the replay rule: at each day, score
using only what was available by then. Getting that wrong in the permissive
direction would report a latency that used future data, which is the same class
of mistake as picking a checkpoint with test labels.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.analyze_detection_latency import first_alert_day  # noqa: E402
from federated_ueba.scoring import ScoringConfig  # noqa: E402


class TestFirstAlertDay(unittest.TestCase):
    def setUp(self):
        # persistence_window 1 makes the aggregate the running maximum, which
        # keeps these cases about the replay rule rather than the averaging.
        self.cfg = ScoringConfig(persistence_window=1)

    def test_alerts_on_the_day_the_score_crosses(self):
        scores = np.array([1.0, 2.0, 9.0, 1.0])
        days = np.array([10.0, 11.0, 12.0, 13.0])
        self.assertEqual(first_alert_day(scores, days, 5.0, self.cfg), 12.0)

    def test_returns_none_when_the_user_is_never_flagged(self):
        scores = np.array([1.0, 2.0, 3.0])
        days = np.array([10.0, 11.0, 12.0])
        self.assertIsNone(first_alert_day(scores, days, 50.0, self.cfg))

    def test_alerts_on_the_first_crossing_not_the_highest_score(self):
        """A later, larger spike must not be reported as the detection day."""
        scores = np.array([9.0, 1.0, 99.0])
        days = np.array([10.0, 11.0, 12.0])
        self.assertEqual(first_alert_day(scores, days, 5.0, self.cfg), 10.0)

    def test_a_future_window_cannot_trigger_an_earlier_alert(self):
        """The property that makes this prospective rather than hindsight.

        The big score is last, so no day before it may alert. If the rule looked
        at all windows at once, day 10 would be reported.
        """
        scores = np.array([0.0, 0.0, 100.0])
        days = np.array([10.0, 11.0, 12.0])
        self.assertEqual(first_alert_day(scores, days, 5.0, self.cfg), 12.0)

    def test_exactly_meeting_the_threshold_counts_as_an_alert(self):
        """Matches the >= used when the reported confusion matrix is built."""
        scores = np.array([5.0])
        days = np.array([10.0])
        self.assertEqual(first_alert_day(scores, days, 5.0, self.cfg), 10.0)

    def test_no_windows_means_no_alert(self):
        self.assertIsNone(first_alert_day(np.empty(0), np.empty(0), 1.0, self.cfg))


class TestPersistenceDelaysTheAlert(unittest.TestCase):
    """Averaging the top 3 windows is a deliberate trade of latency for precision.

    One strange day cannot flag a user, which is the whole point of the
    persistence stage, and the cost is that detection needs several bad windows.
    This test states that cost rather than leaving it implicit.
    """

    def test_a_single_spike_does_not_alert_under_top_three_averaging(self):
        cfg = ScoringConfig(persistence_window=3)
        scores = np.array([0.0, 0.0, 30.0])
        days = np.array([10.0, 11.0, 12.0])
        # Mean of the three highest available is 10.0, below the threshold.
        self.assertIsNone(first_alert_day(scores, days, 15.0, cfg))

    def test_three_sustained_windows_do_alert(self):
        cfg = ScoringConfig(persistence_window=3)
        scores = np.array([0.0, 20.0, 20.0, 20.0])
        days = np.array([10.0, 11.0, 12.0, 13.0])
        self.assertEqual(first_alert_day(scores, days, 15.0, cfg), 13.0)

    def test_persistence_one_alerts_earlier_than_persistence_three(self):
        """Same scores, two settings: the trade-off made visible."""
        scores = np.array([20.0, 20.0, 20.0])
        days = np.array([10.0, 11.0, 12.0])
        fast = first_alert_day(scores, days, 15.0, ScoringConfig(persistence_window=1))
        slow = first_alert_day(scores, days, 15.0, ScoringConfig(persistence_window=3))
        self.assertEqual(fast, 10.0)
        self.assertEqual(slow, 10.0)
        self.assertLessEqual(fast, slow)


if __name__ == "__main__":
    unittest.main()
