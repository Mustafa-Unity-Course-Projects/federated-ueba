"""The model's shape is configuration, and each switch has to actually do something.

These parameters were declared in `pyproject.toml` and validated by the schema
long before any of them was read, so a run configured with `use_bottleneck =
false` built the ordinary model and reported no error. Nothing failed, which is
what made it dangerous: an ablation would have produced a result rather than a
crash, and the result would have been the baseline's.

Two things are pinned here. That the configured default is still the 450,258
parameter model every communication figure in the thesis rests on, and that
turning each switch off changes the model rather than being ignored.
"""

import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config_manager import config
from federated_ueba import task

# The parameter count of the reported architecture. Every MB figure in the thesis
# is this number times 4 bytes, so a change here invalidates all of them.
REPORTED_PARAMETER_COUNT = 450_258

# The two architecture ablations that retrain. Pinned because each one's payload
# differs from the baseline's and its communication figures are read against that
# size: 1.9676 MB and 1.3367 MB against 1.7176.
UNIDIRECTIONAL_COUNT = 515_794          # encoder-unidirectional
NARROW_UNIDIRECTIONAL_COUNT = 350_418   # encoder-unidirectional-64

INPUT_DIM = 50


def build(arch):
    return task.LSTMAutoencoder(input_dim=INPUT_DIM, arch=arch)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


class TestConfiguredArchitecture(unittest.TestCase):
    def setUp(self):
        self.arch = task.Architecture.from_config(config)

    def test_the_configured_model_is_the_one_the_thesis_reports(self):
        self.assertEqual(parameter_count(build(self.arch)),
                         REPORTED_PARAMETER_COUNT)

    def test_construction_reads_configuration_when_no_architecture_is_given(self):
        """The call sites pass only input_dim and hidden_dim, so this is the real path."""
        from_config = task.LSTMAutoencoder(input_dim=INPUT_DIM)
        self.assertEqual(parameter_count(from_config), REPORTED_PARAMETER_COUNT)

    def test_hidden_dim_overrides_the_configured_width_and_nothing_else(self):
        """Evaluation reads the width back from a checkpoint, not from configuration."""
        narrow = task.LSTMAutoencoder(input_dim=INPUT_DIM, hidden_dim=64)
        self.assertLess(parameter_count(narrow), REPORTED_PARAMETER_COUNT)
        self.assertEqual(narrow.arch.encoder_layers, self.arch.encoder_layers)
        self.assertTrue(narrow.arch.use_bottleneck)

    def test_a_missing_model_key_stops_construction(self):
        class Incomplete:
            def get(self, section, key):
                if key == "use_bottleneck":
                    raise KeyError(key)
                return config.get(section, key)

        with self.assertRaises(KeyError):
            task.Architecture.from_config(Incomplete())


class TestSwitchesChangeTheModel(unittest.TestCase):
    """Each switch off must give a different model that still runs end to end."""

    def setUp(self):
        self.arch = task.Architecture.from_config(config)
        self.baseline = parameter_count(build(self.arch))

    def assert_differs_and_still_reconstructs(self, arch):
        model = build(arch)
        self.assertNotEqual(parameter_count(model), self.baseline)
        # Two windows in, two reconstructions of the same shape out. A switch
        # that changes the latent width without changing the decoder would break
        # here rather than at training time.
        batch = torch.randn(2, arch.window_size, INPUT_DIM)
        self.assertEqual(tuple(model(batch).shape), (2, arch.window_size, INPUT_DIM))

    def test_turning_the_bottleneck_off_removes_its_parameters(self):
        self.assert_differs_and_still_reconstructs(
            replace(self.arch, use_bottleneck=False))

    def test_a_unidirectional_encoder_is_a_different_model(self):
        """Larger, not smaller: one direction at the full width beats two at half."""
        arch = replace(self.arch, encoder_bidirectional=False)
        self.assertEqual(parameter_count(build(arch)), UNIDIRECTIONAL_COUNT)
        self.assertGreater(parameter_count(build(arch)), self.baseline)
        self.assert_differs_and_still_reconstructs(arch)

    def test_pinning_the_encoder_width_removes_the_backward_pass_alone(self):
        """`encoder-unidirectional-64`, the complement of the arm above.

        The arm above grows because one direction widens to the full hidden_dim.
        This one holds the per-direction width at what the bidirectional encoder
        already used, so the only thing the switch removes is the second reading
        direction, and the model gets smaller instead of larger.
        """
        arch = replace(self.arch, encoder_bidirectional=False,
                       encoder_hidden_override=64)
        self.assertEqual(arch.encoder_hidden_per_direction, 64)
        self.assertEqual(arch.latent_dim, 64)
        self.assertEqual(parameter_count(build(arch)), NARROW_UNIDIRECTIONAL_COUNT)
        self.assertLess(parameter_count(build(arch)), self.baseline)
        self.assert_differs_and_still_reconstructs(arch)

    def test_a_narrowed_encoder_needs_the_bottleneck_to_absorb_it(self):
        """Identity would hand a 64-wide latent to a decoder built for 128.

        A shape error several minutes into training is the failure this replaces.
        """
        arch = replace(self.arch, encoder_bidirectional=False,
                       encoder_hidden_override=64, use_bottleneck=False)
        with self.assertRaises(ValueError):
            build(arch)

    def test_layer_counts_change_the_model(self):
        self.assert_differs_and_still_reconstructs(
            replace(self.arch, encoder_layers=1))
        self.assert_differs_and_still_reconstructs(
            replace(self.arch, decoder_layers=3))

    def test_the_bottleneck_divisor_sets_the_narrowest_point(self):
        arch = replace(self.arch, bottleneck_divisor=8)
        self.assertEqual(arch.bottleneck_width, arch.hidden_dim // 8)
        self.assert_differs_and_still_reconstructs(arch)


class TestArchitectureInvariants(unittest.TestCase):
    def setUp(self):
        self.arch = task.Architecture.from_config(config)

    def test_the_latent_stays_hidden_dim_wide_in_both_directions_settings(self):
        """Why the decoder does not have to be resized when the encoder changes.

        Holds whenever `encoder_hidden_override` is off, which is every reported
        run except `encoder-unidirectional-64`. That arm sets the override
        precisely to break this invariant, and the bottleneck absorbs the
        difference so the decoder still does not have to know.
        """
        self.assertEqual(self.arch.encoder_hidden_override, 0)

        bidirectional = self.arch
        self.assertEqual(bidirectional.latent_dim, bidirectional.hidden_dim)

        unidirectional = replace(self.arch, encoder_bidirectional=False)
        self.assertEqual(unidirectional.encoder_hidden_per_direction,
                         unidirectional.hidden_dim)
        self.assertEqual(unidirectional.latent_dim, unidirectional.hidden_dim)

        narrowed = replace(unidirectional, encoder_hidden_override=64)
        self.assertNotEqual(narrowed.latent_dim, narrowed.hidden_dim)

    def test_dropout_is_zeroed_for_a_single_layer_lstm(self):
        """PyTorch would ignore it and warn; reporting it as applied would be a lie."""
        self.assertEqual(self.arch.lstm_dropout(1), 0.0)
        self.assertEqual(self.arch.lstm_dropout(2), self.arch.dropout)

    def test_the_shape_cannot_be_mutated_after_a_model_is_built(self):
        """A changed field would leave the state dict describing a model that no
        longer exists, and under FedAvg that surfaces rounds later."""
        with self.assertRaises(Exception):
            self.arch.hidden_dim = 64

    def test_noise_is_training_only(self):
        model = build(self.arch)
        batch = torch.randn(2, self.arch.window_size, INPUT_DIM)

        model.eval()
        with torch.no_grad():
            first, second = model(batch), model(batch)
        # Deterministic in eval: the detector rescores the same window across 26
        # checkpoints and the scores have to be comparable.
        self.assertTrue(torch.equal(first, second))

    def test_input_noise_is_the_only_randomness_left_once_dropout_is_off(self):
        """Dropout is zeroed in both cases so that only the noise setting differs.

        In train mode the LSTM's own dropout is also active, so a model is
        nondeterministic there whatever the noise is set to. Turning dropout off
        isolates the switch under test.
        """
        batch = torch.randn(2, self.arch.window_size, INPUT_DIM)

        silent = build(replace(self.arch, dropout=0.0, input_noise_std=0.0))
        silent.train()
        with torch.no_grad():
            self.assertTrue(torch.equal(silent(batch), silent(batch)))

        noisy = build(replace(self.arch, dropout=0.0, input_noise_std=0.02))
        noisy.train()
        with torch.no_grad():
            self.assertFalse(torch.equal(noisy(batch), noisy(batch)))


if __name__ == "__main__":
    unittest.main()
