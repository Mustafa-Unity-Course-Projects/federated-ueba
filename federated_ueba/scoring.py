"""Multi-vector anomaly scoring, shared by the federated and centralized pipelines.

Both `federated_insider_detection.py` and `train_centralized.py` used to carry
their own copy of this logic. That made the federated-vs-centralized comparison
fragile: a change to one copy silently invalidated the comparison. Everything
that turns a reconstruction error into a per-user threat score now lives here.

The pipeline has three stages, each of which can be switched off for the ablation
study (see the `ablation-*` entries in pyproject.toml):

  1. Z-score calibration   normalise each feature's squared error against the
                           reference mean/std collected during training
  2. Top-m feature focus   keep only the m most deviant features
  3. Temporal persistence  average the highest-scoring windows, so a single
                           strange day does not flag a user

A fourth stage, the diversity multiplier, scaled up windows where many features
deviated at once. It is implemented and still reachable, but it is off by
default: measured across five seeds, removing it improved detection every time
(mean +0.0119). It was kept rather than deleted so `ablation-with-diversity` can
still produce that comparison. `STAGES` therefore lists four names while the
configured default lists three.

Note on ordering: the sliding window treats consecutive *rows* as consecutive
days. The CERT day index has gaps (a user with no activity produces no row), so
a window can span more calendar days than its nominal length.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from federated_ueba import scaling


# The pipeline's stages, in the only order they can run in. Naming them replaces
# four independent booleans: an ablation now names what it keeps rather than
# setting three flags true and one false, and adding a fifth stage does not add a
# fifth boolean to every experiment that does not use it.
#
# The order is fixed rather than configurable, because each stage consumes the
# previous one's output. Top-K selects among z-scores, so running it first would
# be selecting among raw squared errors and would mean something else entirely;
# the diversity multiplier counts how many features cleared a threshold on the
# z-score scale. A configurable order would offer combinations that are not
# alternative pipelines but simply wrong ones.
STAGES = ("zscore", "topk", "diversity", "persistence")


@dataclass(frozen=True)
class ScoringConfig:
    """Parameters of the scoring pipeline, and which of its stages run.

    The defaults here are the full pipeline as reported, and exist so that a test
    can construct a scorer without a configuration file. They are never reached
    in a run: `from_config` reads every field through `config.get`, which raises
    on a missing key. An earlier version had a fallback inside `from_config`
    itself, which is the case that actually matters and is gone.
    """

    top_k_features: int = 5
    persistence_window: int = 3
    diversity_threshold: float = 2.0
    scan_stride: int = 1
    inference_batch_size: int = 128
    # Guards the division when a feature's reference error has no spread, which
    # happens for a feature the model reconstructs identically every time.
    zscore_epsilon: float = 1e-6
    # The floor under a feature's reference std, as a share of the median std.
    #
    # `zscore_epsilon` alone is not a guard, because it is an absolute number
    # added to a quantity whose scale it knows nothing about. Measured on the
    # real reference: four of the fifty features have a std around 3e-5, so
    # dividing by `std + 1e-6` multiplies them by roughly thirty thousand, and
    # the top-m stage then picks those same four features for every user. It
    # never fired for most runs only because their models reconstruct those
    # features exactly, making the numerator zero as well. One run whose model
    # was merely near-exact scored 0.2394 instead of 0.8215 on the same
    # checkpoint. A floor relative to the typical std removes the whole class of
    # failure; 1% and 10% gave the same answer, so the exact value is not
    # delicate.
    zscore_std_floor_fraction: float = 0.01
    # Share of users held out for reporting. The other half fits the threshold
    # and picks the checkpoint.
    validation_fraction: float = 0.5
    # Which users land in which half. A property of the evaluation protocol, not
    # of the run, so it must not be the run seed: two seeds would then report on
    # two different sets of 35 insiders and their difference would be mostly the
    # split. Measured at a standard deviation of 0.0548 across splits, which is
    # larger than every effect this work reports.
    split_seed: int = 42
    stages: tuple = STAGES

    def __post_init__(self):
        unknown = [s for s in self.stages if s not in STAGES]
        if unknown:
            raise ValueError(
                f"Unknown scoring stage(s) {unknown}. Known stages, in pipeline "
                f"order: {list(STAGES)}.")

    def uses(self, stage):
        return stage in self.stages

    def without(self, *dropped):
        """The same pipeline with some stages removed. How an ablation is built."""
        return replace(self, stages=tuple(s for s in self.stages
                                          if s not in dropped))

    @classmethod
    def from_config(cls, config):
        """Build from [tool.fueba.scoring]. Every key must be present."""
        requested = config.get("scoring", "stages")
        # Validated before reordering. Sorting into canonical order would drop a
        # misspelled stage instead of complaining, and an ablation that silently
        # ran the full pipeline would look like a finding.
        unknown = [s for s in requested if s not in STAGES]
        if unknown:
            raise ValueError(
                f"[tool.fueba.scoring] stages contains {unknown}, which the "
                f"scorer does not know. Known stages: {list(STAGES)}.")

        return cls(
            top_k_features=config.get("scoring", "top_k_features"),
            persistence_window=config.get("scoring", "persistence_window"),
            diversity_threshold=config.get("scoring", "diversity_threshold"),
            scan_stride=config.get("scoring", "scan_stride"),
            inference_batch_size=config.get("data", "batch_size"),
            zscore_epsilon=config.get("scoring", "zscore_epsilon"),
            zscore_std_floor_fraction=config.get("scoring",
                                                 "zscore_std_floor_fraction"),
            validation_fraction=config.get("evaluation", "validation_fraction"),
            split_seed=config.get("evaluation", "split_seed"),
            # Ordered canonically rather than as written, so that listing the
            # stages in any order describes the same pipeline.
            stages=tuple(s for s in STAGES if s in requested),
        )


def window_score(sq_err, ref_mean, ref_std, cfg, num_features):
    """Score one window from its per-feature squared error.

    Two of the pipeline's three stages happen here; the third, temporal
    persistence, needs a user's whole set of windows and lives in
    `aggregate_windows`. The diversity multiplier is a fourth stage that the
    default pipeline leaves off, kept reachable so the number the thesis reports
    for it can be reproduced.

    Worked example under the shipped pipeline, three features and top_k = 2.
    Squared errors [0.9, 0.1, 0.4], reference means [0.1, 0.1, 0.1], reference
    stds [0.1, 0.1, 0.1]:

        z             = [8.0, 0.0, 3.0]     stage 1, per feature
        positive only = [8.0, 0.0, 3.0]     negatives clipped
        top 2         = [3.0, 8.0]          stage 2
        score         = mean(3.0, 8.0) = 5.5

    With `diversity` switched on, two of the three features exceed the threshold
    of 2, giving 1 + 2/3 = 1.667 and a score of 9.17. That is the arm
    `ablation-with-diversity` measures, not the reported configuration.

    `ref_std` arrives already floored: `Scorer.__post_init__` raises any
    near-zero reference std before anything divides by it. The epsilon below is
    what remains of an earlier guard and no longer changes a reported number.
    """
    # Stage 1. Per feature, and against the error distribution measured on normal
    # training data. A feature the model reconstructs badly for everyone has a
    # high mean, so a high error on it is not by itself surprising; the z-score
    # asks whether this window's error is unusual for this feature.
    if cfg.uses("zscore"):
        feat_z = (sq_err - ref_mean) / (ref_std + cfg.zscore_epsilon)
    else:
        feat_z = sq_err

    # Reconstructing a feature better than usual is not evidence of anything, so
    # negative deviations are clipped rather than allowed to cancel out a real one.
    pos_feat_z = np.maximum(feat_z, 0)

    # Stage 2. Insider behaviour shows up in a few channels, not all fifty. Taking
    # the K largest keeps that signal instead of diluting it in an average over
    # 45 features that were perfectly normal. `np.sort` is ascending, so the last
    # K entries are the largest.
    if cfg.uses("topk"):
        focus_z = np.sort(pos_feat_z)[-cfg.top_k_features:]
    else:
        focus_z = pos_feat_z

    # Stage 3. Between one feature deviating strongly and several deviating at
    # once, the second is the more suspicious pattern: a single channel is often
    # just a busy day. The multiplier runs from 1.0 to 2.0, reaching 2.0 only if
    # every feature is above the threshold.
    if cfg.uses("diversity"):
        deviating = np.sum(pos_feat_z > cfg.diversity_threshold)
        diversity = 1.0 + (deviating / num_features)
    else:
        diversity = 1.0

    return float(np.mean(focus_z) * diversity)


def aggregate_windows(window_scores, cfg):
    """Collapse a user's window scores into one threat score, the final stage.

    Numbered stage 4 before the v8 flip, when the diversity multiplier sat ahead
    of it. It is the third and last stage of the shipped pipeline, and by a wide
    margin the strongest: removing it costs more than the other two together.

    Averaging the top `persistence_window` (3) windows rather than all of them.
    A user is flagged for a sustained pattern, not for one strange fortnight: with
    the mean over everything, three bad windows out of forty disappear; with the
    maximum, a single outlier is enough. The top 3 sits between those.

    Selected by score, not by time. The three need not be consecutive, and at
    stride 1 they usually overlap in days anyway.
    """
    if len(window_scores) == 0:
        return 0.0

    ascending = np.sort(np.asarray(window_scores))
    if not cfg.uses("persistence"):
        return float(np.mean(ascending))

    # The last `persistence_window` entries are the highest scoring ones. `min`
    # guards a user with fewer windows than that.
    highest = ascending[-min(len(ascending), cfg.persistence_window):]
    return float(np.mean(highest))


@dataclass(frozen=True)
class Scorer:
    """Everything needed to turn a reconstruction error into a user's score.

    These seven values always travelled together and were threaded by hand
    through every layer: `scan_users` took ten parameters, `score_one_round`
    twelve, `measure_latency` eleven, and each one passed most of them straight
    down to the next. They are not seven arguments, they are one thing: the input
    space the model learned, plus the reference it is judged against.

    Holding them together also removes a class of mistake the signatures invited.
    Every call site had to pass `ref_mean` and `ref_std` in the right order, and
    `window_size` and `num_features` were two more integers among many; nothing
    would have complained if a caller swapped a pair.

    `features` and `scaler` belong together for the same reason they do in
    training: the scaler was fitted on those columns in that order, and scoring
    has to happen in the space the model actually saw.
    """

    features: list
    scaler: object
    ref_mean: object
    ref_std: object
    cfg: ScoringConfig
    window_size: int
    device: object

    def __post_init__(self):
        """Put a floor under the reference std before anything divides by it.

        Done here rather than at each call site because every scoring path goes
        through a Scorer: the federated evaluation, the centralized one, the
        latency analysis and the tests. A floor applied in only some of them
        would make their numbers incomparable, which is the mistake this is
        correcting in the first place.
        """
        # object.__setattr__ because the dataclass is frozen. Frozen is right:
        # a scorer describes one fixed reference and nothing should rebind it
        # later. This is construction, which is the one moment it may be set.
        object.__setattr__(self, "ref_std",
                           floor_reference_std(
                               self.ref_std,
                               self.cfg.zscore_std_floor_fraction))

    @property
    def num_features(self):
        """What the diversity stage divides by. Always the scorer's own width."""
        return len(self.features)

    def to_tensor(self, user_df):
        """One user's rows as the model's input: transformed, scaled, on device.

        Uses `scaling.prepare_features` rather than a copy of it. The two were
        duplicated until the transform became configurable, at which point they
        would have silently diverged and scoring would have fed the model a
        differently transformed input than it learned on.
        """
        prepared = scaling.prepare_features(user_df, self.features)
        scaled = self.scaler.transform(prepared)
        return torch.tensor(scaled, dtype=torch.float32).to(self.device)

    def windows(self, user_tensor):
        """Every sliding window of one user, in chronological order."""
        span = len(user_tensor) - self.window_size + 1
        return [user_tensor[i:i + self.window_size]
                for i in range(0, span, self.cfg.scan_stride)]

    def window_scores(self, model, user_tensor):
        """Stage 1 to 3 for each window, batched for inference speed.

        Batched because scoring is 1000 users times tens of windows each, once
        per checkpoint, times 26 checkpoints per run.
        """
        scores = []
        windows = self.windows(user_tensor)
        for start in range(0, len(windows), self.cfg.inference_batch_size):
            chunk = windows[start:start + self.cfg.inference_batch_size]
            batch = torch.stack(chunk).to(self.device)
            # Mean over the time axis (dim=1) leaves one squared error per
            # feature, which is what the per-feature z-score in stage 1 needs.
            errors = torch.mean((model(batch) - batch) ** 2, dim=1).cpu().numpy()
            scores.extend(self.score_window(sq_err) for sq_err in errors)
        return scores

    def score_window(self, sq_err):
        return window_score(sq_err, self.ref_mean, self.ref_std, self.cfg,
                            self.num_features)

    def score_user(self, model, user_tensor):
        """One user's threat score, or 0.0 if they have too few days to window.

        Zero rather than an exclusion, so an unscorable user ranks at the bottom
        instead of quietly leaving the metrics.
        """
        if len(user_tensor) < self.window_size:
            return 0.0
        scores = self.window_scores(model, user_tensor)
        return aggregate_windows(scores, self.cfg) if scores else 0.0

    def scan(self, model, df, users):
        """Score every user, one row each, sorted by user id.

        Sorted rather than left in scan order, so that the validation and test
        split downstream cannot depend on the order users happened to arrive in.
        """
        model.eval()
        rows = []
        with torch.no_grad():
            for user in users:
                user_df = rows_for_user(df, user)
                score = self.score_user(model, self.to_tensor(user_df))
                rows.append({
                    "user": user,
                    "max_z_score": score,
                    "is_actual_insider": 1.0 if is_insider(user_df) else 0.0,
                })
        return pd.DataFrame(rows).sort_values("user").reset_index(drop=True)


def rows_for_user(df, user):
    """One user's rows in chronological order.

    Sorting matters because the windows treat consecutive rows as consecutive
    days; unsorted rows would build windows out of days that never followed one
    another.
    """
    user_df = df[df["user"] == user]
    return user_df.sort_values("day") if "day" in user_df.columns else user_df


def is_insider(user_df):
    """True if any single day is labelled. The task is to rank people, not days."""
    return bool((user_df["insider"] != 0).any()) if "insider" in user_df else False


def calculate_metrics_at_threshold(y_true, y_scores, threshold):
    """Confusion matrix and the metrics derived from it, at a given cut-off."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    # `labels=[0, 1]` forces the full 2x2 shape. Without it, a split in which
    # nothing is predicted positive returns a 1x1 matrix and the unpacking below
    # raises instead of reporting zero true positives.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def floor_reference_std(ref_std, fraction):
    """Raise any near-zero reference std to a share of the typical one.

    A feature whose reference error never varies carries no information about
    whether a window is unusual, but dividing by its std says the opposite as
    loudly as arithmetic allows. The floor is relative to the median rather than
    absolute so it follows the scale of whatever data the model was trained on.

    The median is taken over the features whose std is above zero. That is the
    simple rule and it fits the data: four of fifty features are degenerate, so
    the median is set by the other forty-six and the floor lands where it should.

    Its limit is worth naming. A reference where most features were degenerate
    would have its median set by them and the floor would follow them down,
    guarding little. No threshold separating "flat" from "not flat" avoids this
    without inventing one, and there is nothing in this data to fit it on, so the
    simple rule stays and the limit is written down instead.
    """
    if ref_std is None or fraction <= 0:
        return ref_std

    ref_std = np.asarray(ref_std, dtype=np.float64)
    if ref_std.size == 0:
        return ref_std

    positive = ref_std[ref_std > 0]
    if positive.size == 0:
        return ref_std
    return np.maximum(ref_std, fraction * float(np.median(positive)))


def split_users(users, labels, seed=42, validation_fraction=0.5):
    """The validation/test halves of the user population.

    The single authority on this split. It has to be, because the split decides
    which users a reported number describes, and anything that reports a number
    afterwards has to describe the same ones. The confidence interval in
    `analysis/compare_experiments.py` did not, and so was an interval around a
    quantity other than the one it was printed beside.

    Deterministic in `seed`. Callers pass `ScoringConfig.split_seed`, which is
    fixed across runs and is not the run seed; see that field for why. The split
    can therefore be reproduced from a finished run's per-user CSV without
    re-scoring anything.
    """
    # Stratified so both halves hold a comparable share of the 70 insiders.
    # Without it a split could leave one half with almost none of them and make
    # its PR-AUC meaningless.
    try:
        return train_test_split(users, test_size=validation_fraction,
                                stratify=labels, random_state=seed)
    except ValueError:
        # Raised when a class has too few members to stratify, which happens on
        # small synthetic fixtures rather than on the real dataset.
        return train_test_split(users, test_size=validation_fraction,
                                random_state=seed)


def evaluate_scores(results, seed=42, validation_fraction=0.5):
    """Turn per-user scores into the reported metrics.

    Users are split into a validation and a test half, stratified on the label.
    Everything that involves a choice is made on the validation half; everything
    reported as a result comes from the test half.

    That separation matters twice over:

      threshold   the cut-off that maximises F1 is fitted on the validation half
                  and applied to the test half
      checkpoint  the caller selects which round or epoch to report using
                  `pr_auc_val`, and then reports `pr_auc_test`

    The second one used to be missing, and it mattered more than the first. The
    federated runs reported the best PR-AUC over 26 checkpoints, scored on all
    1000 labelled users, while the centralized run reported whichever epoch had
    the lowest reconstruction loss and never consulted a label. Those two numbers
    were not comparable, and the federated one was optimistic. `pr_auc_all` is
    still returned for continuity with earlier reports, but it must not be used
    to pick a checkpoint.

    `results` must already be sorted by user (scan_users guarantees this) or the
    split will depend on scan order.
    """
    val_users, test_users = split_users(
        results['user'].tolist(), results['is_actual_insider'].tolist(),
        seed=seed, validation_fraction=validation_fraction)

    val_df = results[results['user'].isin(val_users)]
    test_df = results[results['user'].isin(test_users)]

    # The threshold is the one that maximises F1 on the validation half. F1 is
    # computed from the curve directly rather than by sweeping candidate
    # thresholds, since the curve already enumerates every distinct cut-off.
    val_p, val_r, val_t = precision_recall_curve(val_df['is_actual_insider'],
                                                 val_df['max_z_score'])
    denominator = val_p + val_r
    f1_curve = np.zeros_like(denominator)
    # Where precision and recall are both zero, F1 is undefined; left at zero so
    # those points can never win the argmax.
    nonzero = denominator > 0
    f1_curve[nonzero] = (2 * val_p * val_r)[nonzero] / denominator[nonzero]
    best_idx = int(np.argmax(f1_curve))
    # precision_recall_curve returns one more precision/recall point than it does
    # thresholds: the final point is recall 0, precision 1, which corresponds to
    # no threshold at all. The guard covers the case where that point wins.
    threshold = float(val_t[best_idx]) if best_idx < len(val_t) else float(val_t[-1])

    # Fitted on validation, applied to test. The test half is touched once, here.
    metrics = calculate_metrics_at_threshold(test_df['is_actual_insider'],
                                             test_df['max_z_score'], threshold)
    metrics["pr_auc_val"] = float(average_precision_score(
        val_df['is_actual_insider'], val_df['max_z_score']))
    metrics["pr_auc_test"] = float(average_precision_score(
        test_df['is_actual_insider'], test_df['max_z_score']))
    metrics["pr_auc_all"] = float(average_precision_score(
        results['is_actual_insider'], results['max_z_score']))
    # Headline value. Selection never touches these users.
    metrics["pr_auc"] = metrics["pr_auc_test"]
    metrics["threshold"] = threshold
    return metrics
