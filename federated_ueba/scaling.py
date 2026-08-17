"""One scaler, shared by every client.

Each client used to fit its own StandardScaler on its own users. That puts every
client in a different input space: a user with 15 USB events lands at z = 1.0 on
a client whose users are heavy USB users and at z = 48 on a client whose users
barely touch USB. FedAvg then averages weights that were trained to mean
different things, and the reported scores are produced by applying one arbitrary
client's scaler to all 1000 users.

It also produced a failure that was worse than the misalignment. A feature that
happens to be constant across one client's 20 users gets a near-zero scale, and
`log1p` leaves float noise rather than an exact zero, so scikit-learn's
zero-variance guard does not fire. Dividing by 3e-15 turned six features into
values around 1e15, which then dominated the top-5 focus stage and pinned
fourteen users to the top of the ranking on a numerical artifact rather than on
their behaviour.

The fix is one scaler built from statistics every client can contribute without
revealing a record: a count, a sum, and a sum of squares per feature. Those three
numbers are enough to reconstruct the exact global mean and standard deviation,
they are what secure aggregation is designed to add up, and they disclose far
less than the per-feature minimum and maximum a MinMax scaler would need.

The aggregation runs in a loop here because the experiments are a single-process
simulation. In a deployment it is one round: every client sends 3 x 50 numbers,
the server sums them and broadcasts the result.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Imported inside the two functions that need it rather than at module scope, so
# that a caller can pass its own value and nothing here reaches for global state
# on a path where the caller was explicit. Both functions take `None` to mean
# "ask the configuration", which is the only thing a run ever does.


def constant_feature_tolerance(settings=None):
    """Below this spread a feature counts as constant and is left unscaled.

    Configured rather than fixed because it is the guard that decides which
    features are treated as carrying no information, and that decision changes
    what the model sees. Without any floor, float noise in an all-zero feature
    becomes a divisor of about 3.3e-15 and the feature produces z-scores around
    1e15, which is how fourteen users once ended up pinned to the top of the
    ranking on an artefact.
    """
    if settings is None:
        from config_manager import config as settings
    return settings.get("data", "constant_feature_tolerance")


# How a feature value is transformed before scaling. The choice matters more
# than it looks, because the values in the processed dataset are not raw counts:
# `temporal.py` replaces each one with that day's percentile within the user's own
# preceding 30 days, centred at zero, so the range is [-50, +50] and a negative
# value means "below this user's own normal".
#
#   log1p_positive  clip negatives to zero, then log1p. The original behaviour.
#                   Treats only above-normal deviation as signal: 20% of all cells
#                   are negative and every one of them collapses to zero, a
#                   quarter of them from below -35.
#   signed_log1p    log1p of the magnitude, with the sign kept. Compresses the
#                   scale the same way but keeps below-normal days distinguishable
#                   from ordinary ones.
#   raw             the percentile values unchanged. log1p exists to compress
#                   counts spanning orders of magnitude; these are already bounded.
#
# The default is the original, so this setting changes nothing unless asked.
FEATURE_TRANSFORMS = ("log1p_positive", "signed_log1p", "raw")


def _apply_transform(values, transform):
    if transform == "log1p_positive":
        return np.log1p(values.clip(lower=0))
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(values.abs())
    if transform == "raw":
        return values
    raise ValueError(f"Unknown feature transform {transform!r}. "
                     f"Expected one of {list(FEATURE_TRANSFORMS)}.")


def prepare_features(df, features, transform=None):
    """Numeric, missing-safe, transformed. The same on every path.

    Returns a DataFrame rather than an array, keeping the feature names attached
    all the way to `scaler.transform`. The scaler was fitted with names, so it
    checks them, and passing a bare array both skips that check and makes
    scikit-learn warn once per call. The check is worth keeping: the model's
    inputs are positional, so a column arriving out of order would feed the wrong
    feature to the wrong input and fail silently.

    `transform` defaults to the configured one; see FEATURE_TRANSFORMS.
    """
    if transform is None:
        from config_manager import config
        transform = config.get("data", "feature_transform")

    values = (df.reindex(columns=features, fill_value=0)
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0).astype(np.float32))
    return _apply_transform(values, transform)


def _client_statistics(values):
    """What a client would send: count, sum, sum of squares, per feature.

    Accumulated in float64 even though the features are float32. Otherwise the
    totals depend on how many clients the sum was split across, and the scaler
    would quietly differ between an IID and a non-IID run of the same data.
    """
    values = np.asarray(values, dtype=np.float64)
    return len(values), values.sum(axis=0), np.square(values).sum(axis=0)


def minimum_cohort_size(settings=None):
    """Fewest users a client must hold before its statistics are accepted.

    Without a floor, a client holding one user contributes `(n, sum, sum of
    squares)` computed over that user alone, and `sum / n` is then that person's
    own per-feature mean, readable directly by the server. Aggregation stops
    protecting anyone once the cohort is a single person.

    This is not hypothetical here. Under the Dirichlet partition at alpha = 0.5,
    measured over 1000 users and 50 clients: the smallest client holds 1 user and
    16 of the 50 hold two or fewer. The IID partition never goes below 16, so the
    exposure belongs to the non-IID configuration specifically.

    A minimum aggregation threshold is the standard mitigation in federated
    analytics and is what production deployments enforce. It is weaker than
    secure aggregation, which would make the question moot by revealing only the
    federation-wide total, and this pipeline is built to accept that upgrade: the
    three quantities summed here are exactly what a secure-sum protocol adds.
    """
    if settings is None:
        from config_manager import config as settings
    return settings.get("data", "minimum_cohort_size")


def build_global_scaler(df, features, user_chunks, min_cohort=None):
    """Aggregate per-client statistics into one scaler.

    Only normal behaviour contributes, matching what the model is trained on, and
    only clients holding at least `min_cohort` users contribute at all. Excluded
    clients still train; they are held out of this one aggregation round, not out
    of the federation.
    """
    if min_cohort is None:
        min_cohort = minimum_cohort_size()

    count = 0
    total = np.zeros(len(features), dtype=np.float64)
    total_sq = np.zeros(len(features), dtype=np.float64)
    contributing = 0
    excluded_users = 0

    for chunk in user_chunks:
        # Counted before the insider filter and before the empty check, because
        # the question is how many people this client's statistics describe, not
        # how many rows survive. A cohort of five whose rows are mostly filtered
        # is still a cohort of five.
        if len(chunk) < min_cohort:
            excluded_users += len(chunk)
            continue

        client_df = df[df["user"].isin(chunk)]
        if "insider" in client_df.columns:
            client_df = client_df[client_df["insider"] == 0]
        if client_df.empty:
            continue

        n, s, s2 = _client_statistics(prepare_features(client_df, features))
        count += n
        total += s
        total_sq += s2
        contributing += 1

    if count == 0:
        raise ValueError(
            f"No client held at least {min_cohort} users, so no statistics could "
            f"be aggregated. Lower [tool.fueba.data] minimum_cohort_size or use a "
            f"less skewed partition.")

    if excluded_users:
        print(f"[scaling] {contributing} client(s) contributed; cohorts below "
              f"{min_cohort} users were excluded, covering {excluded_users} "
              f"user(s). Their data still trains the model.")

    mean = total / count
    # Clipped at zero because catastrophic cancellation can make this slightly
    # negative for a feature with almost no spread.
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)

    constant = scale < constant_feature_tolerance()
    scale[constant] = 1.0

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = variance
    scaler.scale_ = scale
    scaler.n_features_in_ = len(features)
    scaler.feature_names_in_ = np.asarray(features, dtype=object)
    scaler.n_samples_seen_ = count

    if constant.any():
        # Plain ASCII: this module is imported by processes whose stdout is a
        # legacy Windows code page, where an emoji would raise and kill the run.
        names = [f for f, flat in zip(features, constant) if flat]
        print(f"[scaling] {constant.sum()} feature(s) constant across the whole "
              f"dataset, left unscaled: {', '.join(names[:5])}"
              f"{' ...' if len(names) > 5 else ''}")

    return scaler


def load_or_build_global_scaler(df, features, user_chunks, path):
    """Build the scaler once per run and reuse it from disk afterwards."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    scaler = build_global_scaler(df, features, user_chunks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler
