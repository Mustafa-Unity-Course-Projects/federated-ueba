"""The model, the data it is fed, and the local training loop each client runs.

The pipeline in one line: daily per-user feature rows, scaled by the federation's
single global scaler, cut into overlapping 14-day windows, reconstructed by a
BiLSTM autoencoder that has only ever seen normal behaviour.
"""

import os
import pickle
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config_manager import config
from federated_ueba import proximal, scaling, seeding

# Read once at import. These are module constants rather than lookups because
# they appear in inner loops and because the values are fixed for the lifetime of
# a run; the runner sets the experiment before importing this module.
WINDOW_SIZE = config.get("model", "window_size")
STRIDE = config.get("model", "stride")


def _resolve_device():
    """Where tensors live: the configured device, or whatever is available.

    Measured on the same two-round configuration and the same code, 98.7s on the
    GPU against 334.1s on the CPU. The GPU is worth roughly 3.4x here, so a CPU
    sweep is a fallback rather than a free way to keep the machine responsive.

    Read inside the worker on purpose. Ray rewrites CUDA_VISIBLE_DEVICES per
    actor from client-resources.num_gpus, so setting that variable on the runner
    does not reach the workers; an earlier measurement that appeared to show the
    CPU matching the GPU was in fact two GPU runs.
    """
    choice = (config.get("experiment", "device")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice != "auto":
        raise ValueError(f"Unknown device {choice!r}. Expected auto, cpu or cuda.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _resolve_device()
BATCH_SIZE = config.get("data", "batch_size")
SELECTED_FEATURES = config.get("data", "selected_features")
HIDDEN_DIM = config.get("model", "hidden_dim")
TEST_SIZE = config.get("data", "test_size")

# The full dataset, held once per process. Under Ray each client actor loads it
# separately, and without this cache a 50-client round would reread the same CSV
# fifty times.
_cached_df = None

@dataclass(frozen=True)
class Architecture:
    """The shape of the autoencoder, read from `[tool.fueba.model]`.

    Every field here is a design decision the thesis has to be able to defend,
    which is why none of them is a literal inside the model any more. Switching
    one off is how the ablation study measures what it was worth: with
    `use_bottleneck = false` there is no compression to force generalisation, and
    with `encoder_bidirectional = false` the encoder sees only the days before
    the one it is reading.

    Frozen because a model's shape must not change after its weights exist. A
    silently mutated field would mean the state dict no longer matches the
    architecture that produced it, and under FedAvg that surfaces as a shape
    mismatch several rounds later rather than at the point of the mistake.
    """

    hidden_dim: int
    encoder_layers: int
    decoder_layers: int
    encoder_bidirectional: bool
    encoder_hidden_override: int
    use_bottleneck: bool
    bottleneck_divisor: int
    dropout: float
    input_noise_std: float
    window_size: int

    @classmethod
    def from_config(cls, source):
        """Read every field from configuration. A missing key stops the run."""
        return cls(
            hidden_dim=source.get("model", "hidden_dim"),
            encoder_layers=source.get("model", "encoder_layers"),
            decoder_layers=source.get("model", "decoder_layers"),
            encoder_bidirectional=source.get("model", "encoder_bidirectional"),
            encoder_hidden_override=source.get("model",
                                               "encoder_hidden_override"),
            use_bottleneck=source.get("model", "use_bottleneck"),
            bottleneck_divisor=source.get("model", "bottleneck_divisor"),
            dropout=source.get("model", "dropout"),
            input_noise_std=source.get("model", "input_noise_std"),
            window_size=source.get("model", "window_size"),
        )

    @property
    def encoder_directions(self):
        return 2 if self.encoder_bidirectional else 1

    @property
    def encoder_hidden_per_direction(self):
        """Width of one direction, so the directions concatenate to hidden_dim.

        Halving it under a bidirectional encoder is what keeps the rest of the
        network at one consistent width regardless of this switch: the latent
        vector is hidden_dim wide either way, so the bottleneck and the decoder
        do not have to change shape when the encoder becomes unidirectional.

        `encoder_hidden_override` breaks that tie on purpose, and exists for one
        measurement. Switching bidirectionality off widens the single direction
        from 64 to 128, so the unidirectional arm carries *more* parameters than
        the bidirectional one. That is the right control for a loss, which could
        then not be blamed on capacity, but it is a weak control for the null
        that was actually measured: a real contribution from the backward pass
        could be masked by the 65,536 parameters the switch adds. Pinning the
        per-direction width instead removes the backward pass and nothing else.
        """
        return (self.encoder_hidden_override or
                self.hidden_dim // self.encoder_directions)

    @property
    def latent_dim(self):
        """Width of the vector the encoder hands to the bottleneck.

        Equal to hidden_dim whenever the override is off, which is what the
        decoder assumes and what every reported run before this field used.
        """
        return self.encoder_hidden_per_direction * self.encoder_directions

    @property
    def bottleneck_width(self):
        """The narrowest point, hidden_dim // divisor. 32 at 128 and 4."""
        return self.hidden_dim // self.bottleneck_divisor

    def lstm_dropout(self, num_layers):
        """Dropout as nn.LSTM will actually apply it.

        PyTorch drops between stacked layers only, so a single-layer LSTM ignores
        the value and warns about it. Zeroing it here keeps the configured number
        honest instead of letting the log claim a dropout that never ran.
        """
        return self.dropout if num_layers > 1 else 0.0


class LSTMAutoencoder(nn.Module):
    """Bidirectional LSTM autoencoder over a 14 x 50 window of one user's days.

    It is trained only on normal behaviour and never sees a label. Detection
    comes from what it fails to reproduce: a window it reconstructs badly is one
    unlike anything it was trained on. That is why an autoencoder is used rather
    than a classifier, and it is what makes the approach workable on a dataset
    with 966 insider rows out of hundreds of thousands.

    Bidirectional in the encoder because a day is characterised by the days on
    both sides of it. A logon at 3am reads differently depending on whether the
    fortnight around it was quiet or busy, and a forward-only pass can only use
    the half of the window that came earlier.

    At the default hidden_dim of 128 this is 450,258 parameters, or 1.7176 MB in
    fp32, which is the figure every communication number in the thesis rests on:

        encoder       158,720   35.3%
        bottleneck     20,896    4.6%
        decoder       264,192   58.7%
        output_layer    6,450    1.4%
    """

    def __init__(self, input_dim, hidden_dim=None, arch=None):
        """Build the model described by `arch`, or by configuration if omitted.

        `hidden_dim` stays in the signature because every call site passes it and
        because the width is the one architecture field that varies inside a
        single run: `analyze_detection_latency` and the evaluation pass both read
        it back from a checkpoint rather than from configuration. When given, it
        overrides the configured width and leaves every other field alone.
        """
        super(LSTMAutoencoder, self).__init__()

        if arch is None:
            arch = Architecture.from_config(config)
        if hidden_dim is not None and hidden_dim != arch.hidden_dim:
            arch = replace(arch, hidden_dim=hidden_dim)
        self.arch = arch

        self.encoder = nn.LSTM(
            input_dim, arch.encoder_hidden_per_direction,
            num_layers=arch.encoder_layers, batch_first=True,
            dropout=arch.lstm_dropout(arch.encoder_layers),
            bidirectional=arch.encoder_bidirectional)

        # The narrowest point is hidden_dim // bottleneck_divisor, that is 32
        # values, and it is the whole mechanism: a fortnight of 700 numbers has
        # to pass through 32 before it can be rebuilt. Anything the model can
        # reconstruct is therefore a pattern common enough to be worth those 32
        # dimensions, which is what makes an unusual window reconstruct badly.
        #
        # Identity when switched off, rather than a shorter network. That keeps
        # the latent vector hidden_dim wide, so the decoder is unchanged and the
        # ablation isolates the compression instead of also resizing the decoder.
        if arch.use_bottleneck:
            wide, narrow = arch.hidden_dim, arch.bottleneck_width
            middle = arch.hidden_dim // 2
            # In goes whatever the encoder produced, out goes hidden_dim: the
            # bottleneck is the one place a narrowed encoder is absorbed, so the
            # decoder never has to know the encoder's width.
            self.bottleneck = nn.Sequential(
                nn.Linear(arch.latent_dim, middle),
                nn.LayerNorm(middle),
                nn.ReLU(),
                nn.Linear(middle, narrow),   # the narrowest point
                nn.ReLU(),
                nn.Linear(narrow, middle),
                nn.ReLU(),
                nn.Linear(middle, wide)
            )
        elif arch.latent_dim != arch.hidden_dim:
            # Identity passes the encoder's width straight to a decoder built for
            # hidden_dim. Without the bottleneck there is nothing left to absorb
            # the difference, so this combination cannot be built. Refused here
            # rather than surfacing as a shape error inside the first forward
            # pass of a run that has already started training.
            raise ValueError(
                f"encoder_hidden_override={arch.encoder_hidden_override} gives a "
                f"latent of {arch.latent_dim}, but use_bottleneck is off and the "
                f"decoder expects {arch.hidden_dim}. The bottleneck is what "
                f"reconciles the two widths.")
        else:
            self.bottleneck = nn.Identity()

        # Unidirectional: reconstruction runs forward in time, and a decoder
        # allowed to look ahead would be reading the answer. Not configurable for
        # that reason, unlike the encoder's direction.
        self.decoder = nn.LSTM(arch.hidden_dim, arch.hidden_dim,
                               num_layers=arch.decoder_layers, batch_first=True,
                               dropout=arch.lstm_dropout(arch.decoder_layers))
        self.output_layer = nn.Linear(arch.hidden_dim, input_dim)

    def forward(self, x):
        if self.training and self.arch.input_noise_std > 0:
            # Denoising: a little noise on the input stops the model from
            # learning to copy its input exactly, which would leave no
            # reconstruction error to detect anything with. Training only.
            x = x + torch.randn_like(x) * self.arch.input_noise_std

        # The LSTM returns the hidden state of every layer and direction stacked
        # together.
        _, (hidden, _) = self.encoder(x)

        if self.arch.encoder_bidirectional:
            # The last two entries are the final layer's backward and forward
            # states, which concatenate into the hidden_dim-wide summary of the
            # whole window. The earlier layers' states are intermediate and are
            # not what the bottleneck should compress.
            latent = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # One direction, so the final layer's single state is already the
            # full hidden_dim-wide summary.
            latent = hidden[-1, :, :]

        latent = self.bottleneck(latent)

        # The bottleneck emits one vector for the whole window, but the decoder
        # is a sequence model and wants one input per timestep. Repeating the
        # vector window_size times gives every day the same starting summary and
        # leaves the decoder to recover the day-to-day structure from it.
        decoded_init = latent.unsqueeze(1).repeat(1, self.arch.window_size, 1)

        x_recon, _ = self.decoder(decoded_init)
        return self.output_layer(x_recon)

def _dirichlet_partition(all_users, num_partitions, alpha, seed=42):
    """Split users across clients unevenly, with the imbalance drawn from Dirichlet(alpha).

    This is what "non-IID" means in this work, and it is a deliberately mild form
    of it: every client still holds a random sample of users, only the number of
    users differs. A harsher version would skew *which kind* of user each client
    holds, which this dataset does not label finely enough to do honestly.

    Lower alpha means more heterogeneity. alpha = 0.5 is moderate, alpha = 100 is
    indistinguishable from an even split. Returns num_partitions arrays of users.
    """
    rng = np.random.RandomState(seed)
    all_users = np.array(all_users)
    n = len(all_users)

    # One proportion per client, summing to 1.
    proportions = rng.dirichlet(np.repeat(alpha, num_partitions))

    # At low alpha some proportions round to zero. A client with no users trains
    # on nothing and contributes an untrained model to the average, so the floor
    # of one is not cosmetic.
    counts = np.maximum(1, np.round(proportions * n).astype(int))

    # Rounding and that floor both break the total, so the difference is taken
    # from the largest client or given to the smallest, whichever is needed. This
    # keeps the partition a true partition: every user assigned exactly once.
    difference = n - counts.sum()
    if difference > 0:
        counts[np.argmin(counts)] += difference
    elif difference < 0:
        counts[np.argmax(counts)] -= abs(difference)

    # Shuffle first, so which users a client gets is independent of the sizes.
    shuffled = rng.permutation(all_users)
    chunks = []
    start = 0
    for count in counts:
        chunks.append(shuffled[start:start + count])
        start += count
    return chunks


def _user_behaviour_clusters(df, all_users, num_clusters, seed):
    """Group users by how they behave, as a stand-in for a class label.

    The federated learning literature's non-IID results are about *label*
    distribution skew: clients holding different classes drift apart during local
    training, and FedAvg's average of drifted models is worse than any of them.
    This work is unsupervised, so there is no label to skew, and the quantity
    skew in `_dirichlet_partition` is not a substitute: measured on this dataset
    at alpha 0.5 it moves client sizes between 1 and 102 users while moving their
    feature means 0.005 standard deviations. Every client stays a random sample
    of the same distribution, every local objective stays an unbiased estimate of
    the global one, and there is nothing for FedAvg to get wrong.

    Behavioural clusters are the meaningful analogue, but they cannot be built
    from the mean of a user's daily feature vector. The processed dataset holds
    percentile deviations from each user's own baseline, so every user's signed
    mean is near zero by construction: measured here, user identity accounts for
    0.4% of the variance in the signed means, and k-means on them puts 996 of
    1000 users in one cluster and isolates four outliers.

    What does vary is *how far* a user typically strays from their own baseline.
    Profiling users by their mean absolute deviation instead of their mean gives
    balanced, genuinely different groups, and the difference is one the model can
    feel: an autoencoder trained mostly on steady users reconstructs volatile
    ones badly. That is the skew worth allocating unevenly.

    Returns an array of cluster labels aligned with `all_users`.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = select_features(df)
    profiles = (df[df['user'].isin(all_users)]
                .groupby('user')[features]
                .apply(lambda group: group.abs().mean())
                .reindex(all_users))
    # A user with no rows would become NaN and take the whole fit with it.
    profiles = profiles.fillna(profiles.mean())

    standardised = StandardScaler().fit_transform(profiles.values)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    return kmeans.fit_predict(standardised)


def _role_labels(df, all_users):
    """Each user's job role, which is the natural grouping this dataset has.

    Preferred over `_user_behaviour_clusters` wherever it can be used. The
    federated learning literature reaches for synthetic Dirichlet partitions when
    a dataset has no natural one, and uses the natural one when it does: FEMNIST
    splits by writer, Shakespeare by speaking role, Stack Overflow by user. CERT
    r4.2 ships an organisational hierarchy, and `role` survives into the
    processed table, so the synthetic route is not needed.

    It is also the partition this work already claims to model. A client here
    stands for an organisation holding its own people's activity; an
    organisational unit is what that means, and k-means over behaviour is not.

    Measured on this dataset: random 30-user groups deviate from the global
    profile by 0.124 on average (95th percentile 0.184), while real roles of
    comparable size reach 0.526. The grouping carries signal that random
    allocation does not.
    """
    roles = df.groupby('user')['role'].first().reindex(all_users)
    # Dense integer codes, because the partition indexes by label. Roles arrive
    # as arbitrary numbers with gaps, and a missing role would otherwise become
    # a silent NaN group.
    return pd.factorize(roles.fillna(-1))[0]


def _cluster_dirichlet_partition(df, all_users, num_partitions, alpha,
                                 num_clusters, seed=42, labels=None):
    """Skew *which kind* of user each client holds, not just how many.

    The construction is the standard one for label-skewed federated benchmarks:
    for each cluster independently, draw a Dirichlet over clients and split that
    cluster's members according to it. A small alpha concentrates each cluster on
    a few clients, so clients end up with different cluster mixtures rather than
    different sample counts of the same mixture.

    Deliberately separate from `_dirichlet_partition` rather than a branch inside
    it. The two answer different questions and every result already reported used
    the other one; folding them together would make it impossible to say which
    partition a finished run had used.
    """
    rng = np.random.RandomState(seed)
    all_users = np.array(all_users)
    if labels is None:
        labels = _user_behaviour_clusters(df, all_users, num_clusters, seed)
    num_clusters = int(labels.max()) + 1

    chunks = [[] for _ in range(num_partitions)]
    for cluster in range(num_clusters):
        members = rng.permutation(all_users[labels == cluster])
        if len(members) == 0:
            continue
        proportions = rng.dirichlet(np.repeat(alpha, num_partitions))
        # cumsum rather than per-client counts, so rounding cannot lose or
        # duplicate a user: the split points are indices into one shuffled array.
        cuts = (np.cumsum(proportions) * len(members)).astype(int)[:-1]
        for client, piece in enumerate(np.split(members, cuts)):
            chunks[client].extend(piece)

    # A client with nothing trains on nothing and contributes an untrained model
    # to the average, which is a bug rather than a form of heterogeneity. Users
    # are taken from the largest client, which is the one that misses them least.
    for client, users in enumerate(chunks):
        if users:
            continue
        donor = max(range(num_partitions), key=lambda i: len(chunks[i]))
        chunks[client].append(chunks[donor].pop())

    return [np.array(sorted(users)) for users in chunks]


def drop_constant_features(df, features):
    """Remove features with no spread: the variance filter, applied to the data.

    Four of the configured 50 are zero on every row of the dataset, so they carry
    no information at any point. They cost the model a little capacity and they
    are what forced the scaler's variance floor into existence.

    Uses the same threshold as the scaler, so the two cannot disagree about what
    counts as constant. See `analysis/analyze_features.py` for the measured list.
    """
    tolerance = scaling.constant_feature_tolerance()
    kept = []
    dropped = []
    for name in features:
        column = pd.to_numeric(df[name], errors="coerce").fillna(0)
        (kept if column.std() >= tolerance else dropped).append(name)

    if dropped:
        # Plain ASCII: this runs inside Ray workers whose stdout may be a legacy
        # Windows code page.
        print(f"[features] variance filter dropped {len(dropped)}: "
              f"{', '.join(dropped)}")
    return kept


def select_features(df):
    """The feature columns, from the configured list or by excluding metadata.

    The configured path is the one every reported run takes: pyproject.toml names
    50 features explicitly, so the model input cannot drift when the extraction
    script gains a column. A configured feature the frame does not have is
    dropped rather than filled with zeros, because a silently zero-filled column
    would shift every score without failing anything.

    The variance filter is off by default, because turning it on changes the
    input dimension and therefore the parameter count that every communication
    figure derives from. It exists so that the effect can be measured rather than
    argued about.
    """
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider',
                'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    if SELECTED_FEATURES:
        features = [c for c in SELECTED_FEATURES if c in df.columns]
    else:
        features = [c for c in df.columns if c not in metadata]

    if config.get("data", "drop_constant_features"):
        features = drop_constant_features(df, features)
    return features


def partition_users(df, num_partitions):
    """Assign users to clients, evenly or by a Dirichlet draw.

    Partitioning by user, never by row, is what makes this a plausible federation:
    a client stands for an organisation holding its own people's activity, and one
    user's days never appear at two clients.

    The even path sorts first, so client 0 always holds the same users given the
    same dataset. Deterministic rather than random on purpose: a partition that
    moved between runs would make two seeds differ for two reasons at once.
    """
    all_users = sorted(df['user'].unique())
    if not config.get("data", "is_non_iid"):
        return np.array_split(all_users, num_partitions)

    alpha = config.get("data", "non_iid_alpha")
    mode = config.get("data", "non_iid_mode")
    if mode == "quantity":
        return _dirichlet_partition(all_users, num_partitions, alpha=alpha,
                                    seed=config.seed)
    if mode in ("role", "cluster"):
        labels = (_role_labels(df, all_users) if mode == "role" else None)
        return _cluster_dirichlet_partition(
            df, all_users, num_partitions, alpha=alpha,
            num_clusters=config.get("data", "non_iid_clusters"),
            seed=config.seed, labels=labels)
    raise ValueError(
        f"[tool.fueba.data] non_iid_mode is {mode!r}; expected 'quantity' (skew "
        f"how many users a client holds), 'role' (skew which organisational "
        f"unit) or 'cluster' (skew which behavioural group). These are different "
        f"experiments, so an unrecognised value stops rather than falling back "
        f"to any of them.")


def load_dataset(input_path):
    """Read the daily feature table, once per process. See `_cached_df`."""
    global _cached_df
    if _cached_df is None:
        _cached_df = (pd.read_pickle(input_path) if input_path.endswith('.pkl')
                      else pd.read_csv(input_path))
    return _cached_df


def input_dimension(input_path=None):
    """How many features the model actually takes, after every filter.

    Not `len(selected_features)`. With `drop_constant_features` on, the variance
    filter removes the four features that are zero on every row and the input is
    46 rather than 50, so anything that sizes a model from the configured list
    builds a model 50 wide while the clients build one 46 wide.

    That is not hypothetical: the server used to size its initial global model
    from the configured list, and every client in the `features-filtered`
    experiment then failed to load the broadcast with a shape mismatch, every
    round, for the whole run. Both sides go through `select_features` now, so
    there is one answer rather than two.

    Costs one read of the dataset in the calling process, and nothing after that:
    `load_dataset` caches it, and the server needs the frame only for this.
    """
    if input_path is None:
        input_path = config.get("data", "processed_data_path")
    return len(select_features(load_dataset(input_path)))


def build_windows(values):
    """Sliding windows over one user's chronologically ordered rows.

    At WINDOW_SIZE 14 and STRIDE 1 a user with N days yields N - 13 windows, and
    a user with fewer than 14 days yields none at all and drops out. Consecutive
    windows therefore share 13 of their 14 days; the overlap is intended, since a
    single unusual day should be visible in every window that contains it.
    """
    return [values[i:i + WINDOW_SIZE]
            for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE)]


def load_partitioned_data(input_path, partition_id, num_partitions):
    """Build one client's train and validation loaders.

    Returns `(train_loader, val_loader, feature_count)`, or `(None, None, count)`
    when this client has no usable windows. That happens under a heterogeneous
    partition where a client can draw too few users to fill a single 14-day
    window; the caller gives it empty loaders so the federation keeps running.
    """
    df = load_dataset(input_path)
    features = select_features(df)
    user_chunks = partition_users(df, num_partitions)

    # One scaler for the whole federation, built from statistics each client can
    # contribute without revealing a record. See federated_ueba.scaling for why
    # per-client scalers put every client in a different input space.
    scaler = scaling.load_or_build_global_scaler(
        df, features, user_chunks, config.get("data", "global_scaler_path"))

    client_df = df[df['user'].isin(user_chunks[partition_id])]

    # The model only ever learns from normal behaviour, so every labelled insider
    # day is dropped. Those users' ordinary days stay in: the point is to learn
    # what a normal fortnight looks like for them too, otherwise their anomalous
    # days would have nothing to stand out against.
    if 'insider' in client_df.columns:
        client_df = client_df[client_df['insider'] == 0]
    if client_df.empty:
        return None, None, len(features)

    # Grouped by user so a window never spans two people's days. Scaling happens
    # per user only for convenience; the scaler itself is the federation's one
    # global scaler, so every client ends up in the same input space.
    sequences = []
    for _, group in client_df.groupby('user'):
        scaled = scaler.transform(scaling.prepare_features(group, features))
        sequences.extend(build_windows(scaled))

    if not sequences:
        return None, None, len(features)

    # A local split for the training loop only. It divides windows rather than
    # users, and since consecutive windows share 13 of 14 days the two sides hold
    # near-duplicates: the resulting validation loss is optimistic and is not an
    # independent measurement. Nothing reported depends on it. The round that is
    # actually selected is chosen by PR-AUC over a validation half of the *users*,
    # computed after training in `federated_ueba.scoring`.
    train_seq, val_seq = train_test_split(np.array(sequences), test_size=TEST_SIZE,
                                          random_state=config.seed)

    # An explicit generator rather than the global one. Shuffling would already
    # reproduce, because the client seeds torch before training and the sequence
    # of draws is fixed, but it would share that stream with dropout and the
    # denoising noise: adding a dropout layer would then silently change the
    # batch order too. Its own generator makes shuffling depend on the seed and
    # nothing else.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seeding.derive_seed(config.seed, len(train_seq)))

    train_loader = torch.utils.data.DataLoader(
        torch.tensor(train_seq, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_generator)
    val_loader = torch.utils.data.DataLoader(
        torch.tensor(val_seq, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, len(features)


def train(net, trainloader, valloader, epochs, proximal_mu=0.0):
    """One client's local training for this round.

    The target is the input: an autoencoder is trained to reproduce what it is
    given, and MSE against the input is the reconstruction error the detector
    later reads. `epochs` is the local epoch count (5 in every reported run), run
    once per federated round.

    `proximal_mu` above zero adds the FedProx penalty for drifting away from the
    global model this round started from. At zero, which is what every FedAvg run
    passes, the snapshot is not even taken and the loss is exactly what it was
    before FedProx existed in this file.
    """
    net.to(DEVICE)
    criterion = nn.MSELoss()
    learning_rate = config.get("training", "learning_rate")
    # Weight decay alongside the bottleneck and the input noise: three mild
    # pressures against the model simply memorising its input.
    weight_decay = config.get("training", "weight_decay")
    grad_clip_norm = config.get("training", "grad_clip_norm")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    # Taken once, before the first step, because after that the model no longer
    # holds the weights the server sent and the penalty would be measured against
    # a moving target. Skipped entirely at mu = 0 so FedAvg pays nothing for
    # FedProx being available.
    global_snapshot = proximal.snapshot_global(net) if proximal_mu > 0 else None

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for batch in trainloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(batch), batch)
            if global_snapshot is not None:
                loss = loss + proximal.proximal_penalty(
                    net.parameters(), global_snapshot, proximal_mu)
            loss.backward()
            # LSTMs on 14-step sequences can produce very large gradients on an
            # outlying window; unclipped, one such batch moves the weights far
            # enough that the round is wasted.
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()

        val_loss = test(net, valloader)
        print(f"  Epoch {epoch+1}: Train Loss: {train_loss/len(trainloader):.6f} "
              f"| Val Loss: {val_loss:.6f}")


def test(net, testloader):
    """Mean reconstruction loss over a loader. Reported to the server each round."""
    if testloader is None:
        return 0.0

    net.to(DEVICE)
    criterion = nn.MSELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(DEVICE)
            loss += criterion(net(batch), batch).item()
    return loss / len(testloader) if len(testloader) > 0 else 0


def get_error_distribution(net, trainloader):
    """Per-feature mean and standard deviation of this client's reconstruction error.

    This is what calibrates the Z-score stage of scoring. Without it a feature
    the model reconstructs poorly for everyone would look anomalous for everyone;
    the Z-score asks instead whether *this* window's error on *this* feature is
    unusual against normal behaviour.

    Measured on the training data, which is normal by construction, and never on
    anything labelled. Averaged over the timesteps of each window (dim=1), so the
    result has one value per feature.
    """
    net.eval()
    all_errors = []
    with torch.no_grad():
        for batch in trainloader:
            batch = batch.to(DEVICE)
            squared_error = torch.mean((net(batch) - batch) ** 2, dim=1)
            all_errors.append(squared_error.cpu().numpy())

    # An empty loader means this client had no windows. Neutral statistics keep
    # the round running; the client simply contributes nothing.
    if not all_errors:
        return np.zeros(1), np.ones(1)

    all_errors = np.concatenate(all_errors, axis=0)
    return np.mean(all_errors, axis=0), np.std(all_errors, axis=0)
