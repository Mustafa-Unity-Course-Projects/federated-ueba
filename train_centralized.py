"""Centralized baseline: the same model and scoring pipeline without federation.

This exists to answer "what does federating cost us?", so it must stay
metric-for-metric comparable with the federated runs. It therefore shares the
scoring pipeline (`federated_ueba.scoring`), the model (`federated_ueba.task`)
and the seed with them, rather than keeping its own copy of any of the three.
"""

import copy
import json
import os
import sys
import pickle
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import federated_ueba.task as task
from config_manager import PIPELINE_VERSION, config
from federated_ueba import scaling, scoring

# --- CONFIGURATION ---
# Force UTF-8 on stdout/stderr: the emoji in the progress messages raise
# UnicodeEncodeError under the legacy Windows code page used when output is
# redirected to a log file.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DATA_PATH = config.get("data", "processed_data_path")
# Resolved by `task` rather than here, so that `[tool.fueba.experiment] device`
# means the same thing in both arms. This used to test cuda.is_available()
# directly and so ignored a configured device.
DEVICE = task.DEVICE
BATCH_SIZE = config.get("data", "batch_size")
SELECTED_FEATURES = config.get("data", "selected_features")
HIDDEN_DIM = config.get("model", "hidden_dim")

# Shared with the federated clients. See [tool.fueba.training].
LEARNING_RATE = config.get("training", "learning_rate")
WEIGHT_DECAY = config.get("training", "weight_decay")
GRAD_CLIP_NORM = config.get("training", "grad_clip_norm")

# This arm only. See [tool.fueba.centralized].
EPOCHS = config.get("centralized", "epochs")
EARLY_STOPPING_PATIENCE = config.get("centralized", "early_stopping_patience")
LR_SCHEDULE_FACTOR = config.get("centralized", "lr_schedule_factor")
LR_SCHEDULE_PATIENCE = config.get("centralized", "lr_schedule_patience")
RANDOM_SEED = config.seed

# Artefacts are seed-suffixed for the same reason the federated ones are: so a
# second seed cannot silently overwrite the first one's model and metrics.
SUFFIX = f"__seed{RANDOM_SEED}"
CENTRALIZED_MODEL_PATH = f"centralized_model{SUFFIX}.pth"
CENTRALIZED_SCALER_PATH = f"centralized_scaler{SUFFIX}.pkl"
CENTRALIZED_ERROR_STATS_PATH = f"centralized_error_stats{SUFFIX}.pkl"

CENTRALIZED_REPORT_DIR = os.path.join("centralized_evaluation_reports", f"seed{RANDOM_SEED}")
os.makedirs(CENTRALIZED_REPORT_DIR, exist_ok=True)

SCORING_CFG = scoring.ScoringConfig.from_config(config)


def calculate_error_stats(model, dataloader):
    """Per-feature reconstruction error mean/std over the training set.

    This is the reference distribution the Z-score stage calibrates against.
    Computed from the *train* loader to match what the federated clients do in
    task.get_error_distribution.
    """
    mean_per_feature, std_per_feature = task.get_error_distribution(model, dataloader)
    return {"mean_per_feature": mean_per_feature, "std_per_feature": std_per_feature}


@dataclass(frozen=True)
class DetectionEvaluator:
    """Scores every user, given a model and the epoch's error reference.

    Exists so the training loop takes one collaborator instead of three loose
    values it only forwards. `df`, `scaler` and `features` never change during
    training; only the error statistics do, and those arrive per call because
    they are measured on the model as it stands at that epoch.
    """

    df: object
    scaler: object
    features: list

    def __call__(self, model, error_stats):
        return evaluate_anomaly_detection(model, self.df, self.scaler,
                                          error_stats, self.features)


def evaluate_anomaly_detection(model, df_full, scaler, error_stats, features):
    """Score every user and report the same metrics as the federated runs."""
    scorer = scoring.Scorer(
        features=features, scaler=scaler,
        ref_mean=error_stats["mean_per_feature"],
        ref_std=error_stats["std_per_feature"],
        cfg=SCORING_CFG, window_size=task.WINDOW_SIZE, device=DEVICE)
    results = scorer.scan(model, df_full, sorted(df_full['user'].unique()))

    # The split seed, not the run seed: the centralized arm has to report on the
    # same test users as the federated one or the comparison between them is
    # partly a comparison of two different halves of the population.
    m = scoring.evaluate_scores(results, seed=SCORING_CFG.split_seed,
                                validation_fraction=SCORING_CFG.validation_fraction)
    return {
        "pr_auc": m["pr_auc"],
        "pr_auc_val": m["pr_auc_val"],
        "pr_auc_test": m["pr_auc_test"],
        "pr_auc_all": m["pr_auc_all"],
        "best_f1": m["f1"],
        "optimal_threshold": m["threshold"],
        "precision_at_best_f1": m["precision"],
        "recall_at_best_f1": m["recall"],
        "accuracy_at_best_f1": m["accuracy"],
        "balanced_accuracy_at_best_f1": m["balanced_accuracy"],
        "tp_at_best_f1": m["tp"],
        "fp_at_best_f1": m["fp"],
        "tn_at_best_f1": m["tn"],
        "fn_at_best_f1": m["fn"],
    }, results


# --- Data preparation ------------------------------------------------------

def set_seeds():
    """Seed every generator the run touches, in one place."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"⚙️ Random seed set to {RANDOM_SEED} for reproducibility.")


def select_features(df_full):
    """Feature columns, from the configured list or by excluding metadata."""
    metadata = ['user', 'day', 'week', 'pc', 'activity', 'id', 'label', 'insider',
                'to', 'from', 'starttime', 'endtime', 'pcid', 'time_stamp', 'actid']
    if SELECTED_FEATURES:
        return [c for c in SELECTED_FEATURES if c in df_full.columns]
    return [c for c in df_full.columns if c not in metadata]


def fit_scaler_on_normal_behaviour(df_full, features):
    """Scale the normal-behaviour rows and keep the scaler that did it.

    Built by the same routine the federated clients use, over a single partition
    holding every user. That is what the federated global scaler reduces to when
    there is one client, so both pipelines score in the same input space and the
    comparison between them stays meaningful.
    """
    scaler = scaling.build_global_scaler(
        df_full, features, [sorted(df_full['user'].unique())])

    with open(CENTRALIZED_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to {CENTRALIZED_SCALER_PATH}")

    df_train = (df_full[df_full['insider'] == 0].copy()
                if 'insider' in df_full.columns else df_full.copy())
    df_train[features] = scaler.transform(scaling.prepare_features(df_train, features))
    return df_train, scaler


def build_sequences(df_train, features):
    """Sliding windows per user, in day order."""
    sequences = []
    for user in df_train['user'].unique():
        user_data = df_train[df_train['user'] == user].sort_values('day')[features].values
        if len(user_data) >= task.WINDOW_SIZE:
            for i in range(0, len(user_data) - task.WINDOW_SIZE + 1, task.STRIDE):
                sequences.append(user_data[i:i + task.WINDOW_SIZE])
    return np.array(sequences)


def build_loaders(sequences):
    """Split windows, not users, into train and validation. Matches task.py."""
    train_seqs, val_seqs = train_test_split(sequences, test_size=0.2,
                                            random_state=RANDOM_SEED)
    train_loader = torch.utils.data.DataLoader(
        torch.tensor(train_seqs, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.tensor(val_seqs, dtype=torch.float32),
        batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


# --- Training --------------------------------------------------------------

@dataclass(frozen=True)
class Optimisation:
    """How the model is trained, as opposed to what is trained.

    The three always come from `build_model` together and are only ever used
    together, so the training loop takes one of these rather than three
    parameters it does nothing with except call.
    """

    criterion: object
    optimizer: object
    scheduler: object


def build_model(n_features):
    """The same architecture the federated clients train, with a scheduler.

    Returns `(model, Optimisation)`. The architecture has to match the federated
    one exactly, otherwise the centralized-versus-federated comparison would be
    measuring two different models rather than two training regimes.
    """
    model = task.LSTMAutoencoder(input_dim=n_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    # The same optimiser settings the clients use, read from the same place. They
    # used to be separate literals here, and the drift was invisible: this arm
    # trained at weight_decay 1e-5 against the federation's 1e-4, so a difference
    # in the reported detection quality could not be attributed to federation
    # alone, which is the only thing the comparison is supposed to isolate.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    # The one deliberate difference, and it is a difference in what is possible
    # rather than in configuration: no client ever trains for more than 5
    # consecutive epochs, so a plateau schedule has nothing to observe there.
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=LR_SCHEDULE_FACTOR,
                                  patience=LR_SCHEDULE_PATIENCE, verbose=True)
    return model, Optimisation(criterion=criterion, optimizer=optimizer,
                               scheduler=scheduler)


def run_train_epoch(model, loader, criterion, optimizer):
    """One pass over the training windows. Returns the mean loss."""
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        # The target is the input: this is an autoencoder, and the loss it
        # minimises is exactly the reconstruction error detection later reads.
        loss = criterion(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def run_validation(model, loader, criterion):
    """Mean reconstruction loss with the model in eval mode and no gradients.

    Eval mode matters beyond speed: it disables the dropout and the input noise
    the model trains with, so this measures the model as it will actually score.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            total += criterion(model(batch), batch).item()
    return total / len(loader)


@dataclass
class TrainingOutcome:
    """Everything the training loop produces that the later stages need."""
    history: list
    best_epoch: int
    best_val_pr_auc: float
    best_val_loss: float
    best_model_state: dict
    best_error_stats: dict
    best_metrics: dict
    best_results: pd.DataFrame


def train_model(model, optimisation, train_loader, val_loader, evaluate):
    """Train until convergence, keeping the checkpoint that detects best.

    Two separate notions of "best", and they come apart sharply here:

      best_val_loss     drives early stopping only. It says when the autoencoder
                        has stopped learning to reconstruct.
      best_val_pr_auc   decides which checkpoint is reported, using the
                        validation user half.

    As reconstruction keeps improving the model also learns to reconstruct
    anomalies, the error gap the detector relies on closes, and detection
    degrades. Selecting on validation loss picked the *worst* detector and made
    the centralized baseline look far weaker than it is.
    """
    best = TrainingOutcome(history=[], best_epoch=0, best_val_pr_auc=-1.0,
                           best_val_loss=float('inf'),
                           best_model_state=copy.deepcopy(model.state_dict()),
                           best_error_stats=None, best_metrics=None,
                           best_results=None)
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        avg_train_loss = run_train_epoch(model, train_loader,
                                         optimisation.criterion,
                                         optimisation.optimizer)
        avg_val_loss = run_validation(model, val_loader, optimisation.criterion)

        # Error stats come from the TRAIN loader, matching the federated clients
        error_stats = calculate_error_stats(model, train_loader)
        metrics, results = evaluate(model, error_stats)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} "
              f"| Val Loss: {avg_val_loss:.6f} | F1: {metrics['best_f1']:.4f} "
              f"| PR-AUC: {metrics['pr_auc']:.4f}")

        best.history.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "F1_Score": metrics['best_f1'],
            "PR_AUC": metrics['pr_auc'],
            "PR_AUC_val": metrics['pr_auc_val'],
            "PR_AUC_all": metrics['pr_auc_all'],
        })

        optimisation.scheduler.step(avg_val_loss)

        # Reported checkpoint: best detection on the validation half.
        if metrics['pr_auc_val'] > best.best_val_pr_auc:
            best.best_val_pr_auc = metrics['pr_auc_val']
            best.best_epoch = epoch + 1
            best.best_model_state = copy.deepcopy(model.state_dict())
            # Error stats and per-user scores must describe the checkpoint we
            # ship, or the bootstrap interval belongs to a different model.
            best.best_error_stats = error_stats
            best.best_metrics = metrics
            best.best_results = results

        # Early stopping: independent of the above, purely about convergence.
        if avg_val_loss < best.best_val_loss:
            best.best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    return best


# --- Reporting -------------------------------------------------------------

def save_model_artifacts(model, outcome):
    """Persist the selected checkpoint and the error statistics that go with it.

    The two must come from the same epoch. The error statistics calibrate the
    Z-score stage, so pairing them with a different epoch's weights would score
    every user against a distribution the model never produced.
    """
    model.load_state_dict(outcome.best_model_state)
    torch.save(model.state_dict(), CENTRALIZED_MODEL_PATH)
    print(f"💾 Model saved to {CENTRALIZED_MODEL_PATH}")

    with open(CENTRALIZED_ERROR_STATS_PATH, "wb") as f:
        pickle.dump(outcome.best_error_stats, f)
    print(f"💾 Optimized Stats saved to {CENTRALIZED_ERROR_STATS_PATH}")


def _save_figure(name):
    """Write the current figure into the report directory and close it."""
    path = os.path.join(CENTRALIZED_REPORT_DIR, name)
    plt.savefig(path)
    # Closed explicitly: matplotlib keeps every unclosed figure in memory, and
    # this module draws several per run.
    plt.close()
    print(f"💾 Saved {path}")


def plot_learning_curves(history_df):
    """Train and validation loss against epoch.

    Unlike the federated equivalent, this validation split is clean, so this is
    the figure to use when the question is whether the model overfits.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['Epoch'], history_df['Train Loss'], label='Train Loss')
    plt.plot(history_df['Epoch'], history_df['Val Loss'], label='Validation Loss')
    plt.title('Centralized Model Learning Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_figure('centralized_learning_progress.png')


def plot_detection_over_epochs(history_df):
    """Detection against epoch.

    This is the figure that shows the two curves diverging: the loss keeps
    falling while PR-AUC turns over. It is the evidence that the late-training
    decline is not overfitting.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['Epoch'], history_df['F1_Score'], label='F1 Score',
             marker='o', markersize=4)
    plt.plot(history_df['Epoch'], history_df['PR_AUC'], label='PR-AUC',
             marker='x', markersize=4)
    plt.title('Centralized Anomaly Detection Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    _save_figure('centralized_ad_metrics_by_epoch.png')


def plot_final_metrics(metrics):
    """Bar chart of the selected checkpoint's headline metrics."""
    names = ['PR-AUC', 'Max-F1', 'Balanced Accuracy']
    values = [metrics['pr_auc'], metrics['best_f1'],
              metrics['balanced_accuracy_at_best_f1']]

    plt.figure(figsize=(8, 6))
    plt.bar(names, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title('Centralized Final Anomaly Detection Metrics (Best Model)')
    plt.ylabel('Score')
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f'{value:.4f}', ha='center', va='bottom')
    _save_figure('centralized_final_metrics.png')


def write_summary(model, outcome, features, results_path):
    """Write centralized_experiment_summary.json.

    Deliberately parallel to the federated runner's summary: the fields the two
    have in common carry the same meaning, which is what lets
    `analysis/compare_experiments.py` put both in one table.
    """
    total_params = int(sum(p.numel() for p in model.parameters()))
    summary = {
        # Stamped so a comparison table cannot mix this run with results from a
        # pipeline version whose numbers mean something else. See config_manager.
        "pipeline_version": PIPELINE_VERSION,
        "model_path": CENTRALIZED_MODEL_PATH,
        "scaler_path": CENTRALIZED_SCALER_PATH,
        "error_stats_path": CENTRALIZED_ERROR_STATS_PATH,
        "results_path": results_path,
        "seed": int(RANDOM_SEED),
        "selection_rule": "max PR-AUC on the validation user half; "
                          "reported metrics come from the test half",
        "best_epoch": int(outcome.best_epoch),
        "best_validation_pr_auc": float(outcome.best_val_pr_auc),
        "best_validation_loss": float(outcome.best_val_loss),
        "epochs_trained": int(len(outcome.history)),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "epochs_config": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "hidden_dim": int(HIDDEN_DIM),
        "input_features_count": int(len(features)),
        "window_size": int(task.WINDOW_SIZE),
        # Payload size of the model; see the federated runner for why this is
        # recorded alongside the metrics.
        "model_parameters_total": total_params,
        "model_dense_payload_mb": round(total_params * 4 / 1024 / 1024, 6),
        "input_matrix_shape": [int(task.WINDOW_SIZE), int(len(features))],
        "stride": int(task.STRIDE),
        # Scoring pipeline settings. `top_k_features` is the number of features
        # the score focuses on; it is unrelated to [efficiency] top_k_ratio,
        # which is the communication sparsification ratio. An earlier version
        # wrote this value under the name "top_k_ratio".
        "top_k_features": int(SCORING_CFG.top_k_features),
        "persistence_window": int(SCORING_CFG.persistence_window),
        "diversity_threshold": float(SCORING_CFG.diversity_threshold),
        "device": str(DEVICE),
        "anomaly_detection_metrics": outcome.best_metrics,
    }
    path = os.path.join(CENTRALIZED_REPORT_DIR, "centralized_experiment_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"💾 Experiment summary saved to {path}")


def write_reports(model, outcome, features):
    """Every artefact this run leaves behind: curves, figures, scores, summary."""
    history_df = pd.DataFrame(outcome.history)
    progress_path = os.path.join(CENTRALIZED_REPORT_DIR,
                                 "centralized_training_progress.csv")
    history_df.to_csv(progress_path, index=False)
    print(f"💾 Training progress saved to {progress_path}")

    plot_learning_curves(history_df)
    plot_detection_over_epochs(history_df)
    plot_final_metrics(outcome.best_metrics)

    # Per-user scores of the best model. analysis/compare_experiments.py bootstraps its
    # confidence interval from this file, so it must describe the same model the
    # summary reports; previously it was left over from an earlier run and the
    # interval belonged to a different model than the point estimate.
    results_path = os.path.join(CENTRALIZED_REPORT_DIR,
                                "centralized_insider_results.csv")
    outcome.best_results.to_csv(results_path, index=False)
    print(f"💾 Per-user scores saved to {results_path}")

    write_summary(model, outcome, features, results_path)


def train_centralized():
    """Train the single-model baseline the federated results are compared against.

    Same architecture, same features, same windowing and same scoring pipeline as
    the federated runs. The only difference is that one model sees every user's
    data at once, which is the comparison the thesis is making.
    """
    print(f"🚀 Starting Bidirectional Centralized Model Training on {DEVICE}...")
    set_seeds()

    df_full = pd.read_csv(DATA_PATH)
    features = select_features(df_full)
    print(f"📊 Training with {len(features)} features.")

    df_train, scaler = fit_scaler_on_normal_behaviour(df_full, features)
    train_loader, val_loader = build_loaders(build_sequences(df_train, features))

    model, optimisation = build_model(len(features))
    outcome = train_model(
        model, optimisation, train_loader, val_loader,
        DetectionEvaluator(df=df_full, scaler=scaler, features=features))

    save_model_artifacts(model, outcome)
    write_reports(model, outcome, features)


if __name__ == "__main__":
    train_centralized()
