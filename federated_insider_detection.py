"""Experiment runner: trains a federated configuration, then scores every user.

Training and evaluation are deliberately separate. Training runs the Flower
simulation and checkpoints the global model every round; evaluation reloads each
even-numbered checkpoint and rescores all 1000 users through the pipeline in
`federated_ueba.scoring`.

Artefacts are written under a seed-qualified run id (`baseline__seed42`), so
running the same experiment under several seeds never overwrites earlier results.

  python federated_insider_detection.py --experiment all --seeds 1,2,3
  python federated_insider_detection.py --experiment baseline --mode full
  python federated_insider_detection.py --experiment ablation-full --mode eval
"""

import argparse
import glob
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Safe to import at module scope: unlike `federated_ueba.task`, this module reads
# no configuration when it loads, so it does not care whether the environment
# variables that select an experiment have been set yet.
from federated_ueba import scoring, seeding, strategy

# Progress messages contain emoji. On Windows the console falls back to a legacy
# code page as soon as output is redirected, and any such message then raises
# UnicodeEncodeError. That is fatal rather than cosmetic: it kills the Flower
# ServerApp thread before the first round.
#
# Two layers are needed. Reconfiguring the streams fixes this process; exporting
# PYTHONIOENCODING fixes the child processes (the per-experiment subprocess and
# the flower-simulation it launches), which are separate interpreters and do not
# inherit the reconfiguration.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_REPORT_DIR = "federated_evaluation_reports"

# Configurations that exist for checking the pipeline, not for reporting. They
# are never part of `--experiment all`, so a sweep cannot pick them up and no
# comparison table can end up quoting their meaningless numbers.
NOT_IN_SWEEP = {"smoke-test", "smoke-test-downlink", "smoke-test-fedprox"}

# How many experiments may fail back to back before the sweep gives up. One
# failure can be a flaky run worth stepping over; two in a row means the
# environment is broken and every remaining experiment would fail the same way.
MAX_CONSECUTIVE_FAILURES = 2

# Files an evaluation pass produces. These are deleted before every evaluation so
# that a shorter run can never inherit round files from a longer earlier one.
EVAL_ARTIFACTS = ["federated_rounds_comparison.csv", "experiment_summary.json",
                  "learning_progress.png"]


def load_error_reference(scaler_dir, n_features):
    """Average the per-client reconstruction error statistics.

    These calibrate the Z-score stage. Each client writes its own after training;
    averaging them gives one reference for all users rather than picking an
    arbitrary client, which is what an earlier version did.
    """
    parts = sorted(glob.glob(os.path.join(scaler_dir, "error_stats_client_*.pkl")))
    means, stds = [], []
    for path in parts:
        with open(path, "rb") as f:
            stats = pickle.load(f)
        means.append(stats["mean_per_feature"])
        stds.append(stats["std_per_feature"])

    if not means:
        from config_manager import config
        # Only reachable when a run's per-client statistics are missing, which
        # means the z-score stage is calibrated against a guess rather than
        # against measured normal behaviour. Configured rather than hardcoded so
        # that the guess is at least visible and named.
        print(f"⚠️ No error statistics under {scaler_dir}. Using the configured "
              f"fallback; scores from this run are not calibrated.")
        return (np.full(n_features, config.get("scoring", "fallback_error_mean")),
                np.full(n_features, config.get("scoring", "fallback_error_std")))

    print(f"📐 Error reference averaged over {len(means)} clients.")
    return np.mean(means, axis=0), np.mean(stds, axis=0)


# The two definitions of "converged" below. Both are reported because they answer
# different questions and the difference between them is itself informative: the
# first says when the score got close to its eventual best, the second says when
# training stopped buying anything. Both read the validation half, for the same
# reason checkpoint selection does.
# Read on use rather than at import. `config_manager` builds its singleton the
# moment it is imported, and that has to happen after the environment variables
# that select the experiment and seed are set, so this module cannot read
# configuration at module scope. `None` here means "ask the configuration", which
# keeps both functions callable with an explicit value from a test.


# The untrained model. It is scored, logged and kept, but it is never allowed to
# win a selection: the clients measure their reference error distribution with
# their *trained* local models, so round 0 is graded against a yardstick it never
# earned and comes out artificially high. Left in, it wins the argmax outright —
# 21 of the first 63 finished runs recorded `best_round = 0`, all three baseline
# seeds among them, which would have put an untrained model on the reference side
# of every comparison. `analysis/seed_variance.py` drops it for the same reason.
UNTRAINED_ROUND = 0


def trained_rounds(rounds, scores):
    """`(rounds, scores)` with the untrained round removed, order preserved."""
    kept = [(r, s) for r, s in zip(rounds, scores) if r != UNTRAINED_ROUND]
    if not kept:
        return [], []
    return [r for r, _ in kept], [s for _, s in kept]


def find_convergence_round(rounds, scores, fraction=None):
    """First round whose score reaches `fraction` of the peak score.

    This is the definition written in the thesis (§4.3) and it is the one
    reported as `convergence_round`. It is sensitive to a single lucky round: a
    noisy early spike close to the peak makes it fire immediately, which is why
    `find_plateau_round` is reported alongside it.
    """
    if not rounds:
        return None
    if fraction is None:
        from config_manager import config
        fraction = config.get("evaluation", "convergence_peak_fraction")
    target = max(scores) * fraction
    for round_number, score in zip(rounds, scores):
        if score >= target:
            return int(round_number)
    return int(rounds[-1])


def find_plateau_round(rounds, scores, tolerance=None):
    """First round after which no later round improves by more than `tolerance`.

    The stricter reading of convergence: further rounds are still bought and paid
    for in communication, but they no longer return anything.
    """
    if tolerance is None:
        from config_manager import config
        tolerance = config.get("evaluation", "plateau_tolerance")

    for i, (round_number, score) in enumerate(zip(rounds, scores)):
        if all(other <= score + tolerance for other in scores[i + 1:]):
            return int(round_number)
    return int(rounds[-1]) if rounds else None


def set_all_seeds(seed):
    """Seed this process. The server and the clients seed themselves separately.

    This covers the runner only: partitioning, the evaluation user split and the
    scoring pass. `flower-simulation` is a separate interpreter and its Ray
    actors are separate processes again, so none of them inherit this. See
    `federated_ueba.seeding` for how each is handled.
    """
    seeding.seed_everything(seed)
    if torch.cuda.is_available():
        # Traded for determinism: without these, cudnn is free to pick a
        # different algorithm per run and the same seed diverges on a GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class RunLayout:
    """Everything one experiment needs to know about itself, resolved once.

    These used to be a dozen loose locals inside a 250-line function, which is
    why the pieces below could not be tested or read in isolation. Resolving them
    in one place also puts the train/eval directory distinction somewhere visible
    rather than in the middle of the control flow.
    """

    experiment_name: str
    run_id: str                 # e.g. baseline__seed42
    seed: int
    mode: str                   # full, train or eval
    data_path: str
    hidden_dim: int
    num_supernodes: int
    ablation_source: str        # empty unless this run rescores another's weights

    # Where this run writes.
    report_dir: str
    round_results_dir: str
    summary_path: str
    train_save_path: str
    train_scaler_dir: str

    # Where evaluation reads from. The same as the train directories, except for
    # an ablation, which reads the weights of the experiment it rescores.
    eval_save_path: str
    eval_scaler_dir: str


def resolve_run_layout(config, experiment_name, mode, seed):
    """Work out this run's directories and settings from the active config."""
    run_id = config.run_id
    report_dir = os.path.join(BASE_REPORT_DIR, run_id)

    train_save_path = config.get("federation", "save_path")
    train_scaler_dir = config.get("data", "scaler_dir")

    # An ablation rescores an already-trained experiment rather than training its
    # own model, so it reads that experiment's directories and must never write
    # to or delete them. Asking one to train is a configuration mistake, not a
    # request: there is nothing for it to train that the source has not trained.
    ablation_source = config.get("scoring", "ablation_source")
    if ablation_source:
        source_id = config.qualify(ablation_source)
        eval_save_path = os.path.join("./model_pickle", source_id)
        eval_scaler_dir = os.path.join("scaler_data", source_id)
        if mode in ("full", "train"):
            print(f"⚠️  '{experiment_name}' reuses the weights of '{source_id}' "
                  f"and cannot train. Falling back to --mode eval.")
            mode = "eval"
    else:
        eval_save_path = train_save_path
        eval_scaler_dir = train_scaler_dir

    # Federation size: an experiment may declare its own (the node-count
    # sensitivity configs do), otherwise the pyproject default applies. Passed
    # straight to flower-simulation, so changing it needs no manual edit.
    num_supernodes = (config.get("federation", "num_supernodes")
                      or config.get_pyproject("tool", "flwr", "federations",
                                              "local-simulation", "options",
                                              "num-supernodes")
                      or 10)

    return RunLayout(
        experiment_name=experiment_name,
        run_id=run_id,
        seed=seed,
        mode=mode,
        data_path=config.get("data", "processed_data_path"),
        hidden_dim=config.get("model", "hidden_dim"),
        num_supernodes=num_supernodes,
        ablation_source=ablation_source,
        report_dir=report_dir,
        round_results_dir=os.path.join(report_dir, "round_by_round_results"),
        summary_path=os.path.join(report_dir, "experiment_summary.json"),
        train_save_path=train_save_path,
        train_scaler_dir=train_scaler_dir,
        eval_save_path=eval_save_path,
        eval_scaler_dir=eval_scaler_dir,
    )


def clean_evaluation_artifacts(layout):
    """Remove everything a previous evaluation wrote, keeping the comm log.

    Necessary because a shorter run would otherwise inherit the round files of a
    longer earlier one and report a mixture of the two. The communication log is
    deliberately spared: it is written during training, not evaluation, and an
    eval-only pass would destroy the record of the training it is describing.
    """
    if os.path.exists(layout.round_results_dir):
        shutil.rmtree(layout.round_results_dir)
    for name in EVAL_ARTIFACTS:
        path = os.path.join(layout.report_dir, name)
        if os.path.exists(path):
            os.remove(path)
    os.makedirs(layout.round_results_dir, exist_ok=True)


def load_reference_scaler(scaler_dir):
    """The single global scaler every client trained through.

    Scoring has to happen in the input space the model actually learned, so the
    scaler is loaded rather than refitted. Returns (None, None) if it is missing,
    which means the experiment was never trained.
    """
    path = os.path.join(scaler_dir, "global_scaler.pkl")
    if not os.path.exists(path):
        print(f"❌ Global scaler not found at {path}. Train the experiment first.")
        return None, None

    with open(path, "rb") as f:
        scaler = pickle.load(f)
    # The scaler records the feature names it was fitted on, so it is also the
    # authority on which columns the model expects and in what order.
    return scaler, list(scaler.feature_names_in_)


def find_checkpoints(save_path, stride=None):
    """The saved rounds to evaluate, oldest first, as (round number, path).

    Only every `stride`-th round is scored. Every checkpoint costs a full
    rescoring of all 1000 users, and adjacent rounds differ too little to be
    worth doubling that, so at the configured stride of 2 a 50-round run is
    summarised by 26 points (round 0, the initial model, through round 50).

    Round 0 is always included, because it is the untrained model and the only
    point that shows what the score was before any training happened.
    """
    if not os.path.exists(save_path):
        return []

    if stride is None:
        from config_manager import config
        stride = config.get("evaluation", "checkpoint_stride")

    checkpoints = []
    for filename in os.listdir(save_path):
        if not (filename.startswith("parameters_round_") and filename.endswith(".pkl")):
            continue
        # "parameters_round_12.pkl" -> "12.pkl" -> 12
        round_number = int(filename.split("_")[-1].split(".")[0])
        if round_number % stride == 0:
            checkpoints.append((round_number, os.path.join(save_path, filename)))

    return sorted(checkpoints)


def load_checkpoint_into(model, path):
    """Load one saved round's weights into an existing model.

    The checkpoint stores a plain list of arrays under `global_parameters`, in
    `state_dict` order, because that is the form Flower aggregates. Zipping it
    back against the current keys is what turns it into a state dict again.
    """
    with open(path, "rb") as f:
        weights = pickle.load(f).get("global_parameters")
    model.load_state_dict(
        {key: torch.tensor(weight)
         for key, weight in zip(model.state_dict().keys(), weights)})


@dataclass(frozen=True)
class ModelFacts:
    """What the summary records about the model that produced a run.

    Grouped because they are one description, not five arguments, and because
    they are what lets a reported MB figure be checked rather than trusted: the
    payload is the parameter count times four bytes, and the input shape says
    which model those parameters belong to.
    """

    total_parameters: int
    by_component: dict
    feature_count: int
    window_size: int
    rounds_evaluated: int

    @property
    def dense_payload_mb(self):
        """The uncompressed upload, in MB. 1.7176 at 450,258 parameters."""
        return round(self.total_parameters * 4 / 1024 / 1024, 6)


def count_parameters(model):
    """Total parameter count and the per-component breakdown.

    The parameter count is the payload: every upload figure in the thesis derives
    from it (450,258 x 4 B = 1.7176 MB dense). Recorded per run so the
    communication numbers can be checked against the model that produced them
    rather than taken on trust.
    """
    per_tensor = {name: int(p.numel()) for name, p in model.named_parameters()}

    by_component = {}
    for name, count in per_tensor.items():
        # "encoder.weight_ih_l0" belongs to the encoder; the text quotes the four
        # components, not the individual tensors.
        component = name.split(".")[0]
        by_component[component] = by_component.get(component, 0) + count

    return int(sum(per_tensor.values())), by_component


def expected_upload_count(config, num_supernodes, num_rounds):
    """How many upload records a correct log for this configuration holds."""
    fraction_fit = config.get("federation", "fraction_fit")
    min_fit = config.get("federation", "min_fit_clients")
    # Flower samples this many clients to train each round; the rest sit out.
    clients_per_round = max(int(num_supernodes * fraction_fit), min_fit)
    return num_rounds * clients_per_round


def score_one_round(scorer, model, checkpoint_path, df, all_users, seed=None):
    """Rescore every user with one checkpoint. Returns (metrics row, per-user frame).

    `seed` is accepted and ignored. The user split is now driven by
    `cfg.split_seed`, which is fixed across runs: it belongs to the evaluation
    protocol rather than to the run. Passing the run seed here made every seed
    report on a different set of 35 insiders, and that alone accounted for a
    0.095 gap between two runs of the same configuration.
    """
    load_checkpoint_into(model, checkpoint_path)

    results = scorer.scan(model, df, all_users)
    metrics = scoring.evaluate_scores(
        results, seed=scorer.cfg.split_seed,
        validation_fraction=scorer.cfg.validation_fraction)

    row = {
        "Round": None,  # filled in by the caller, which knows the round number
        "PR-AUC": metrics["pr_auc"],
        "PR-AUC_val": metrics["pr_auc_val"],
        "PR-AUC_all": metrics["pr_auc_all"],
        "Max-F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Accuracy": metrics["accuracy"],
        "Balanced_Accuracy": metrics["balanced_accuracy"],
        "TP": metrics["tp"], "FP": metrics["fp"],
        "TN": metrics["tn"], "FN": metrics["fn"],
        "Optimal-Threshold": metrics["threshold"],
    }
    return row, results


def build_summary(layout, config, summary_df, best_round, comm, facts):
    """The experiment_summary.json contents, assembled in one readable place."""
    # Imported here rather than at module scope for the same reason `config` is:
    # importing config_manager builds the singleton, which must happen after the
    # environment variables that configure a run are set.
    from config_manager import PIPELINE_VERSION

    # Both convergence figures read the validation half, for the same reason
    # checkpoint selection does: the test half must not influence any choice.
    # The untrained round is dropped first; see UNTRAINED_ROUND. Its inflated
    # score would otherwise satisfy the convergence fraction immediately and
    # report that the run converged before it trained.
    rounds, val_scores = trained_rounds(summary_df["Round"].tolist(),
                                        summary_df["PR-AUC_val"].tolist())
    convergence_round = find_convergence_round(rounds, val_scores)
    plateau_round = find_plateau_round(rounds, val_scores)

    best_row = summary_df.loc[summary_df["Round"] == best_round].iloc[0]

    return {
        # Stamped so a comparison table cannot mix this run with results from a
        # pipeline version whose numbers mean something else. See config_manager.
        "pipeline_version": PIPELINE_VERSION,
        "experiment_name": layout.experiment_name,
        "run_id": layout.run_id,
        "seed": layout.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config._active_config,

        "model_parameters_total": facts.total_parameters,
        "model_parameters_by_component": facts.by_component,
        "model_dense_payload_mb": facts.dense_payload_mb,
        "input_matrix_shape": [int(facts.window_size), int(facts.feature_count)],

        "best_metrics": best_row.to_dict(),
        "best_round": int(best_round),
        "rounds_evaluated": facts.rounds_evaluated,
        "selection_rule": "max PR-AUC on the validation user half, round 0 "
                          "excluded; reported metrics come from the test half",
        # Read from the same configuration the two functions read, so the
        # recorded description cannot drift from the rule that was applied.
        "convergence_rule": f"first round reaching "
                            f"{config.get('evaluation', 'convergence_peak_fraction'):.0%}"
                            f" of the peak validation PR-AUC, round 0 excluded "
                            f"(thesis section 4.3)",
        "plateau_rule": f"first round after which validation PR-AUC never "
                        f"improves by more than "
                        f"{config.get('evaluation', 'plateau_tolerance')}, "
                        f"round 0 excluded",
        "convergence_round": convergence_round if convergence_round is not None else "N/A",
        "plateau_round": plateau_round if plateau_round is not None else "N/A",

        "total_communication_mb": comm["total"],
        "total_upload_mb": comm["upload"],
        "total_download_mb": comm["download"],
        "avg_communication_per_round_mb": comm["per_round"],
        "communication_log_matches_run": comm["log_consistent"],

        "ablation_source": (config.qualify(layout.ablation_source)
                            if layout.ablation_source else None),
        "done": True,
    }


def plot_learning_progress(summary_df, run_id, report_dir):
    """Detection quality against round number, for the round-by-round figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['Round'], summary_df['PR-AUC'], marker='o', label='PR-AUC')
    plt.plot(summary_df['Round'], summary_df['Balanced_Accuracy'], marker='^',
             label='Balanced Acc')
    plt.plot(summary_df['Round'], summary_df['Max-F1'], marker='s', label='F1-Score')
    plt.title(f'FL Progress: {run_id}')
    plt.xlabel('Round Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(report_dir, 'learning_progress.png'))
    plt.close()


def run_training(layout):
    """Hand the federation over to flower-simulation.

    A separate process, not a library call: Flower's simulation engine wants to
    own the event loop and the Ray cluster. Training artefacts (checkpoints,
    scalers, communication log) are cleared by `server_app.cleanup()` when it
    starts, so nothing here has to prepare the directories.
    """
    print(f"🚀 Starting federated training for {layout.run_id} "
          f"({layout.num_supernodes} supernodes, seed {layout.seed})...")

    venv_scripts = os.path.dirname(sys.executable)
    flower_bin = os.path.join(venv_scripts, "flower-simulation")
    if not os.path.exists(flower_bin):
        flower_bin = os.path.join(venv_scripts, "flower-simulation.exe")

    subprocess.run([flower_bin, "--app", ".",
                    "--num-supernodes", str(layout.num_supernodes)],
                   check=True)


def run_evaluation(layout, config, task, scoring_cfg, device):
    """Rescore every saved round and write this run's report."""
    print(f"📊 Starting evaluation for '{layout.run_id}'...")
    clean_evaluation_artifacts(layout)

    scaler, features = load_reference_scaler(layout.eval_scaler_dir)
    if scaler is None:
        return False

    checkpoints = find_checkpoints(layout.eval_save_path)
    if not checkpoints:
        print(f"❌ No checkpoints found in {layout.eval_save_path}.")
        return False

    df = pd.read_csv(layout.data_path, low_memory=False)
    # Sorted so the validation/test user split does not depend on how the data
    # happens to be partitioned across clients.
    all_users = sorted(df['user'].unique())

    ref_mean, ref_std = load_error_reference(layout.eval_scaler_dir, len(features))

    # Assembled once and passed as one thing. These seven values are what it
    # takes to score in the input space the model learned, and they used to be
    # threaded through three call layers as separate arguments.
    scorer = scoring.Scorer(
        features=features, scaler=scaler, ref_mean=ref_mean, ref_std=ref_std,
        cfg=scoring_cfg, window_size=task.WINDOW_SIZE, device=device)

    num_rounds = config.get("federation", "num_rounds")
    comm = read_communication_log(
        layout.report_dir,
        expected_upload_count(config, layout.num_supernodes, num_rounds),
        num_rounds)

    model = task.LSTMAutoencoder(input_dim=len(features),
                                 hidden_dim=layout.hidden_dim).to(device)
    total_params, by_component = count_parameters(model)
    print(f"🧮 Model: {total_params:,} parameters "
          f"({total_params * 4 / 1024 / 1024:.4f} MB dense fp32)")

    # The best round is the one with the highest PR-AUC on the validation user
    # half. An earlier version maximised PR-AUC over all 1000 users, which let
    # the labels of the reported half choose the checkpoint.
    rows = []
    best_val_pr_auc, best_round = -1.0, None

    for round_number, checkpoint_path in checkpoints:
        print(f"🔄 Evaluating round {round_number}...")
        row, results = score_one_round(
            scorer, model, checkpoint_path, df, all_users, layout.seed)

        row["Round"] = round_number
        rows.append(row)
        results.to_csv(
            os.path.join(layout.round_results_dir, f"round_{round_number}_results.csv"),
            index=False)

        # Round 0 is still scored and written out, but it cannot be selected;
        # see UNTRAINED_ROUND for why it would otherwise win.
        if (round_number != UNTRAINED_ROUND
                and row["PR-AUC_val"] > best_val_pr_auc):
            best_val_pr_auc, best_round = row["PR-AUC_val"], round_number

    if best_round is None:
        return False

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        os.path.join(layout.report_dir, "federated_rounds_comparison.csv"), index=False)

    facts = ModelFacts(total_parameters=total_params, by_component=by_component,
                       feature_count=len(features), window_size=task.WINDOW_SIZE,
                       rounds_evaluated=len(checkpoints))
    summary = build_summary(layout, config, summary_df, best_round, comm, facts)
    with open(layout.summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    plot_learning_progress(summary_df, layout.run_id, layout.report_dir)
    return True


def run_single_experiment(experiment_name, mode, seed=None):
    """Train one configuration, then score every round it saved."""
    # These must be set before config_manager is first imported: the config is a
    # singleton that reads them at construction time, and flower-simulation is a
    # separate interpreter that inherits them rather than the config object.
    os.environ["EXPERIMENT_NAME"] = experiment_name
    if seed is not None:
        os.environ["SEED"] = str(seed)
        # Set here rather than inside seed_everything, because CPython reads this
        # once at interpreter start. Assigning it mid-process does nothing, which
        # was verified; putting it in the environment does reach the
        # flower-simulation subprocess, which is a fresh interpreter. It fixes
        # set and dict iteration order, the one remaining way ordering could
        # differ between two runs of the same seed.
        os.environ["PYTHONHASHSEED"] = str(seed)

    from config_manager import config
    if seed is not None:
        config.set_seed(seed)
    config.set_experiment(experiment_name)
    set_all_seeds(config.seed)

    # Imported after the config is configured, because both modules read settings
    # at import time.
    import federated_ueba.task as task

    layout = resolve_run_layout(config, experiment_name, mode, config.seed)
    if layout.mode != "train" and already_done(layout.run_id):
        print(f"✅ Experiment '{layout.run_id}' already completed. Skipping.")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_cfg = scoring.ScoringConfig.from_config(config)

    try:
        os.makedirs(layout.report_dir, exist_ok=True)
        if layout.mode in ("full", "train"):
            run_training(layout)
            reject_degraded_training(layout)
        if layout.mode in ("full", "eval"):
            return run_evaluation(layout, config, task, scoring_cfg, device)
        return True
    except Exception as e:
        print(f"❌ Experiment {layout.run_id} failed: {e}")
        traceback.print_exc()
        return False


def reject_degraded_training(layout):
    """Refuse to evaluate a run that trained with clients missing.

    Flower counts a crashed client as an absent one: the round aggregates
    whoever answered, the run reports 50 rounds, and the summary says `done`.
    A run that averaged 5 clients instead of 25 for ten rounds is a different
    experiment from the one the thesis describes, and by the time it reaches a
    results table there is nothing left to distinguish it from a good run.

    Raising here rather than warning, because the sweep's own failure handling
    is the right place to deal with it: a failed pair is left unfinished, the
    runner steps over it, and re-running the sweep picks it up again. A warning
    would be read once, in a log nobody opens, and the number would stay.

    The strategy writes the file (see `ModelSavingMixin._record_failures`); it
    runs in `flower-simulation`, a separate interpreter, so disk is the only
    channel between them.
    """
    path = os.path.join(layout.train_save_path,
                        strategy.ModelSavingMixin.FAILURE_LOG)
    if not os.path.exists(path):
        return

    with open(path) as f:
        record = json.load(f)
    if not record:
        return

    worst = min(entry["results"] for entry in record.values())
    expected = max(entry["results"] + entry["failures"]
                   for entry in record.values())
    raise RuntimeError(
        f"{layout.run_id} lost clients in {len(record)} round(s); the worst "
        f"aggregated {worst} of {expected}. This is not the configuration being "
        f"reported, so the run is rejected rather than scored. Re-run the sweep "
        f"to retry it; see {path} for the affected rounds.")


def read_communication_log(report_dir, expected_uploads, num_rounds):
    """Sum a run's communication log and check that it belongs to that run.

    One file per client partition; see `client_app._log_transfer` for why a
    single shared file loses records under Ray on Windows.

    `per_round` divides by the round count, which is the figure the thesis needs:
    "how much traffic does one round of this configuration cost?" An earlier
    version divided by the number of log records instead, which answers "what did
    the average single transfer weigh?" and is roughly twelve times smaller. The
    field was named per_round either way.

    A record count that does not match the configuration means the log survived
    from a run with a different round or client count. That silently produced
    wrong MB totals for months, so it is reported rather than averaged over.
    """
    totals = {"upload": 0.0, "download": 0.0, "total": 0.0, "per_round": 0.0,
              "log_consistent": None, "upload_records": 0}

    parts = sorted(glob.glob(os.path.join(report_dir, "comm", "client_*.csv")))
    if not parts:
        return totals

    # Each part is `direction,phase,mb`.
    comm_df = pd.concat(
        [pd.read_csv(p, header=None, names=["direction", "phase", "mb"])
         for p in parts],
        ignore_index=True)

    uploads = comm_df[comm_df["direction"] == "upload"]
    downloads = comm_df[comm_df["direction"] == "download"]

    totals["upload"] = float(uploads["mb"].sum())
    totals["download"] = float(downloads["mb"].sum())
    totals["total"] = totals["upload"] + totals["download"]
    totals["per_round"] = totals["total"] / max(num_rounds, 1)
    totals["upload_records"] = int(len(uploads))

    # Flower asks one client for the initial parameters, hence the +1.
    consistent = abs(len(uploads) - (expected_uploads + 1)) <= 1
    totals["log_consistent"] = consistent
    if not consistent:
        print(f"⚠️  comm/ holds {len(uploads)} upload records across {len(parts)} "
              f"client files, but this configuration should produce "
              f"~{expected_uploads + 1}. The reported MB figures do not describe "
              f"this run. Retrain to regenerate them.")
    return totals


def already_done(run_id):
    """Whether a finished result for this run exists that is still usable.

    The version check is the important half. A sweep interrupted by a power cut
    should resume where it stopped, but a sweep started after a pipeline change
    must not treat the old results as finished work: those are precisely the runs
    that have to be redone, and skipping them would leave a mixture of two
    pipelines under one set of names.
    """
    # Imported here rather than at module scope for the same reason `config` is:
    # importing config_manager builds the singleton, which must happen after the
    # environment variables that configure a run are set.
    from config_manager import PIPELINE_VERSION

    path = os.path.join(BASE_REPORT_DIR, run_id, "experiment_summary.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Corrupted summary for '{run_id}'. Re-running.")
        return False

    if not summary.get("done", False):
        return False
    if summary.get("pipeline_version") != PIPELINE_VERSION:
        print(f"♻️  '{run_id}' was produced by an older pipeline version. "
              f"Re-running.")
        return False
    return True


def _names(argument):
    return {name.strip() for name in argument.split(",") if name.strip()}


def is_ablation(config, name):
    """Whether this experiment rescores another one's weights instead of training.

    The override lives under `[tool.fueba.experiments.<name>]` as
    `scoring.ablation_source`, which TOML parses into a nested table. Reading a
    flat "scoring.ablation_source" key instead finds nothing, which is how this
    check was once a silent no-op that only worked because of declaration order.
    """
    return bool(config._experiments.get(name, {})
                .get("scoring", {}).get("ablation_source"))


def select_experiments(config, only, skip):
    """The experiments a sweep should run, honouring --only and --skip.

    Raises rather than returning an empty list. A sweep that selects nothing used
    to print "Running 0 experiment(s)" and then "All experiments finished",
    exiting successfully with no work done. That reads as a completed sweep in a
    log and in an exit code, which is the same silent failure as a sweep that
    spawns experiments and lets them all die.
    """
    known = set(config.experiment_names)
    requested = _names(only)
    skipped = _names(skip)

    for argument, names in (("--only", requested), ("--skip", skipped)):
        unknown = names - known
        if unknown:
            print(f"⚠️  {argument} names no such experiment: "
                  f"{', '.join(sorted(unknown))}")

    # The default sweep is the reported set, not everything defined. `--only`
    # overrides it, so an experiment left out of the list is still one command
    # away; what it is not is 33 minutes on every full re-run.
    reported = [name for name in config.get("sweep", "experiments")
                if name not in NOT_IN_SWEEP]
    unknown_reported = set(reported) - known
    if unknown_reported:
        raise SystemExit(
            f"[tool.fueba.sweep] experiments names configurations that do not "
            f"exist: {', '.join(sorted(unknown_reported))}.\n"
            f"A typo here silently shrinks the sweep, so this stops rather than "
            f"quietly running fewer experiments than the list appears to ask for.")

    chosen = [name for name in config.experiment_names
              if name not in NOT_IN_SWEEP
              and name not in skipped
              and (not requested or name in requested)
              # An explicit --only may reach past the reported set; a plain
              # sweep may not.
              and (requested or not reported or name in reported)]

    if not chosen:
        # The likely cause is naming a smoke test, which is excluded from every
        # sweep by design, so the message says so rather than only listing what
        # was asked for.
        raise SystemExit(
            f"No experiment selected, so there is nothing to sweep.\n"
            f"  --only : {sorted(requested) or 'not given'}\n"
            f"  --skip : {sorted(skipped) or 'not given'}\n"
            f"  always excluded from sweeps: {sorted(NOT_IN_SWEEP)}\n"
            f"Run one of those directly with --experiment <name> instead.")

    print(f"▶️  Running {len(chosen)} experiment(s): {', '.join(chosen)}")
    return chosen


def spawn_experiment(exp_name, mode, seed):
    """Run one experiment in its own process. Returns its exit code.

    A subprocess per experiment, so the config singleton and the cached dataframe
    in `task` start clean for every configuration. Separated from `sweep` so the
    loop's failure handling can be tested without launching anything.
    """
    return subprocess.run([sys.executable, __file__,
                           "--experiment", exp_name,
                           "--mode", mode,
                           "--seed", str(seed)]).returncode


def sweep(experiments, seeds, mode, spawn=spawn_experiment):
    """Run every (experiment, seed) pair. False if it gave up early.

    A failed experiment used to be ignored and the loop simply spawned the next
    one. When the parent process was killed mid-sweep, every remaining subprocess
    died on startup, so the loop burned through 139 experiments in eight seconds
    and reported "All experiments finished" with almost nothing trained, exit code
    zero. Stopping after `MAX_CONSECUTIVE_FAILURES` back to back keeps one flaky
    run from ending a sweep while making that collapse impossible.

    `spawn` is injected so a test can drive the failure logic without starting
    real training; nothing in a run passes it.
    """
    consecutive_failures = 0

    for seed in seeds:
        for exp_name in experiments:
            run_id = f"{exp_name}__seed{seed}"
            if mode != "train" and already_done(run_id):
                print(f"✅ '{run_id}' already completed. Skipping.")
                continue

            print(f"\n{'=' * 60}\n🚀 Spawning: {exp_name} (seed {seed})\n{'=' * 60}")
            started = time.monotonic()
            returncode = spawn(exp_name, mode, seed)
            elapsed = time.monotonic() - started

            if returncode == 0:
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            print(f"⚠️  '{run_id}' exited with code {returncode} after "
                  f"{elapsed:.0f}s.")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n🛑 {consecutive_failures} experiments failed in a row. "
                      f"Stopping so the rest of the sweep is not lost too. Fix "
                      f"the cause and re-run; finished experiments are skipped.")
                return False

    return True


def main():
    """Run one experiment, or sweep every experiment across several seeds.

    A sweep runs each experiment in its own subprocess. That is not caution: the
    config singleton and the cached dataframe in `task` both live for the life of
    a process, so a second experiment in the same one would inherit the first's
    configuration.
    """
    parser = argparse.ArgumentParser(description="F-UEBA experiment runner")
    parser.add_argument("--mode", choices=["full", "train", "eval"], default="full")
    parser.add_argument("--experiment", default="all",
                        help="experiment name from [tool.fueba.experiments], or 'all'")
    parser.add_argument("--seed", type=int, default=None,
                        help="single seed (used when this process runs one experiment)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds to sweep, e.g. --seeds 1,2,3")
    parser.add_argument("--skip", default="",
                        help="comma-separated experiments to leave out of the sweep")
    parser.add_argument("--only", default="",
                        help="comma-separated experiments to run, excluding all "
                             "others; easier to read than a long --skip list")
    args = parser.parse_args()

    from config_manager import config
    # Both --seed and --seeds reach here, and the single-experiment path used to
    # read only --seed. `--experiment X --seeds 1` therefore ran seed 42 without
    # saying so, wrote to a run directory nobody was looking at, and reported a
    # missing scaler for an experiment that had been trained all along. Resolved
    # in one place so the two paths cannot disagree again.
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [config.seed]

    if args.experiment != "all":
        for seed in seeds:
            print(f"\n{'=' * 60}\n🌟 EXECUTING: {args.experiment} (seed {seed})\n"
                  f"{'=' * 60}")
            run_single_experiment(args.experiment, args.mode, seed)
        return

    experiments = select_experiments(config, args.only, args.skip)
    # Ablations rescore another experiment's weights, so that experiment has to
    # have been trained first.
    experiments.sort(key=lambda name: is_ablation(config, name))

    if not sweep(experiments, seeds, args.mode):
        return

    # Named before the comparison, because a pair can go missing without any
    # error reaching the log: a subprocess killed by the OS, or one that dies
    # without raising, leaves no message and the loop simply moves on. One run
    # did exactly that and was noticed only by accident, days later. Anything
    # still outstanding here is either a failure or was never reached.
    outstanding = [f"{name}__seed{seed}"
                   for seed in seeds for name in experiments
                   if not already_done(f"{name}__seed{seed}")]
    if outstanding:
        print("\n" + "=" * 60)
        print(f"⚠️  {len(outstanding)} of {len(seeds) * len(experiments)} pair(s) "
              f"did not finish:")
        for run_id in outstanding:
            print(f"     {run_id}")
        print("Re-run this sweep to retry them; finished pairs are skipped.")

    print("\n" + "=" * 60)
    print("🏁 All experiments finished. Generating comparison...")
    from analysis.compare_experiments import compare_experiments
    compare_experiments()


if __name__ == "__main__":
    main()
