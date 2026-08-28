"""What separates the 50 selected features from the 491 candidates.

The jury asked why these fifty (item J5). "Domain knowledge" is an honest answer
but not a checkable one, so this script recovers the rule the selection actually
follows and measures whether the rule is justified.

The extraction pipeline crosses each activity channel against four breakdown
axes: work-hour context, device class, file or URL type, and within-day mean
instead of count. A column that departs from its channel base on at most one of
those axes is a main effect; one that departs on two or more is an interaction
term.

The finding is that the selection takes main effects only. Not as a tendency:
none of the 412 interaction terms is in the set. The rule is worth stating in
the thesis because it is checkable, and the measurement below is worth stating
because it says why the rule is not arbitrary.

    python analysis/feature_selection_rule.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config_manager import config

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Identity, date and label columns. Everything else is a feature candidate.
META = {"starttime", "endtime", "user", "day", "week", "insider",
        "ITAdmin", "O", "C", "E", "A", "N",
        "role", "team", "dept", "b_unit", "f_unit",
        "org_isweekday", "org_isweekend"}

# The file-type and URL-category suffixes the extraction pipeline emits.
TYPES = ("exef", "docf", "txtf", "phof", "compf", "otherf",
         "socnetf", "cloudf", "jobf", "leakf", "hackf")


def axes(column):
    """Which breakdown axes this column departs from its channel base on."""
    found = []
    if "workhour" in column or "afterhour" in column:
        found.append("work-hour context")
    if re.search(r"n-pc\d", column):
        found.append("device class")
    if re.search(r"n-disk\d", column):
        found.append("disk")
    if any(t in column for t in TYPES):
        found.append("file or URL type")
    if "_mean_" in column:
        found.append("within-day mean")
    return found


def summarise(label, frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    sd = frame[columns].std().to_numpy()
    constant = int((sd < 1e-6).sum())
    print(f"{label:34s} n={len(columns):3d}  "
          f"zero cells {100 * (values == 0).mean():5.1f}%  "
          f"constant {constant:3d} ({100 * constant / len(columns):3.0f}%)  "
          f"median sd {np.median(sd):6.2f}")


def main():
    path = config.get("data", "processed_data_path")
    frame = pd.read_csv(path)
    selected = set(config.get("data", "selected_features"))

    candidates = [c for c in frame.columns
                  if c not in META and pd.api.types.is_numeric_dtype(frame[c])]
    main_effects = [c for c in candidates if len(axes(c)) <= 1]
    interactions = [c for c in candidates if len(axes(c)) >= 2]
    deep = [c for c in candidates if len(axes(c)) >= 3]

    print(f"{len(frame.columns)} columns, {len(META & set(frame.columns))} of "
          f"them identity/date/label, {len(candidates)} feature candidates\n")

    print(f"main effects   {len(main_effects):3d} candidates, "
          f"{len([c for c in main_effects if c in selected]):3d} selected")
    print(f"interactions   {len(interactions):3d} candidates, "
          f"{len([c for c in interactions if c in selected]):3d} selected")

    chosen_interactions = [c for c in interactions if c in selected]
    if chosen_interactions:
        raise SystemExit(f"Rule broken by: {chosen_interactions}")
    print("\nThe rule holds without exception: no interaction term is selected.")

    print("\nWhy the rule is not arbitrary:")
    summarise("  main effects", frame, main_effects)
    summarise("  interactions", frame, interactions)
    summarise("  interactions on 3+ axes", frame, deep)
    summarise("  the selected 50", frame,
              [c for c in candidates if c in selected])

    print("\nWhat the rule does not decide, and the thesis calls a limitation:")
    left = sorted(c for c in main_effects if c not in selected)
    print(f"  {len(left)} main effects were available and not taken:")
    for column in left:
        print(f"    {column}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
