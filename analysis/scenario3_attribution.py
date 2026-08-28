"""Where scenario 3's defining event is actually logged.

CERT r4.2's third scenario reads, in the dataset's own words: a disgruntled
system administrator downloads a keylogger, carries it to his supervisor's
machine on a thumb drive, then uses the captured credentials to log in as the
supervisor and send an alarming mass email.

The detection pipeline models users. That makes the last step a problem the
feature set cannot fix: the mass email is written to the *supervisor's* user
row, not the actor's. Every per-user-day aggregate the actor produces that day
is missing the one event that defines the campaign.

This script checks that claim against the ground truth and the raw log rather
than against the scenario prose, because the prose does not say which account
carries the events. Run it from the project root; it reads only.

    python analysis/scenario3_attribution.py
"""

import collections
import csv
import os
import sys

ANSWERS = "dataset/r4.2/answers"
SCENARIO_3 = os.path.join(ANSWERS, "r4.2-3")
EMAIL_LOG = "dataset/r4.2/email.csv"

# Columns in the answers files are positional and undocumented; these are the
# four this script needs. The raw logs carry a header and are read by name.
TYPE, EVENT_ID, STAMP, USER, PC, TARGET = 0, 1, 2, 3, 4, 5


def labelled_insiders():
    """Users r4.2 marks as insiders, in any scenario."""
    with open(os.path.join(ANSWERS, "insiders.csv")) as f:
        return {row["user"] for row in csv.DictReader(f)
                if row["dataset"].strip() == "4.2"}


def foreign_events(path, actor):
    """Ground-truth events on this actor's trail logged under someone else."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) <= TARGET or parts[USER].strip() == actor:
                continue
            recipients = 0
            if parts[TYPE] == "email":
                recipients = len([a for a in parts[TARGET].split(";") if a.strip()])
            rows.append({"type": parts[TYPE], "id": parts[EVENT_ID].strip(),
                         "stamp": parts[STAMP], "user": parts[USER].strip(),
                         "detail": parts[TARGET].strip(), "recipients": recipients})
    return rows


def confirm_in_raw_log(wanted):
    """Re-read the mass emails from email.csv, which is the source of truth.

    The answers files are derived, so agreeing with them proves nothing about
    the log the feature pipeline actually consumes.
    """
    found = {}
    with open(EMAIL_LOG, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["id"] in wanted:
                found[row["id"]] = row
                if len(found) == len(wanted):
                    break
    return found


def main():
    if not os.path.isdir(SCENARIO_3):
        raise SystemExit(f"{SCENARIO_3} not found. Run from the project root.")

    insiders = labelled_insiders()
    kinds = collections.Counter()
    supervisors = collections.Counter()
    mass_emails = {}
    actors = 0

    for name in sorted(os.listdir(SCENARIO_3)):
        actor = name.replace(".csv", "").split("-")[-1]
        actors += 1
        events = foreign_events(os.path.join(SCENARIO_3, name), actor)
        for event in events:
            kinds[event["type"]] += 1
            supervisors[event["user"]] += 1
        emails = [e for e in events if e["type"] == "email"]
        if emails:
            biggest = max(emails, key=lambda e: e["recipients"])
            mass_emails[biggest["id"]] = (actor, biggest["user"], biggest["recipients"])

    print(f"scenario 3 actors: {actors}")
    print(f"events logged under another account, by type: {dict(kinds)}")
    print(f"accounts they land on: {dict(supervisors)}")
    for user in supervisors:
        mark = "IS an insider" if user in insiders else "is NOT an insider"
        print(f"  {user} {mark}")

    confirmed = confirm_in_raw_log(set(mass_emails))
    print(f"\nmass emails re-read from {EMAIL_LOG}: "
          f"{len(confirmed)}/{len(mass_emails)}")
    mismatched = []
    counts = []
    for event_id, (actor, supervisor, _) in sorted(mass_emails.items()):
        row = confirmed.get(event_id)
        if row is None:
            mismatched.append((actor, "missing from raw log"))
            continue
        recipients = len([a for a in row["to"].split(";") if a.strip()])
        counts.append(recipients)
        if row["user"] != supervisor:
            mismatched.append((actor, f"{row['user']} != {supervisor}"))
        print(f"  {actor:9s} sent as {row['user']:9s} to {recipients:3d} recipients")

    if mismatched:
        raise SystemExit(f"attribution disagrees with the answers file: {mismatched}")
    print(f"\nrecipients range {min(counts)}-{max(counts)}")
    print("Every scenario-3 mass email is logged under the supervisor's account, "
          "and no supervisor account is labelled an insider.")


if __name__ == "__main__":
    sys.exit(main())
