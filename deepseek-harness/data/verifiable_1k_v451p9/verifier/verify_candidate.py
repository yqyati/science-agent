#!/usr/bin/env python3
"""Run the exact verifier for one of the 1000 released training problems.

Only public problem rows enter the verifier call. Private trusted_specs.jsonl is
used by offline QA and must never be mounted into a model-visible workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from validate_dataset import checked_verify, load_verifier

ROOT = Path(__file__).resolve().parent
BUCKETS = ("target400", "source300", "general300")
PUBLIC_FIELDS = ("task_id", "bucket", "family", "problem", "parameters",
                 "answer_contract", "objective", "verifier_id")


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def task_for(identifier: str):
    for bucket in BUCKETS:
        for row in rows(ROOT / "final" / bucket / "problems.jsonl"):
            p9_id = "train_" + hashlib.sha256(row["task_id"].encode()).hexdigest()[:16]
            if identifier in {row["task_id"], p9_id}:
                return {key: row[key] for key in PUBLIC_FIELDS}
    raise KeyError(f"Unknown task ID: {identifier}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True,
                        help="Original task ID or train_<hash> rollout ID")
    answer = parser.add_mutually_exclusive_group()
    answer.add_argument("--answer-file", type=Path)
    answer.add_argument("--answer-json")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    text = (args.answer_file.read_text(encoding="utf-8") if args.answer_file
            else args.answer_json if args.answer_json is not None
            else sys.stdin.read())
    try:
        submitted = json.loads(text)
    except json.JSONDecodeError:
        submitted = text
    try:
        task = task_for(args.task_id)
        verifier = load_verifier(ROOT / "final" / task["bucket"] / "verifier.py")
        verdict = checked_verify(verifier, task, submitted, args.timeout)
    except Exception as error:
        print(json.dumps({"task_id": args.task_id, "valid": False,
                          "score": None, "error": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False))
        return 2
    print(json.dumps({"task_id": task["task_id"], **verdict},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
