"""Independent per-item smoke and diversity audit for the rebuilt train set.

Only read-only checks are performed on the generated rows. A successful run is
necessary, not sufficient, for admitting a problem to cold rollout or SFT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parent
BUCKETS = {"target400": 400, "source300": 300, "general300": 300}
REQUIRED = {"task_id", "bucket", "family", "seed_id", "problem",
            "parameters", "answer_contract", "objective", "verifier_id",
            "transformation", "target_alignment", "qa"}


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_verifier(path: Path):
    name = "v1114_rebuild_" + path.parent.name
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    if not callable(getattr(module, "verify", None)):
        raise ValueError(f"{path} has no verify(task,answer)")
    return module.verify


def _timeout(_signum, _frame):
    raise TimeoutError("verifier exceeded per-task timeout")


def checked_verify(fn, task, answer, seconds):
    old = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(task, answer)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def shape(text):
    text = re.sub(r"\d+(?:\.\d+)?", "#", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--timeout", type=float, default=20.)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"schema": "v1114_rebuilt_dataset_qa_v1", "buckets": {},
              "issues": [], "counts": {}, "accepted_count": 0}
    all_ids = set()
    all_questions = set()
    for bucket, target_count in BUCKETS.items():
        folder = args.root / bucket
        problems_path = folder / "problems.jsonl"
        specs_path = folder / "trusted_specs.jsonl"
        verifier_path = folder / "verifier.py"
        if not all(p.is_file() for p in (problems_path, specs_path, verifier_path)):
            report["issues"].append({"bucket": bucket, "error": "missing_artifact"})
            continue
        problems = rows(problems_path)
        specs = rows(specs_path)
        spec_map = {r["task_id"]: r for r in specs}
        verifier = load_verifier(verifier_path)
        families, seeds, shapes = Counter(), Counter(), Counter()
        valid = 0
        by_task = []
        for row in problems:
            task_id = row.get("task_id", "")
            errs = []
            if task_id in all_ids:
                errs.append("duplicate_task_id")
            all_ids.add(task_id)
            question = str(row.get("problem") or "").strip()
            if question in all_questions:
                errs.append("duplicate_exact_question")
            all_questions.add(question)
            if "smoke_answer" in row:
                errs.append("private_smoke_answer_in_public_row")
            if row.get("bucket") != bucket:
                errs.append("wrong_bucket")
            missing = REQUIRED - row.keys()
            if missing:
                errs.append("missing_fields:" + ",".join(sorted(missing)))
            if len(str(row.get("problem") or "")) < 80:
                errs.append("problem_too_short")
            trans = row.get("transformation") or {}
            if not trans.get("why_distinct") or not (trans.get("changed_conditions") or
                                                      trans.get("changed_parameters")):
                errs.append("unexplained_transformation")
            spec = spec_map.get(task_id)
            if spec is None or "smoke_answer" not in spec:
                errs.append("no_trusted_smoke_answer")
            else:
                task = {**row, **spec}
                try:
                    first = checked_verify(verifier, task, spec["smoke_answer"], args.timeout)
                    second = checked_verify(verifier, task, spec["smoke_answer"], args.timeout)
                    if first != second:
                        errs.append("nondeterministic_verifier")
                    if not isinstance(first, dict) or first.get("valid") is not True:
                        errs.append("smoke_answer_rejected")
                    score = first.get("score") if isinstance(first, dict) else None
                    if type(score) not in (int, float) or not math.isfinite(float(score)):
                        errs.append("no_finite_numeric_score")
                    expected = spec.get("expected_smoke_score")
                    if expected is not None and score is not None and not math.isclose(
                            float(score), float(expected), rel_tol=1e-9, abs_tol=1e-9):
                        errs.append("smoke_score_mismatch")
                    alternate_key = {"target400": "sensitivity_answer",
                                     "source300": "better_answer",
                                     "general300": "baseline_answer"}[bucket]
                    alternate = spec.get(alternate_key)
                    if alternate is None:
                        errs.append("no_second_valid_score_witness")
                    else:
                        other = checked_verify(verifier, task, alternate, args.timeout)
                        other_score = other.get("score") if isinstance(other, dict) else None
                        if (not isinstance(other, dict) or other.get("valid") is not True or
                                type(other_score) not in (int, float) or
                                not math.isfinite(float(other_score))):
                            errs.append("second_score_witness_invalid")
                        elif (score is None or math.isclose(float(score),float(other_score),
                                                            rel_tol=1e-12,abs_tol=1e-12)):
                            errs.append("second_score_witness_not_distinct")
                        else:
                            expected_other = spec.get({
                                "target400": "expected_sensitivity_score",
                                "source300": "expected_better_score",
                                "general300": "baseline_score"}[bucket])
                            if (expected_other is not None and not math.isclose(
                                    float(other_score),float(expected_other),
                                    rel_tol=1e-9,abs_tol=1e-9)):
                                errs.append("second_score_witness_score_mismatch")
                            direction = row["objective"]["direction"]
                            better = (float(other_score)>float(score) if
                                      direction=="maximize" else float(other_score)<float(score))
                            if bucket in {"target400", "source300"} and not better:
                                errs.append("proposed_better_answer_not_improvement")
                            if bucket == "general300" and better:
                                errs.append("general_smoke_not_better_than_baseline")
                    bad = spec.get("invalid_answer", None)
                    if bad is not None:
                        reject = checked_verify(verifier, task, bad, args.timeout)
                        if not isinstance(reject, dict) or reject.get("valid") is not False:
                            errs.append("invalid_answer_accepted")
                except Exception as error:
                    errs.append("verifier_exception:" + type(error).__name__)
            families[str(row.get("family"))] += 1
            seeds[str(row.get("seed_id"))] += 1
            shapes[shape(str(row.get("problem") or ""))] += 1
            if not errs:
                valid += 1
            by_task.append({"task_id": task_id, "issues": errs})
        if len(spec_map) != len(specs) or set(spec_map) != {r.get("task_id") for r in problems}:
            report["issues"].append({"bucket": bucket, "error": "trusted_spec_ids_mismatch"})
        if any(n > 12 for n in seeds.values()):
            report["issues"].append({"bucket": bucket,
                "error": "seed_overrepresented", "seeds": {k:v for k,v in seeds.items() if v>12}})
        if any(n > 40 for n in families.values()):
            report["issues"].append({"bucket": bucket,
                "error": "family_overrepresented", "families": {k:v for k,v in families.items() if v>40}})
        if any(n > 12 for n in shapes.values()):
            report["issues"].append({"bucket": bucket,
                "error": "single_prompt_template_overrepresented",
                "max_template_instances": max(shapes.values())})
        report["buckets"][bucket] = {"rows": len(problems), "target_rows": target_count,
            "accepted_static": valid, "families": dict(families), "seeds": dict(seeds),
            "distinct_prompt_shapes": len(shapes), "per_task": by_task}
        report["accepted_count"] += valid
    report["counts"] = {k: v.get("rows", 0) for k,v in report["buckets"].items()}
    report["unique_exact_questions"] = len(all_questions)
    report["audited_artifact_sha256"] = {
        str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for bucket in BUCKETS for path in sorted((args.root/bucket).glob('*'))
        if path.name in {'problems.jsonl','trusted_specs.jsonl'} or path.suffix=='.py'}
    report["full_1000_accepted"] = (report["accepted_count"] == 1000 and
        all(report["counts"].get(k) == n for k,n in BUCKETS.items()) and
        not report["issues"])
    output = args.report or args.root / "dataset_qa_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts":report["counts"], "accepted_static":report["accepted_count"],
                      "full_1000_accepted":report["full_1000_accepted"],
                      "top_level_issues":report["issues"]}, ensure_ascii=False))
    raise SystemExit(0 if report["full_1000_accepted"] else 2)


if __name__ == "__main__":
    main()
