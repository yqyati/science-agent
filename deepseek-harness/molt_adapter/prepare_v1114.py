"""Freeze the released 1K public tasks and exact verifiers, without private answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pipeline = output / "verifier"
    pipeline.mkdir()
    hashes = {}
    files = [source / "verify_candidate.py", source / "validate_dataset.py"]
    for bucket in ("target400", "source300", "general300"):
        files.extend(sorted((source / "final" / bucket).glob("*.py")))
        files.append(source / "final" / bucket / "problems.jsonl")
    for path in files:
        relative = path.relative_to(source)
        destination = pipeline / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()

    rows, seeds = [], []
    public = ("task_id", "bucket", "family", "problem", "parameters", "answer_contract", "objective", "verifier_id")
    for bucket in ("target400", "source300", "general300"):
        for line in (pipeline / "final" / bucket / "problems.jsonl").read_text().splitlines():
            raw = json.loads(line)
            task = {key: raw[key] for key in public}
            task["verifier"] = {"kind": "v1114_exact"}
            anchor = raw.get("qa", {}).get("baseline_score")
            if type(anchor) not in (int, float):
                raise ValueError(f"Missing public baseline: {raw['task_id']}")
            item = {"task_id": raw["task_id"], "generation_split": "train", "task": task,
                    "direction": {"minimize": "min", "maximize": "max"}[raw["objective"]["direction"]],
                    "reward_anchor": anchor, "reward_scale": max(1.0, abs(anchor) * 0.1)}
            prompt = (raw["problem"] + "\n\nPublic parameters: " + json.dumps(raw["parameters"], ensure_ascii=False)
                      + "\n\nAnswer contract: " + str(raw["answer_contract"])
                      + "\n\nObjective: " + json.dumps(raw["objective"], ensure_ascii=False))
            rows.append({"prompt": [{"role": "user", "content": prompt}], "dsh_item": item})
            seeds.append({"task_id": raw["task_id"], "bucket": bucket, "family": raw["family"],
                          "seed_id": raw["seed_id"]})
    counts = Counter(row["dsh_item"]["task"]["bucket"] for row in rows)
    if counts != {"target400": 400, "source300": 300, "general300": 300} or len({r["dsh_item"]["task_id"] for r in rows}) != 1000:
        raise ValueError(f"Unexpected task counts: {counts}")
    data = output / "train.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    (output / "source_index.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds))
    provenance = {"source": str(source), "records": len(rows), "buckets": dict(counts),
                  "sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "verifier_sha256": hashes,
                  "private_answers_copied": False, "teacher_answers_in_prompt": False,
                  "reward_baseline": "public qa.baseline_score; label only, not prompt",
                  "target400_included": True, "target400_source": "20 OpenProblemBench-L2 + 29 EinsteinArena seeds",
                  "selection": "All 1000 questions included at user request; not an evaluation-disjoint claim"}
    (output / "train.provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    # Arrow cannot infer one schema for heterogeneous problem parameters or large integers.
    for row in rows:
        row["dsh_item"] = json.dumps(row["dsh_item"], ensure_ascii=False, allow_nan=False)
    molt_data = output / "train.molt.jsonl"
    molt_data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    provenance.update(sha256=hashlib.sha256(molt_data.read_bytes()).hexdigest(), label_encoding="json_string",
                      source_dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    (output / "train.molt.provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "buckets": dict(counts), "dataset": str(data), "verifier": str(pipeline)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="support/training_problems_version1114")
    parser.add_argument("--output", type=Path, required=True, help="New snapshot directory")
    args = parser.parse_args()
    prepare(args.source, args.output)
