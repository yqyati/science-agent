"""Convert a frozen training-only snapshot into Molt rows without teacher answers."""
import argparse
import hashlib
import json
from pathlib import Path

from verifier import KINDS


def convert(snapshot, output):
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name, digest in manifest["data_sha256"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen input changed: {name}")
    for name, digest in manifest["verifier_code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen verifier changed: {name}")
    tasks = {row["task_id"]: row for row in map(json.loads, (snapshot / "tasks.jsonl").read_text().splitlines())}
    rows = []
    seen = set()
    for row in map(json.loads, (snapshot / "inputs.jsonl").read_text().splitlines()):
        if row["generation_split"] != "train":
            raise ValueError("Evaluation tasks cannot enter RL training")
        task = tasks[row["task_id"]]
        if row["verifier_kind"] != task["verifier"]["kind"]:
            raise ValueError("Input/task verifier mismatch")
        if task["verifier"]["kind"] not in KINDS:
            raise ValueError(f"Unsupported verifier: {task['verifier']['kind']}")
        if row["task_id"] in seen:
            raise ValueError("Duplicate task ID")
        seen.add(row["task_id"])
        prompt = row["problem"] + "\n\nAnswer contract: " + row["answer_contract"]
        if row["verifier_kind"] == "ac_official":
            name = "stable_move_spec.json" if row["family"] == "stable_ac" else "move_spec.json"
            prompt += "\n\nGeneral move specification:\n" + (snapshot / name).read_text()
        baseline = task.get("baseline", {})
        anchor = baseline.get("value")
        direction = baseline.get("direction", "min")
        item = {"task_id": row["task_id"], "generation_split": "train", "task": task,
                "direction": direction, "reward_anchor": anchor,
                "reward_scale": max(1.0, abs(anchor) * 0.1) if type(anchor) in (int, float) else 10.0,
                "verification_only": row["family"] == "difference_set"}
        rows.append({"prompt": [{"role": "user", "content": prompt}], "dsh_item": item})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    provenance = {"records": len(rows), "output": str(output.resolve()),
                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "source_snapshot": str(snapshot.resolve()), "source_manifest": manifest,
                  "teacher_answers_in_prompt": False}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return {"records": len(rows), "output": str(output), "verifier_root": manifest["pipeline"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.snapshot, args.output), ensure_ascii=False))
