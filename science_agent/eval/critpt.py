"""CritPt data, answer formatting and submission helpers; no agent imports."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from science_agent.eval.answer_extraction import (
    find_boxed_spans,
    last_anchored_final_answer_region,
    last_complete_answer_region,
)

PROBLEM_IDS = {f"Challenge_{n}_main" for n in range(1, 71)}
EVALUATION_URL = "https://artificialanalysis.ai/api/v2/critpt/evaluate"


def problem_id(value: str) -> str:
    match = re.fullmatch(r"(?:CritPt_challenge_(\d+)|Challenge_(\d+)_main)", value)
    if not match:
        raise ValueError(f"Invalid CritPt problem ID: {value!r}")
    result = f"Challenge_{int(match[1] or match[2])}_main"
    if result not in PROBLEM_IDS:
        raise ValueError(f"CritPt problem ID out of range: {value!r}")
    return result


def load_dataset(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [problem_id(str(row.get("id", ""))) for row in rows]
    if len(ids) != 70 or set(ids) != PROBLEM_IDS:
        raise ValueError("CritPt input must contain all 70 unique public challenge IDs")
    for row in rows:
        for key in ("context", "question", "code_template"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row['id']}: missing {key}")
        if row.get("modality", "text-only") != "text-only":
            raise ValueError(f"{row['id']}: this adapter currently supports text-only challenges")
        template = ast.parse(row["code_template"])
        if not any(isinstance(node, ast.FunctionDef) and node.name == "answer" for node in template.body):
            raise ValueError(f"{row['id']}: template has no answer function")
    return rows


def solver_prompt(row: dict) -> str:
    return (
        "Solve this physics research problem. Follow its conventions, units and precision "
        "requirements. If none are specified, retain exact expressions where possible, "
        "otherwise at least 12 significant digits. Use the available tools when helpful. "
        "Put ALL requested final results, including parameter definitions, in one complete "
        "<answer>...</answer> block. Do not omit components of a multi-part answer.\n\n"
        f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\n"
        "The final results will be placed into this answer interface. Its signature, output "
        "types and ordering are part of the question:\n```python\n"
        f"{row['code_template']}\n```"
    )


def extract_final_answer(text: str) -> str:
    """Preserve entire terminal regions and all boxes, never choose only the last box."""
    visible = text.rsplit("</think>", 1)[-1]
    region = last_complete_answer_region(visible)
    if region:
        return region.content
    region = last_anchored_final_answer_region(visible)
    if region:
        return region.content
    boxes = find_boxed_spans(visible)
    return "\n".join(span.raw for span in boxes)


def code_prompt(row: dict, answer: str) -> str:
    return (
        "Populate the supplied final answer into the Python template. This is ONLY a "
        "format conversion: do not solve the problem again, change results, infer missing "
        "results, or import modules beyond those in the template. Preserve the answer "
        "function signature and output order. Return one complete Python code block, "
        "without placeholders.\n\nTemplate:\n```python\n"
        f"{row['code_template']}\n```\n\nFinal answer:\n{answer}"
    )


def extract_code(text: str) -> str:
    visible = text.rsplit("</think>", 1)[-1].strip()
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", visible, re.DOTALL | re.IGNORECASE)
    if len(blocks) > 1:
        raise ValueError("Expected exactly one Python code block")
    return (blocks[0] if blocks else visible).strip()


def validate_code(code: str, template: str) -> None:
    """Structural validation only; never execute generated code on the host."""
    parsed, expected = ast.parse(code), ast.parse(template)
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "answer"]
    expected_function = next(node for node in expected.body if isinstance(node, ast.FunctionDef) and node.name == "answer")
    if len(functions) != 1:
        raise ValueError("Expected exactly one answer function")
    if ast.dump(functions[0].args) != ast.dump(expected_function.args):
        raise ValueError("answer function signature differs from template")
    annotation = lambda value: ast.dump(value) if value is not None else None
    if functions[0].decorator_list or annotation(functions[0].returns) != annotation(expected_function.returns):
        raise ValueError("answer function decoration/return annotation differs from template")
    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant) and node.value is Ellipsis:
            raise ValueError("Unfilled ellipsis placeholder")
        if isinstance(node, ast.Pass):
            raise ValueError("Unfilled pass placeholder")
    allowed = {ast.dump(node) for node in ast.walk(expected) if isinstance(node, (ast.Import, ast.ImportFrom))}
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.dump(node) not in allowed:
            raise ValueError("Import not present in template")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_batch(rows: list[dict], run_dir: Path, model: str, config: dict) -> dict:
    submissions = []
    for row in rows:
        pid = problem_id(row["id"])
        solver = json.loads((run_dir / "rollouts" / f"{pid}.json").read_text())
        conversion = json.loads((run_dir / "conversions" / f"{pid}.json").read_text())
        if solver.get("status") != "completed" or not solver.get("prediction", "").strip():
            raise ValueError(f"{pid}: no final answer; refusing an incomplete batch")
        if conversion.get("status") != "completed":
            raise ValueError(f"{pid}: conversion failed; refusing an incomplete batch")
        if conversion.get("answer_sha256") != fingerprint(solver["prediction"]):
            raise ValueError(f"{pid}: conversion belongs to a different final answer")
        code = conversion["generated_code"]
        validate_code(code, row["code_template"])
        submissions.append({
            "problem_id": pid, "generated_code": code, "model": model,
            "generation_config": config,
        })
    ids = [submission["problem_id"] for submission in submissions]
    if len(ids) != 70 or set(ids) != PROBLEM_IDS:
        raise ValueError("Official grading requires exactly all 70 public challenges")
    return {"submissions": submissions, "batch_metadata": {"num_submissions": 70, "source": "science-agent-sandbox"}}
