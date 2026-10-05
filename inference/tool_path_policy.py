import ast
import os
from dataclasses import dataclass
from typing import Iterable


PATH_POLICY_LEGACY = "legacy"
PATH_POLICY_RELATIVE = "relative"


@dataclass(frozen=True)
class PathPolicyViolation:
    value: str
    reason: str


def normalize_path_policy(value: str | None, *, default: str = PATH_POLICY_RELATIVE) -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"off", "none", "legacy"}:
        return PATH_POLICY_LEGACY
    if normalized in {"relative", "deny", "strict"}:
        return PATH_POLICY_RELATIVE
    return default


def _iter_string_literals(code: str) -> Iterable[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
    return values


def _looks_like_absolute_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped == "/":
        return False
    if stripped.startswith(("http://", "https://", "data:image/")):
        return False
    return stripped.startswith(("/", "~"))


def _has_parent_traversal(value: str) -> bool:
    normalized = value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    return ".." in parts


def find_relative_path_policy_violations(code: str) -> list[PathPolicyViolation]:
    violations: list[PathPolicyViolation] = []
    seen: set[tuple[str, str]] = set()

    for value in _iter_string_literals(code):
        stripped = value.strip()
        reason = ""
        if _looks_like_absolute_path(stripped):
            reason = "absolute paths are not allowed"
        elif _has_parent_traversal(stripped):
            reason = "parent-directory traversal is not allowed"

        if not reason:
            continue
        key = (stripped, reason)
        if key in seen:
            continue
        seen.add(key)
        violations.append(PathPolicyViolation(value=stripped, reason=reason))

    return violations


def format_path_policy_error(tool_label: str, violations: list[PathPolicyViolation]) -> str:
    examples = "; ".join(
        f"{violation.value!r}: {violation.reason}" for violation in violations[:5]
    )
    return (
        f"[{tool_label} Error]: Path policy blocked this code before execution. "
        "Use relative paths inside the sandbox working directory only. "
        "Do not use /mnt, /tmp, /home, /root, or ../ paths. "
        f"Blocked path literal(s): {examples}"
    )


def should_enforce_relative_paths(env_var: str, *, default: str = PATH_POLICY_RELATIVE) -> bool:
    policy = normalize_path_policy(os.getenv(env_var), default=default)
    return policy == PATH_POLICY_RELATIVE

