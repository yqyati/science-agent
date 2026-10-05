from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_env_file(env_file: str | os.PathLike | None = None, *, override: bool = False) -> Path | None:
    candidate = env_file or os.environ.get("ENV_FILE") or DEFAULT_ENV_FILE
    path = Path(candidate)
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_matching_quotes(value.strip())
        if override or key not in os.environ:
            os.environ[key] = value

    return path


def get_env_value(*names: str, default=None):
    for name in names:
        if not name:
            continue
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def get_env_int(*names: str, default: int) -> int:
    value = get_env_value(*names, default=None)
    if value is None:
        return int(default)
    return int(value)


def get_env_float(*names: str, default: float) -> float:
    value = get_env_value(*names, default=None)
    if value is None:
        return float(default)
    return float(value)


def get_env_bool(*names: str, default: bool = False) -> bool:
    value = get_env_value(*names, default=None)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_env_int_clamped(
    *names: str,
    default: int,
    minimum: int | None = None,
) -> int:
    value = get_env_int(*names, default=default)
    if minimum is not None:
        value = max(minimum, value)
    return value
