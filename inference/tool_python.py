import json
import re
import base64
import hashlib
import io
import tarfile
from typing import Dict, List, Optional, Union, Any
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
import requests
from requests.exceptions import RequestException, Timeout
import os
from pathlib import Path
import random
import time
from runtime_env import get_env_bool, get_env_int
from tool_path_policy import (
    PATH_POLICY_RELATIVE,
    find_relative_path_policy_violations,
    format_path_policy_error,
    should_enforce_relative_paths,
)
from sandbox_gateway import AgentSandboxRun, SandboxGatewayError, shared_sandbox_run
from tool_url_guard import blocked_python_message, is_text_blocked, is_value_blocked

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover - inference can be used outside science-agent.
    def emit_trace(event: str, *, layer: str, **fields: Any) -> None:
        del event, layer, fields

CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
SANDBOX_TMP_PATH_RE = re.compile(
    r"(^|[\s\"'=:\[(,{])"
    r"(/tmp/tmp[A-Za-z0-9_.-]+(?:/[^\s\"'<>]*)?)"
)
SANDBOX_TMP_TRAILING_PUNCTUATION = ".,;:)]}"


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


def normalize_sandbox_fusion_endpoint(endpoint: Optional[str]) -> str:
    if endpoint is None:
        return ""
    return endpoint.strip().rstrip("/")


def parse_sandbox_fusion_endpoints(raw_endpoints: Optional[str]) -> List[str]:
    if not raw_endpoints:
        return []
    return [
        endpoint
        for endpoint in (
            normalize_sandbox_fusion_endpoint(item)
            for item in raw_endpoints.split(",")
        )
        if endpoint
    ]


def load_sandbox_fusion_endpoints_from_env() -> List[str]:
    endpoints: List[str] = []
    seen: set[str] = set()
    for raw in (
        os.environ.get("SANDBOX_FUSION_ENDPOINTS"),
        os.environ.get("SANDBOX_FUSION_ENDPOINT"),
    ):
        for endpoint in parse_sandbox_fusion_endpoints(raw):
            if endpoint not in seen:
                endpoints.append(endpoint)
                seen.add(endpoint)
    return endpoints


def missing_sandbox_fusion_endpoint_message() -> str:
    return (
        "[Python Interpreter Error]: Missing SANDBOX_FUSION_ENDPOINTS or "
        "SANDBOX_FUSION_ENDPOINT. Set it to one or more SandboxFusion base URLs "
        "such as http://127.0.0.1:8080."
    )


def _safe_relative_dir_from_env(name: str, default: str) -> str:
    value = (os.getenv(name) or default).strip().strip("/")
    rel = os.path.normpath(value or default)
    if not rel or rel == "." or os.path.isabs(rel) or rel.startswith(".."):
        return default
    return rel


# Array of sandbox fusion endpoints
SANDBOX_FUSION_ENDPOINTS = load_sandbox_fusion_endpoints_from_env()


PYTHON_TOOL_TIMEOUT_SEC = max(
    1,
    get_env_int("PYTHON_TOOL_TIMEOUT_SEC", default=50),
)
PYTHON_TOOL_MAX_RETRIES = max(
    1,
    get_env_int("PYTHON_TOOL_MAX_RETRIES", default=5),
)
PYTHON_TOOL_TOTAL_TIMEOUT_SEC = max(
    0,
    get_env_int(
        "PYTHON_TOOL_TOTAL_TIMEOUT_SEC",
        default=get_env_int(
            "SCIENCE_AGENT_POOL_PROXY_TIMEOUT",
            "SCIENCE_AGENT_GATEWAY_TIMEOUT",
            default=0,
        ),
    ),
)
PYTHON_SESSION_BACKEND = (
    os.getenv("PYTHON_SESSION_BACKEND")
    or os.getenv("PYTHON_JUPYTER_BACKEND")
    or "sandbox_checkpoint"
).strip().lower()
SANDBOX_GATEWAY_BACKEND = "gateway_sandbox"
PYTHON_CHECKPOINT_MAX_BYTES_MB = max(
    1,
    get_env_int("PYTHON_CHECKPOINT_MAX_BYTES_MB", default=256),
)
PYTHON_CHECKPOINT_FILE_MAX_BYTES_MB = max(
    1,
    get_env_int("PYTHON_CHECKPOINT_FILE_MAX_BYTES_MB", default=256),
)
PYTHON_CHECKPOINT_MAX_OBJECT_BYTES_MB = max(
    1,
    get_env_int("PYTHON_CHECKPOINT_MAX_OBJECT_BYTES_MB", default=512),
)
PYTHON_CHECKPOINT_MAX_TAIL_CELLS = max(
    0,
    get_env_int("PYTHON_CHECKPOINT_MAX_TAIL_CELLS", default=20),
)
PYTHON_CHECKPOINT_SAVE_RETRIES = max(
    1,
    get_env_int("PYTHON_CHECKPOINT_SAVE_RETRIES", default=2),
)
PYTHON_CHECKPOINT_FILE_GUARD = get_env_bool(
    "PYTHON_CHECKPOINT_FILE_GUARD",
    default=True,
)
PYTHON_CHECKPOINT_PDF_SIGNATURE_GUARD = get_env_bool(
    "PYTHON_CHECKPOINT_PDF_SIGNATURE_GUARD",
    default=True,
)
PYTHON_CHECKPOINT_LEGACY_PATH_COMPAT = get_env_bool(
    "PYTHON_CHECKPOINT_LEGACY_PATH_COMPAT",
    default=True,
)
PYTHON_CHECKPOINT_PROTECTED_SUFFIXES = tuple(
    suffix if suffix.startswith(".") else f".{suffix}"
    for suffix in (
        item.strip().lower()
        for item in os.getenv(
            "PYTHON_CHECKPOINT_PROTECTED_SUFFIXES",
            ".pdf,.png,.jpg,.jpeg,.webp,.tif,.tiff",
        ).split(",")
    )
    if suffix
)
PYTHON_CHECKPOINT_STATE_FILE = ".science_agent_state/session.pkl"
PYTHON_CHECKPOINT_FILES_TAR = ".science_agent_state/files.tar.gz"
PYTHON_CHECKPOINT_META_FILE = ".science_agent_state/checkpoint_meta.json"
PYTHON_CHECKPOINT_REQUIRED_FILES = {
    PYTHON_CHECKPOINT_STATE_FILE,
    PYTHON_CHECKPOINT_FILES_TAR,
    PYTHON_CHECKPOINT_META_FILE,
}
PYTHON_CHECKPOINT_FETCH_FILES = [
    PYTHON_CHECKPOINT_STATE_FILE,
    PYTHON_CHECKPOINT_FILES_TAR,
    PYTHON_CHECKPOINT_META_FILE,
]
PYTHON_FILE_DIAGNOSTIC_TOP_N = max(
    1,
    get_env_int("PYTHON_FILE_DIAGNOSTIC_TOP_N", default=20),
)
PYTHON_TOOL_RETURN_MAX_CHARS = max(
    0,
    get_env_int("PYTHON_TOOL_RETURN_MAX_CHARS", default=50000),
)
PYTHON_TOOL_RETURN_TAIL_CHARS = max(
    0,
    get_env_int("PYTHON_TOOL_RETURN_TAIL_CHARS", default=10000),
)
PYTHON_TOOL_NO_OUTPUT_MESSAGE = (
    "Execution succeeded with no stdout/stderr/display/error."
)
PYTHON_JUPYTER_KERNEL = "python3"
TOOL_OUTPUTS_DIR = _safe_relative_dir_from_env("TOOL_OUTPUTS_DIR", "tool_outputs")
TOOL_OUTPUTS_SYNC_LOCAL = get_env_bool("TOOL_OUTPUTS_SYNC_LOCAL", default=True)
TOOL_OUTPUTS_SYNC_MODE = (os.getenv("TOOL_OUTPUTS_SYNC_MODE") or "images").strip().lower()
if TOOL_OUTPUTS_SYNC_MODE not in {"all", "images", "none", "off", "0", "false"}:
    TOOL_OUTPUTS_SYNC_MODE = "images"
TOOL_OUTPUTS_SYNC_SUFFIXES = tuple(
    suffix if suffix.startswith(".") else f".{suffix}"
    for suffix in (
        item.strip().lower()
        for item in os.getenv(
            "TOOL_OUTPUTS_SYNC_SUFFIXES",
            ".png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff,.svg",
        ).split(",")
    )
    if suffix
)
TOOL_OUTPUTS_RETURN_MAX_FILES = max(
    0,
    get_env_int("TOOL_OUTPUTS_RETURN_MAX_FILES", default=20),
)
PYTHON_SETUP_CELL = (
    "import os\n"
    f"os.makedirs({TOOL_OUTPUTS_DIR!r}, exist_ok=True)"
)
PYTHON_INTERPRETER_DESCRIPTION_LEGACY = (
    "Execute Python code in a stateful Jupyter-style notebook environment. "
    "Use when Python is necessary for scientific reasoning, exact computation, "
    "data analysis, plotting, or checking intermediate results."
)
PYTHON_INTERPRETER_DESCRIPTION_RELATIVE = (
    PYTHON_INTERPRETER_DESCRIPTION_LEGACY
    + " Use paths that are valid inside the sandbox. Save downloaded or "
    f"generated files under `{TOOL_OUTPUTS_DIR}/` when they need to persist, "
    "be inspected, or be reused later in the notebook."
)


def _checkpoint_backend_enabled() -> bool:
    return PYTHON_SESSION_BACKEND in {
        "sandbox_checkpoint",
        "checkpoint",
        "stateful",
        "stateful_checkpoint",
    }


def _checkpoint_max_state_bytes() -> int:
    return PYTHON_CHECKPOINT_MAX_BYTES_MB * 1024 * 1024


def _checkpoint_max_files_bytes() -> int:
    return PYTHON_CHECKPOINT_FILE_MAX_BYTES_MB * 1024 * 1024


def _checkpoint_max_object_bytes() -> int:
    return PYTHON_CHECKPOINT_MAX_OBJECT_BYTES_MB * 1024 * 1024


def _tool_outputs_sync_enabled() -> bool:
    return TOOL_OUTPUTS_SYNC_LOCAL and TOOL_OUTPUTS_SYNC_MODE not in {"none", "off", "0", "false"}


def _should_sync_local_tool_artifact(path: str, *, source: str = "") -> bool:
    if not _tool_outputs_sync_enabled():
        return False
    if TOOL_OUTPUTS_SYNC_MODE == "all":
        return True
    if source == "vision_display":
        return True
    suffix = Path(str(path or "")).suffix.lower()
    return bool(suffix and suffix in TOOL_OUTPUTS_SYNC_SUFFIXES)


def _effective_python_tool_max_attempts(timeout: int) -> int:
    if PYTHON_TOOL_TOTAL_TIMEOUT_SEC <= 0 or timeout <= 0:
        return PYTHON_TOOL_MAX_RETRIES
    budget_attempts = max(1, PYTHON_TOOL_TOTAL_TIMEOUT_SEC // max(1, timeout))
    return max(1, min(PYTHON_TOOL_MAX_RETRIES, budget_attempts))


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _extract_file_error_diagnostics(output: str) -> dict[str, Any]:
    text = str(output or "")
    markers = [
        marker
        for marker in (
            "EmptyFileError",
            "FileNotFoundError",
            "PDFSyntaxError",
            "FileDataError",
            "cannot open empty file",
            "Cannot read an empty file",
            "No /Root object",
            "No such file or directory",
        )
        if marker.lower() in text.lower()
    ]
    if not markers:
        return {}

    paths: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"filename=(['\"])(?P<path>.+?)\1",
        r"No such file or directory:\s*(['\"])(?P<path>.+?)\1",
        r"no such file:\s*(['\"])(?P<path>.+?)\1",
        r"Cannot open empty file:\s*filename=(['\"])(?P<path>.+?)\1",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            path = str(match.group("path") or "").strip()
            if path and path not in seen:
                seen.add(path)
                paths.append(path)

    lowered_markers = {marker.lower() for marker in markers}
    lowered_paths = [path.lower() for path in paths]
    checkpoint_infra_suspect = ".science_agent_state" in text.lower() or any(
        ".science_agent_state" in path for path in lowered_paths
    )
    if checkpoint_infra_suspect:
        diagnostic_category = "checkpoint_infra_suspect"
    elif any(
        marker in lowered_markers
        for marker in (
            "emptyfileerror",
            "pdfsyntaxerror",
            "filedataerror",
            "cannot open empty file",
            "cannot read an empty file",
            "no /root object",
        )
    ):
        diagnostic_category = "empty_or_corrupt_file"
    elif any(
        marker in lowered_markers
        for marker in (
            "filenotfounderror",
            "no such file or directory",
        )
    ):
        diagnostic_category = "missing_file"
    else:
        diagnostic_category = "unknown_file_error"

    return {
        "markers": markers,
        "paths": paths[:PYTHON_FILE_DIAGNOSTIC_TOP_N],
        "output_chars": len(text),
        "diagnostic_category": diagnostic_category,
        "checkpoint_infra_suspect": checkpoint_infra_suspect,
    }


def _checkpoint_rel_for_virtual_abs_path(path: Any) -> str:
    text = str(path or "").strip()
    if not text.startswith("/"):
        return ""
    norm = os.path.normpath(text)
    if not norm.startswith("/"):
        return ""
    return os.path.normpath(os.path.join(".sandbox_abs", norm.lstrip("/")))


def _checkpoint_manifest_paths(meta: Optional[dict[str, Any]]) -> set[str]:
    if not isinstance(meta, dict):
        return set()
    paths: set[str] = set()
    manifest = meta.get("file_manifest")
    if isinstance(manifest, dict):
        paths.update(str(path) for path in manifest if path)
    for item in meta.get("file_manifest_top") or []:
        if isinstance(item, dict) and item.get("path"):
            paths.add(str(item["path"]))
    return paths


def _refine_file_error_diagnostics(
    diagnostic: dict[str, Any],
    checkpoint_meta: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not diagnostic:
        return {}
    refined = dict(diagnostic)
    base_category = str(refined.get("diagnostic_category") or "")
    if base_category == "checkpoint_infra_suspect":
        return refined

    manifest_paths = _checkpoint_manifest_paths(checkpoint_meta)
    new_category = base_category
    if base_category == "empty_or_corrupt_file":
        new_category = "file_content_invalid_or_corrupt"
    elif base_category == "missing_file":
        matches = []
        for path in refined.get("paths") or []:
            checkpoint_rel = _checkpoint_rel_for_virtual_abs_path(path)
            if checkpoint_rel and checkpoint_rel in manifest_paths:
                matches.append({"path": str(path), "checkpoint_rel": checkpoint_rel})
        if matches:
            new_category = "path_redirect_bypass_suspect"
            refined["checkpoint_manifest_matches"] = matches[:PYTHON_FILE_DIAGNOSTIC_TOP_N]
        else:
            new_category = "missing_file_not_in_checkpoint"
        refined["checkpoint_manifest_paths_checked"] = len(manifest_paths)

    if new_category != base_category:
        refined["base_diagnostic_category"] = base_category
        refined["diagnostic_category"] = new_category
    return refined


def _truncate_python_tool_return(output: str, *, code_hash: str = "", checkpoint_ok: Optional[bool] = None) -> str:
    original_chars = len(output)
    max_chars = PYTHON_TOOL_RETURN_MAX_CHARS
    if max_chars <= 0 or original_chars <= max_chars:
        emit_trace(
            "python_tool_return_prepared",
            layer="tool_python",
            code_hash=code_hash,
            checkpoint_ok=checkpoint_ok,
            output_truncated=False,
            output_chars_before_truncate=original_chars,
            output_chars_after_truncate=original_chars,
            return_max_chars=max_chars,
        )
        return output

    notice = (
        "\n\n[Python tool output truncated: "
        f"original_chars={original_chars}, output_limit_chars={max_chars} (including this notice). "
        "The output exceeded the length limit; the middle was omitted and the beginning/end retained. "
        "This is an incomplete output, not the complete evidence. Files in the sandbox are unchanged. "
        "For a saved page, read smaller line ranges from its path and print line numbers with context. "
        f"For computed results, save them to {TOOL_OUTPUTS_DIR}/ and read smaller slices.]\n\n"
    )
    budget = max_chars - len(notice)
    if budget <= 0:
        truncated = notice[:max_chars]
    else:
        tail_chars = min(PYTHON_TOOL_RETURN_TAIL_CHARS, max(0, budget // 2))
        head_chars = max(0, budget - tail_chars)
        tail = output[-tail_chars:] if tail_chars else ""
        truncated = output[:head_chars] + notice + tail

    emit_trace(
        "python_tool_return_prepared",
        layer="tool_python",
        code_hash=code_hash,
        checkpoint_ok=checkpoint_ok,
        output_truncated=True,
        output_chars_before_truncate=original_chars,
        output_chars_after_truncate=len(truncated),
        return_max_chars=max_chars,
        return_tail_chars=PYTHON_TOOL_RETURN_TAIL_CHARS,
    )
    return truncated


def _checkpoint_update_failed_category(
    reject_reasons: list[str],
    errors: list[str],
    *,
    file_restore_failed: bool = False,
    state_restore_failed: bool = False,
) -> str:
    reasons = set(str(reason) for reason in reject_reasons)
    error_text = "\n".join(str(error) for error in errors).lower()
    if file_restore_failed:
        return "file_restore_failed"
    if state_restore_failed and not reasons.intersection({"missing_checkpoint_files", "files_save_failed", "manifest_save_failed"}):
        return "state_restore_failed"
    if "file_guard_failed" in reasons:
        return "file_guard_failed"
    if "too large" in error_text or "checkpoint too large" in error_text:
        return "checkpoint_size_limit"
    if reasons.intersection({"file_save_verify_errors", "checkpoint_payload_verify_failed"}):
        return "checkpoint_payload_verify_failed"
    if reasons.intersection({"missing_checkpoint_files", "files_save_failed", "manifest_save_failed", "save_errors"}):
        return "checkpoint_save_failed"
    if "file_save_verify_failed" in reasons:
        return "checkpoint_save_verify_failed"
    if state_restore_failed:
        return "state_restore_failed"
    if reasons:
        return "other_checkpoint_update_failed"
    return "unknown_checkpoint_update_failed"


def _checkpoint_file_guard_warning(meta: dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        return ""
    restored_count = len(meta.get("zero_binary_restored_files") or []) + len(
        meta.get("invalid_pdf_restored_files") or []
    )
    dropped_count = len(meta.get("zero_binary_dropped_files") or []) + len(
        meta.get("invalid_pdf_dropped_files") or []
    )
    if restored_count <= 0 and dropped_count <= 0:
        return ""
    parts = []
    if restored_count:
        parts.append(f"{restored_count} previous non-empty file(s) preserved")
    if dropped_count:
        parts.append(f"{dropped_count} empty or invalid file(s) omitted")
    detail = "; ".join(parts)
    return (
        "[Python Interpreter Warning]: Checkpoint file guard ran: "
        f"{detail}. Re-download or regenerate omitted files before reading them."
    )


def _merge_fetch_files(*groups: Optional[list[str]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            if item not in seen:
                output.append(item)
                seen.add(item)
    return output


def _drop_checkpoint_response_files(response_data: Any) -> None:
    if isinstance(response_data, dict):
        response_data.pop("files", None)


def _safe_checkpoint_rel_path(path: Any) -> str:
    rel = os.path.normpath(str(path or ""))
    if not rel or rel == "." or os.path.isabs(rel) or rel.startswith(".."):
        raise ValueError(f"unsafe checkpoint path: {path}")
    return rel


def _verify_checkpoint_tar_payload(files: dict[str, str], meta: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "file_manifest_top": [],
        "largest_file_bytes": 0,
        "verified_files_count": 0,
    }
    manifest = meta.get("file_manifest")
    if not isinstance(manifest, dict):
        return ["file_manifest is not a dict"], summary

    tar_b64 = files.get(PYTHON_CHECKPOINT_FILES_TAR)
    if not isinstance(tar_b64, str) or not tar_b64:
        return [f"missing {PYTHON_CHECKPOINT_FILES_TAR}"], summary

    expected: dict[str, dict[str, Any]] = {}
    for name, info in manifest.items():
        try:
            rel = _safe_checkpoint_rel_path(name)
        except Exception as exc:
            errors.append(str(exc))
            continue
        expected[rel] = info if isinstance(info, dict) else {}

    try:
        tar_bytes = base64.b64decode(tar_b64)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            seen: set[str] = set()
            for member in tf.getmembers():
                if member.isdir():
                    continue
                try:
                    rel = _safe_checkpoint_rel_path(member.name)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if rel != member.name:
                    errors.append(f"non-normal checkpoint path: {member.name}")
                    continue
                if not member.isfile():
                    errors.append(f"non-regular checkpoint entry: {rel}")
                    continue
                seen.add(rel)
                info = expected.get(rel)
                if info is None:
                    errors.append(f"unexpected file in checkpoint tar: {rel}")
                    continue
                extracted = tf.extractfile(member)
                data = extracted.read() if extracted is not None else b""
                actual_size = len(data)
                expected_size = int(info.get("size", -1))
                if expected_size >= 0 and actual_size != expected_size:
                    errors.append(f"size mismatch: {rel} {actual_size} != {expected_size}")
                expected_sha = str(info.get("sha256") or "")
                if expected_sha:
                    actual_sha = hashlib.sha256(data).hexdigest()
                    if actual_sha != expected_sha:
                        errors.append(f"sha256 mismatch: {rel} {actual_sha} != {expected_sha}")
            for rel in sorted(set(expected) - seen):
                errors.append(f"missing file in checkpoint tar: {rel}")
    except Exception as exc:
        errors.append(f"checkpoint tar verify: {exc!r}")

    top_files = sorted(
        (
            {"path": rel, "size": int((info or {}).get("size", 0) or 0)}
            for rel, info in expected.items()
        ),
        key=lambda item: item["size"],
        reverse=True,
    )[:5]
    summary["file_manifest_top"] = top_files
    summary["largest_file_bytes"] = top_files[0]["size"] if top_files else 0
    summary["verified_files_count"] = len(expected)
    return errors, summary


def _build_checkpoint_restore_cell(extra_setup: str = "", *, legacy_path_compat: bool = False) -> str:
    setup = (extra_setup or "").strip()
    if setup:
        setup = setup + "\n"
    return f"""
{setup}import os as __sa_os
import json as __sa_json
import tarfile as __sa_tarfile
import hashlib as __sa_hashlib
import importlib as __sa_importlib
import re as __sa_re
__sa_state_dir = ".science_agent_state"
__sa_state_file = "{PYTHON_CHECKPOINT_STATE_FILE}"
__sa_files_tar = "{PYTHON_CHECKPOINT_FILES_TAR}"
__sa_meta_file = "{PYTHON_CHECKPOINT_META_FILE}"
__sa_legacy_path_compat = {bool(legacy_path_compat)}
__sa_restore_meta = {{
    "checkpoint_present": False,
    "state_restored": False,
    "files_restored": False,
    "manifest_verified": False,
    "file_restore_failed": False,
    "state_restore_failed": False,
    "restore_ok": True,
    "errors": [],
    "meta_errors": [],
    "file_errors": [],
    "state_errors": [],
    "module_errors": [],
    "legacy_path_compat": __sa_legacy_path_compat,
    "legacy_path_compat_normalized_globals": 0,
    "legacy_path_compat_errors": [],
}}
__sa_os.makedirs({TOOL_OUTPUTS_DIR!r}, exist_ok=True)
__sa_os.makedirs(__sa_state_dir, exist_ok=True)
try:
    __sa_restore_meta["checkpoint_present"] = (
        __sa_os.path.exists(__sa_state_file)
        or __sa_os.path.exists(__sa_files_tar)
        or __sa_os.path.exists(__sa_meta_file)
    )
except Exception:
    pass
try:
    if __sa_os.path.exists(__sa_meta_file):
        with open(__sa_meta_file, "r", encoding="utf-8") as __sa_handle:
            __sa_previous_meta = __sa_json.load(__sa_handle)
    else:
        __sa_previous_meta = {{}}
except Exception as __sa_exc:
    __sa_previous_meta = {{}}
    __sa_err = "meta_restore: " + repr(__sa_exc)
    __sa_restore_meta["meta_errors"].append(__sa_err)
    __sa_restore_meta["errors"].append(__sa_err)
try:
    if __sa_os.path.exists(__sa_files_tar):
        with __sa_tarfile.open(__sa_files_tar, "r:gz") as __sa_tf:
            __sa_root = __sa_os.path.abspath(".")
            for __sa_member in __sa_tf.getmembers():
                __sa_target = __sa_os.path.abspath(__sa_os.path.join(".", __sa_member.name))
                if not (__sa_target == __sa_root or __sa_target.startswith(__sa_root + __sa_os.sep)):
                    raise RuntimeError(f"unsafe checkpoint path: {{__sa_member.name}}")
            __sa_tf.extractall(".")
        __sa_restore_meta["files_restored"] = True
        __sa_manifest = __sa_previous_meta.get("file_manifest") or {{}}
        if isinstance(__sa_manifest, dict):
            __sa_manifest_errors = []
            for __sa_rel, __sa_info in __sa_manifest.items():
                try:
                    __sa_rel_norm = __sa_os.path.normpath(str(__sa_rel))
                    if __sa_rel_norm.startswith("..") or __sa_os.path.isabs(__sa_rel_norm):
                        raise RuntimeError(f"unsafe manifest path: {{__sa_rel}}")
                    __sa_path = __sa_os.path.abspath(__sa_os.path.join(".", __sa_rel_norm))
                    if not (__sa_path == __sa_root or __sa_path.startswith(__sa_root + __sa_os.sep)):
                        raise RuntimeError(f"unsafe manifest path: {{__sa_rel}}")
                    if not __sa_os.path.isfile(__sa_path):
                        __sa_manifest_errors.append(f"missing file: {{__sa_rel_norm}}")
                        continue
                    __sa_expected_size = int((__sa_info or {{}}).get("size", -1))
                    __sa_actual_size = __sa_os.path.getsize(__sa_path)
                    if __sa_expected_size >= 0 and __sa_actual_size != __sa_expected_size:
                        __sa_manifest_errors.append(
                            f"size mismatch: {{__sa_rel_norm}} {{__sa_actual_size}} != {{__sa_expected_size}}"
                        )
                        continue
                    __sa_expected_sha = str((__sa_info or {{}}).get("sha256") or "")
                    if __sa_expected_sha:
                        __sa_hasher = __sa_hashlib.sha256()
                        with open(__sa_path, "rb") as __sa_file_handle:
                            for __sa_chunk in iter(lambda: __sa_file_handle.read(1024 * 1024), b""):
                                __sa_hasher.update(__sa_chunk)
                        __sa_actual_sha = __sa_hasher.hexdigest()
                        if __sa_actual_sha != __sa_expected_sha:
                            __sa_manifest_errors.append(
                                f"sha256 mismatch: {{__sa_rel_norm}} {{__sa_actual_sha}} != {{__sa_expected_sha}}"
                            )
                except Exception as __sa_exc:
                    __sa_manifest_errors.append(f"manifest check {{__sa_rel}}: {{__sa_exc!r}}")
            if __sa_manifest_errors:
                __sa_restore_meta["file_errors"].extend(__sa_manifest_errors)
                __sa_restore_meta["errors"].extend(__sa_manifest_errors)
            else:
                __sa_restore_meta["manifest_verified"] = True
        else:
            __sa_err = "file_manifest is not a dict"
            __sa_restore_meta["file_errors"].append(__sa_err)
            __sa_restore_meta["errors"].append(__sa_err)
except Exception as __sa_exc:
    __sa_err = "files_restore: " + repr(__sa_exc)
    __sa_restore_meta["file_errors"].append(__sa_err)
    __sa_restore_meta["errors"].append(__sa_err)
try:
    __sa_module_aliases = __sa_previous_meta.get("module_aliases") or {{}}
    if isinstance(__sa_module_aliases, dict):
        for __sa_alias, __sa_module_name in __sa_module_aliases.items():
            try:
                if (
                    isinstance(__sa_alias, str)
                    and isinstance(__sa_module_name, str)
                    and not __sa_alias.startswith("__sf_")
                ):
                    globals()[__sa_alias] = __sa_importlib.import_module(__sa_module_name)
            except Exception as __sa_exc:
                __sa_err = f"module_alias {{__sa_alias}}={{__sa_module_name}}: {{__sa_exc!r}}"
                __sa_restore_meta["module_errors"].append(__sa_err)
                __sa_restore_meta["errors"].append(__sa_err)
    if __sa_os.path.exists(__sa_state_file):
        try:
            import dill as __sa_pickle
        except Exception:
            import cloudpickle as __sa_pickle
        with open(__sa_state_file, "rb") as __sa_handle:
            __sa_payload = __sa_pickle.load(__sa_handle)
        for __sa_alias, __sa_module_name in (__sa_payload.get("modules") or {{}}).items():
            try:
                if isinstance(__sa_alias, str) and not __sa_alias.startswith("__sf_"):
                    globals()[__sa_alias] = __sa_importlib.import_module(__sa_module_name)
            except Exception as __sa_exc:
                __sa_err = f"module {{__sa_alias}}={{__sa_module_name}}: {{__sa_exc!r}}"
                __sa_restore_meta["module_errors"].append(__sa_err)
                __sa_restore_meta["errors"].append(__sa_err)
        for __sa_name, __sa_value in (__sa_payload.get("globals") or {{}}).items():
            if isinstance(__sa_name, str) and (__sa_name.startswith("__sa_") or __sa_name.startswith("__sf_")):
                continue
            globals()[__sa_name] = __sa_value
        __sa_restore_meta["state_restored"] = True
except Exception as __sa_exc:
    __sa_err = "state_restore: " + repr(__sa_exc)
    __sa_restore_meta["state_errors"].append(__sa_err)
    __sa_restore_meta["errors"].append(__sa_err)
try:
    if __sa_legacy_path_compat:
        __sa_tmp_re = __sa_re.compile(r"/tmp/tmp[A-Za-z0-9_.-]+(?:/[^\\s\\\"'<>]*)?")
        __sa_tmp_prefix_re = __sa_re.compile(r"^/tmp/tmp[A-Za-z0-9_.-]+(?:/(.*))?$")
        __sa_trailing_punct = ".,;:)]}}"

        def __sa_norm_tmp_match(__sa_match):
            __sa_value = __sa_match.group(0)
            __sa_trailing = ""
            while __sa_value and __sa_value[-1] in __sa_trailing_punct:
                __sa_trailing = __sa_value[-1] + __sa_trailing
                __sa_value = __sa_value[:-1]
            __sa_prefix_match = __sa_tmp_prefix_re.match(__sa_value)
            if not __sa_prefix_match:
                return __sa_match.group(0)
            __sa_rel = (__sa_prefix_match.group(1) or "").strip("/")
            return (__sa_rel or ".") + __sa_trailing

        def __sa_norm_path_text(__sa_text):
            return __sa_tmp_re.sub(__sa_norm_tmp_match, __sa_text)

        def __sa_norm_path_value(__sa_value, __sa_depth=0):
            if __sa_depth > 4:
                return __sa_value
            if isinstance(__sa_value, str):
                return __sa_norm_path_text(__sa_value)
            if isinstance(__sa_value, list):
                return [__sa_norm_path_value(__sa_item, __sa_depth + 1) for __sa_item in __sa_value]
            if isinstance(__sa_value, tuple):
                return tuple(__sa_norm_path_value(__sa_item, __sa_depth + 1) for __sa_item in __sa_value)
            if isinstance(__sa_value, set):
                return {{__sa_norm_path_value(__sa_item, __sa_depth + 1) for __sa_item in __sa_value}}
            if isinstance(__sa_value, dict):
                return {{
                    __sa_norm_path_value(__sa_key, __sa_depth + 1): __sa_norm_path_value(__sa_item, __sa_depth + 1)
                    for __sa_key, __sa_item in __sa_value.items()
                }}
            return __sa_value

        __sa_normalized_count = 0
        for __sa_name in list(globals()):
            if not isinstance(__sa_name, str) or __sa_name.startswith("__sa_"):
                continue
            try:
                __sa_old_value = globals()[__sa_name]
                __sa_new_value = __sa_norm_path_value(__sa_old_value)
                if __sa_new_value != __sa_old_value:
                    globals()[__sa_name] = __sa_new_value
                    __sa_normalized_count += 1
            except Exception as __sa_exc:
                __sa_restore_meta["legacy_path_compat_errors"].append(f"{{__sa_name}}: {{__sa_exc!r}}")
        __sa_restore_meta["legacy_path_compat_normalized_globals"] = __sa_normalized_count
except Exception as __sa_exc:
    __sa_err = "legacy_path_compat: " + repr(__sa_exc)
    __sa_restore_meta["legacy_path_compat_errors"].append(__sa_err)
__sa_checkpoint_had_files = False
try:
    __sa_checkpoint_had_files = __sa_os.path.exists(__sa_files_tar) or bool((__sa_previous_meta or {{}}).get("file_manifest"))
except Exception:
    __sa_checkpoint_had_files = __sa_os.path.exists(__sa_files_tar)
__sa_checkpoint_had_state = __sa_os.path.exists(__sa_state_file)
if __sa_restore_meta["checkpoint_present"]:
    __sa_restore_meta["file_restore_failed"] = bool(
        __sa_checkpoint_had_files
        and (
            not __sa_restore_meta["files_restored"]
            or not __sa_restore_meta["manifest_verified"]
            or __sa_restore_meta["file_errors"]
            or __sa_restore_meta["meta_errors"]
        )
    )
    __sa_restore_meta["state_restore_failed"] = bool(
        (__sa_checkpoint_had_state and not __sa_restore_meta["state_restored"])
        or __sa_restore_meta["state_errors"]
        or __sa_restore_meta["module_errors"]
    )
if __sa_restore_meta["file_restore_failed"]:
    __sa_restore_meta["restore_ok"] = False
""".strip()


def _build_checkpoint_pre_current_cell() -> str:
    return """
import os as __sa_os
import hashlib as __sa_hashlib
__sa_pre_exclude_prefixes = (
    ".science_agent_state/",
    "tmp/sandbox/",
    "__pycache__/",
    "input_images/",
)
__sa_pre_exclude_names = {
    "main.py",
    ".science_agent_state",
    "__pycache__",
}
__sa_current_cell_pre_file_manifest = {}
def __sa_is_internal_sandbox_checkpoint_path(__sa_rel):
    __sa_rel = __sa_os.path.normpath(str(__sa_rel))
    return (
        __sa_rel == ".sandbox_abs/dev/null"
        or __sa_rel == ".sandbox_abs/home/sandbox/.ipython"
        or __sa_rel.startswith(".sandbox_abs/home/sandbox/.ipython/")
        or (__sa_rel.startswith(".sandbox_abs/tmp/") and "/profile_default/" in (__sa_rel + "/"))
    )
try:
    for __sa_root, __sa_dirs, __sa_files in __sa_os.walk("."):
        __sa_rel_root = __sa_os.path.relpath(__sa_root, ".")
        if __sa_rel_root == ".":
            __sa_rel_root = ""
        __sa_dirs[:] = [
            __sa_dir for __sa_dir in __sa_dirs
            if __sa_dir not in __sa_pre_exclude_names
            and not (__sa_rel_root + "/" + __sa_dir if __sa_rel_root else __sa_dir).startswith(__sa_pre_exclude_prefixes)
            and not __sa_is_internal_sandbox_checkpoint_path(__sa_rel_root + "/" + __sa_dir if __sa_rel_root else __sa_dir)
        ]
        for __sa_file in __sa_files:
            __sa_rel = __sa_os.path.normpath(__sa_os.path.join(__sa_rel_root, __sa_file))
            if __sa_rel in __sa_pre_exclude_names:
                continue
            if __sa_rel.startswith(__sa_pre_exclude_prefixes):
                continue
            if __sa_is_internal_sandbox_checkpoint_path(__sa_rel):
                continue
            if __sa_os.path.islink(__sa_rel) or not __sa_os.path.isfile(__sa_rel):
                continue
            __sa_stat = __sa_os.stat(__sa_rel)
            __sa_current_cell_pre_file_manifest[__sa_rel] = {
                "size": int(__sa_stat.st_size),
                "mtime_ns": int(getattr(__sa_stat, "st_mtime_ns", int(__sa_stat.st_mtime * 1000000000))),
            }
except Exception as __sa_exc:
    __sa_current_cell_pre_file_manifest = {
        "__snapshot_error__": {"error": repr(__sa_exc)}
    }
""".strip()


def _build_checkpoint_save_cell(
    *,
    max_state_bytes: Optional[int] = None,
    max_files_bytes: Optional[int] = None,
    max_object_bytes: Optional[int] = None,
) -> str:
    max_state_bytes = _checkpoint_max_state_bytes() if max_state_bytes is None else max_state_bytes
    max_files_bytes = _checkpoint_max_files_bytes() if max_files_bytes is None else max_files_bytes
    max_object_bytes = _checkpoint_max_object_bytes() if max_object_bytes is None else max_object_bytes
    return f"""
import os as __sa_os
import json as __sa_json
import tarfile as __sa_tarfile
import io as __sa_io
import types as __sa_types
import hashlib as __sa_hashlib
__sa_state_dir = ".science_agent_state"
__sa_state_file = "{PYTHON_CHECKPOINT_STATE_FILE}"
__sa_files_tar = "{PYTHON_CHECKPOINT_FILES_TAR}"
__sa_meta_file = "{PYTHON_CHECKPOINT_META_FILE}"
__sa_max_state_bytes = {int(max_state_bytes)}
__sa_max_files_bytes = {int(max_files_bytes)}
__sa_max_object_bytes = {int(max_object_bytes)}
__sa_checkpoint_save_retries = {int(PYTHON_CHECKPOINT_SAVE_RETRIES)}
__sa_skip_names = {{
    "In", "Out", "get_ipython", "exit", "quit", "open",
    "_", "__", "___", "__builtin__", "__builtins__",
    "_i", "_ii", "_iii", "_ih", "_oh", "_dh",
}}
__sa_unstable_type_names = {{"function", "method", "builtin_function_or_method"}}
__sa_unstable_module_prefixes = ("flax", "jax", "tensorflow", "torch", "cv2")
__sa_meta = {{
    "state_ok": False,
    "state_checkpoint_ok": False,
    "files_ok": False,
    "files_checkpoint_ok": False,
    "manifest_ok": False,
    "restore_ok": bool(globals().get("__sa_restore_meta", {{}}).get("restore_ok", True)),
    "restore_meta": globals().get("__sa_restore_meta", {{}}),
    "module_aliases": {{}},
    "state_bytes": 0,
    "files_bytes": 0,
    "files_count": 0,
    "file_manifest": {{}},
    "file_save_verified": False,
    "file_verify_errors": [],
    "file_manifest_top": [],
    "largest_file_bytes": 0,
    "current_cell_file_diagnostics": {{}},
    "file_guard_ok": True,
    "file_guard_errors": [],
    "zero_binary_restored_files": [],
    "zero_binary_dropped_files": [],
    "invalid_pdf_restored_files": [],
    "invalid_pdf_dropped_files": [],
    "skipped_files_count": 0,
    "skipped_files": [],
    "skipped_globals": [],
    "checkpoint_save_attempts": 0,
    "checkpoint_save_retry_reasons": [],
    "errors": [],
}}
__sa_file_guard_enabled = {bool(PYTHON_CHECKPOINT_FILE_GUARD)}
__sa_pdf_signature_guard_enabled = {bool(PYTHON_CHECKPOINT_PDF_SIGNATURE_GUARD)}
__sa_protected_suffixes = tuple({repr(list(PYTHON_CHECKPOINT_PROTECTED_SUFFIXES))})
__sa_meta["file_guard_enabled"] = __sa_file_guard_enabled
__sa_meta["pdf_signature_guard_enabled"] = __sa_pdf_signature_guard_enabled
__sa_meta["protected_suffixes"] = list(__sa_protected_suffixes)
__sa_os.makedirs(__sa_state_dir, exist_ok=True)
def __sa_retry_replace(__sa_src, __sa_dst, __sa_label):
    __sa_retry_errors = []
    for __sa_attempt in range(max(1, int(__sa_checkpoint_save_retries))):
        try:
            __sa_os.replace(__sa_src, __sa_dst)
            return __sa_attempt + 1, __sa_retry_errors
        except Exception as __sa_exc:
            __sa_retry_errors.append(
                f"{{__sa_label}}_replace_attempt_{{__sa_attempt + 1}}: {{repr(__sa_exc)}}"
            )
            if __sa_attempt >= max(1, int(__sa_checkpoint_save_retries)) - 1:
                raise
            try:
                import time as __sa_time
                __sa_time.sleep(0.05)
            except Exception:
                pass
    return max(1, int(__sa_checkpoint_save_retries)), __sa_retry_errors

try:
    for __sa_value in list(globals().values()):
        __sa_flush = getattr(__sa_value, "flush", None)
        if callable(__sa_flush):
            try:
                __sa_flush()
            except Exception:
                pass
except Exception:
    pass
try:
    try:
        import dill as __sa_pickle
    except Exception:
        import cloudpickle as __sa_pickle
    __sa_globals = {{}}
    __sa_modules = {{}}
    for __sa_name, __sa_value in list(globals().items()):
        if (
            not isinstance(__sa_name, str)
            or __sa_name.startswith("__sa_")
            or __sa_name.startswith("__sf_")
            or __sa_name in __sa_skip_names
            or (__sa_name.startswith("_i") and __sa_name[2:].isdigit())
        ):
            continue
        if isinstance(__sa_value, __sa_types.ModuleType):
            __sa_modules[__sa_name] = getattr(__sa_value, "__name__", __sa_name)
            continue
        __sa_type = type(__sa_value)
        __sa_type_module = getattr(__sa_type, "__module__", "") or ""
        __sa_type_name = getattr(__sa_type, "__name__", "") or ""
        if __sa_type_name in __sa_unstable_type_names or __sa_type_module.startswith(__sa_unstable_module_prefixes):
            __sa_meta["skipped_globals"].append({{
                "name": __sa_name,
                "reason": "unstable_type",
                "type": f"{{__sa_type_module}}.{{__sa_type_name}}",
            }})
            continue
        try:
            __sa_value_payload = __sa_pickle.dumps(__sa_value)
            __sa_value_size = len(__sa_value_payload)
            if __sa_value_size > __sa_max_object_bytes:
                __sa_meta["skipped_globals"].append({{
                    "name": __sa_name,
                    "reason": "object_too_large",
                    "size": __sa_value_size,
                    "limit": __sa_max_object_bytes,
                    "type": f"{{__sa_type_module}}.{{__sa_type_name}}",
                }})
                continue
            __sa_globals[__sa_name] = __sa_value
        except Exception as __sa_exc:
            __sa_meta["skipped_globals"].append({{
                "name": __sa_name,
                "reason": "pickle_failed",
                "error": repr(__sa_exc),
                "type": f"{{__sa_type_module}}.{{__sa_type_name}}",
            }})
    __sa_meta["module_aliases"] = __sa_modules
    __sa_tmp_state = __sa_state_file + ".tmp"
    with open(__sa_tmp_state, "wb") as __sa_handle:
        __sa_pickle.dump({{"globals": __sa_globals, "modules": __sa_modules}}, __sa_handle)
    __sa_state_size = __sa_os.path.getsize(__sa_tmp_state)
    __sa_meta["state_bytes"] = __sa_state_size
    if __sa_state_size <= __sa_max_state_bytes:
        __sa_attempts, __sa_retry_errors = __sa_retry_replace(__sa_tmp_state, __sa_state_file, "state")
        __sa_meta["checkpoint_save_attempts"] = max(
            int(__sa_meta.get("checkpoint_save_attempts", 0) or 0),
            __sa_attempts,
        )
        __sa_meta["checkpoint_save_retry_reasons"].extend(__sa_retry_errors)
        __sa_meta["state_ok"] = True
        __sa_meta["state_checkpoint_ok"] = True
    else:
        __sa_os.remove(__sa_tmp_state)
        __sa_meta["errors"].append(f"state checkpoint too large: {{__sa_state_size}} > {{__sa_max_state_bytes}}")
except Exception as __sa_exc:
    __sa_meta["errors"].append("state_save: " + repr(__sa_exc))
try:
    __sa_tmp_tar = __sa_files_tar + ".tmp"
    __sa_file_manifest = {{}}
    __sa_file_payloads = {{}}
    __sa_skipped_files = []
    __sa_exclude_prefixes = (
        ".science_agent_state/",
        "tmp/sandbox/",
        "__pycache__/",
        "input_images/",
    )
    __sa_exclude_names = {{
        "main.py",
        ".science_agent_state",
        "__pycache__",
    }}
    def __sa_is_internal_sandbox_checkpoint_path(__sa_rel):
        __sa_rel = __sa_os.path.normpath(str(__sa_rel))
        return (
            __sa_rel == ".sandbox_abs/dev/null"
            or __sa_rel == ".sandbox_abs/home/sandbox/.ipython"
            or __sa_rel.startswith(".sandbox_abs/home/sandbox/.ipython/")
            or (__sa_rel.startswith(".sandbox_abs/tmp/") and "/profile_default/" in (__sa_rel + "/"))
        )
    with __sa_tarfile.open(__sa_tmp_tar, "w:gz") as __sa_tf:
        for __sa_root, __sa_dirs, __sa_files in __sa_os.walk("."):
            __sa_rel_root = __sa_os.path.relpath(__sa_root, ".")
            if __sa_rel_root == ".":
                __sa_rel_root = ""
            __sa_dirs[:] = [
                __sa_dir for __sa_dir in __sa_dirs
                if __sa_dir not in __sa_exclude_names
                and not (__sa_rel_root + "/" + __sa_dir if __sa_rel_root else __sa_dir).startswith(__sa_exclude_prefixes)
                and not __sa_is_internal_sandbox_checkpoint_path(__sa_rel_root + "/" + __sa_dir if __sa_rel_root else __sa_dir)
            ]
            for __sa_file in __sa_files:
                __sa_rel = __sa_os.path.normpath(__sa_os.path.join(__sa_rel_root, __sa_file))
                if __sa_rel in __sa_exclude_names:
                    continue
                if __sa_rel.startswith(__sa_exclude_prefixes):
                    continue
                if __sa_is_internal_sandbox_checkpoint_path(__sa_rel):
                    continue
                if __sa_rel == __sa_files_tar or __sa_rel == __sa_state_file or __sa_rel == __sa_meta_file:
                    continue
                if __sa_os.path.islink(__sa_rel) or not __sa_os.path.isfile(__sa_rel):
                    __sa_skipped_files.append({{"path": __sa_rel, "reason": "not_regular_file"}})
                    continue
                __sa_file_size = __sa_os.path.getsize(__sa_rel)
                if __sa_file_size > __sa_max_files_bytes:
                    __sa_skipped_files.append({{"path": __sa_rel, "reason": "file_too_large", "size": __sa_file_size}})
                    continue
                with open(__sa_rel, "rb") as __sa_file_handle:
                    __sa_data = __sa_file_handle.read()
                __sa_hasher = __sa_hashlib.sha256(__sa_data)
                __sa_file_manifest[__sa_rel] = {{
                    "size": len(__sa_data),
                    "sha256": __sa_hasher.hexdigest(),
                }}
                __sa_file_payloads[__sa_rel] = __sa_data
        __sa_previous_meta = globals().get("__sa_previous_meta") or {{}}
        if not isinstance(__sa_previous_meta, dict):
            __sa_previous_meta = {{}}
        __sa_previous_manifest = __sa_previous_meta.get("file_manifest") or {{}}
        if not isinstance(__sa_previous_manifest, dict):
            __sa_previous_manifest = {{}}
        if __sa_file_guard_enabled:
            __sa_zero_binary_restored = []
            __sa_zero_binary_dropped = []
            __sa_invalid_pdf_restored = []
            __sa_invalid_pdf_dropped = []
            __sa_guard_errors = []

            def __sa_guard_size(__sa_info):
                try:
                    return int((__sa_info or {{}}).get("size", 0) or 0)
                except Exception:
                    return 0

            def __sa_guard_item(__sa_rel, __sa_data, __sa_reason, __sa_previous_info=None, __sa_error=None):
                __sa_item = {{
                    "path": __sa_rel,
                    "size": len(__sa_data or b""),
                    "reason": __sa_reason,
                }}
                __sa_previous_size = __sa_guard_size(__sa_previous_info)
                if __sa_previous_size:
                    __sa_item["checkpoint_previous_size"] = __sa_previous_size
                if __sa_error:
                    __sa_item["error"] = str(__sa_error)[:300]
                return __sa_item

            def __sa_read_previous_file(__sa_rel, __sa_previous_info):
                if not __sa_os.path.exists(__sa_files_tar):
                    return None, "previous_checkpoint_tar_missing"
                try:
                    with __sa_tarfile.open(__sa_files_tar, "r:gz") as __sa_prev_tf:
                        try:
                            __sa_member = __sa_prev_tf.getmember(__sa_rel)
                        except KeyError:
                            return None, "previous_file_missing"
                        if not __sa_member.isfile():
                            return None, "previous_file_not_regular"
                        __sa_extracted = __sa_prev_tf.extractfile(__sa_member)
                        if __sa_extracted is None:
                            return None, "previous_file_extract_failed"
                        __sa_previous_data = __sa_extracted.read()
                except Exception as __sa_exc:
                    return None, "previous_file_read_failed: " + repr(__sa_exc)
                __sa_expected_size = int((__sa_previous_info or {{}}).get("size", -1) or -1)
                if __sa_expected_size >= 0 and len(__sa_previous_data) != __sa_expected_size:
                    return None, "previous_file_size_mismatch"
                __sa_expected_sha = str((__sa_previous_info or {{}}).get("sha256") or "")
                if __sa_expected_sha:
                    __sa_actual_sha = __sa_hashlib.sha256(__sa_previous_data).hexdigest()
                    if __sa_actual_sha != __sa_expected_sha:
                        return None, "previous_file_sha256_mismatch"
                return __sa_previous_data, ""

            def __sa_apply_restored_file(__sa_rel, __sa_data):
                __sa_file_payloads[__sa_rel] = __sa_data
                __sa_file_manifest[__sa_rel] = {{
                    "size": len(__sa_data),
                    "sha256": __sa_hashlib.sha256(__sa_data).hexdigest(),
                }}
                try:
                    __sa_parent = __sa_os.path.dirname(__sa_rel)
                    if __sa_parent:
                        __sa_os.makedirs(__sa_parent, exist_ok=True)
                    with open(__sa_rel, "wb") as __sa_restore_handle:
                        __sa_restore_handle.write(__sa_data)
                except Exception as __sa_exc:
                    __sa_guard_errors.append("restore_writeback " + __sa_rel + ": " + repr(__sa_exc))

            try:
                for __sa_rel in sorted(list(__sa_file_payloads)):
                    __sa_data = __sa_file_payloads.get(__sa_rel, b"")
                    __sa_rel_lower = str(__sa_rel).lower()
                    __sa_previous_info = __sa_previous_manifest.get(__sa_rel)
                    __sa_previous_size = __sa_guard_size(__sa_previous_info)
                    __sa_is_protected = bool(__sa_protected_suffixes and __sa_rel_lower.endswith(__sa_protected_suffixes))
                    __sa_is_pdf = __sa_rel_lower.endswith(".pdf")
                    __sa_current_sha = __sa_hashlib.sha256(__sa_data).hexdigest()
                    __sa_previous_sha = str((__sa_previous_info or {{}}).get("sha256") or "")
                    __sa_same_as_previous = bool(
                        __sa_previous_info
                        and __sa_previous_size == len(__sa_data)
                        and __sa_previous_sha
                        and __sa_previous_sha == __sa_current_sha
                    )

                    __sa_zero_protected = __sa_is_protected and len(__sa_data) == 0
                    __sa_invalid_pdf = bool(
                        __sa_pdf_signature_guard_enabled
                        and __sa_is_pdf
                        and len(__sa_data) > 0
                        and not __sa_same_as_previous
                        and b"%PDF-" not in __sa_data[:1024]
                    )
                    if not (__sa_zero_protected or __sa_invalid_pdf):
                        continue

                    __sa_reason = "zero_binary_file" if __sa_zero_protected else "invalid_pdf_signature"
                    __sa_previous_data = None
                    __sa_previous_error = ""
                    if __sa_previous_size > 0:
                        __sa_previous_data, __sa_previous_error = __sa_read_previous_file(__sa_rel, __sa_previous_info)
                        if (
                            __sa_previous_data
                            and __sa_pdf_signature_guard_enabled
                            and __sa_is_pdf
                            and b"%PDF-" not in __sa_previous_data[:1024]
                        ):
                            __sa_previous_data = None
                            __sa_previous_error = "previous_pdf_signature_invalid"
                    if __sa_previous_data:
                        __sa_apply_restored_file(__sa_rel, __sa_previous_data)
                        if __sa_zero_protected:
                            __sa_zero_binary_restored.append(
                                __sa_guard_item(__sa_rel, __sa_previous_data, __sa_reason, __sa_previous_info)
                            )
                        else:
                            __sa_invalid_pdf_restored.append(
                                __sa_guard_item(__sa_rel, __sa_previous_data, __sa_reason, __sa_previous_info)
                            )
                        continue

                    __sa_file_payloads.pop(__sa_rel, None)
                    __sa_file_manifest.pop(__sa_rel, None)
                    try:
                        if __sa_os.path.isfile(__sa_rel):
                            __sa_os.remove(__sa_rel)
                    except Exception as __sa_exc:
                        __sa_guard_errors.append("drop_remove " + __sa_rel + ": " + repr(__sa_exc))
                    __sa_skipped_files.append({{
                        "path": __sa_rel,
                        "reason": __sa_reason + "_dropped_from_checkpoint",
                        "size": len(__sa_data),
                        "previous_size": __sa_previous_size,
                        "previous_error": __sa_previous_error,
                    }})
                    if __sa_zero_protected:
                        __sa_zero_binary_dropped.append(
                            __sa_guard_item(__sa_rel, __sa_data, __sa_reason, __sa_previous_info, __sa_previous_error)
                        )
                    else:
                        __sa_invalid_pdf_dropped.append(
                            __sa_guard_item(__sa_rel, __sa_data, __sa_reason, __sa_previous_info, __sa_previous_error)
                        )
            except Exception as __sa_exc:
                __sa_meta["file_guard_ok"] = False
                __sa_guard_errors.append("file_guard: " + repr(__sa_exc))
            __sa_meta["file_guard_errors"] = __sa_guard_errors
            __sa_meta["zero_binary_restored_files"] = __sa_zero_binary_restored[:20]
            __sa_meta["zero_binary_dropped_files"] = __sa_zero_binary_dropped[:20]
            __sa_meta["invalid_pdf_restored_files"] = __sa_invalid_pdf_restored[:20]
            __sa_meta["invalid_pdf_dropped_files"] = __sa_invalid_pdf_dropped[:20]
        for __sa_rel in sorted(__sa_file_payloads):
            __sa_data = __sa_file_payloads[__sa_rel]
            __sa_info = __sa_tarfile.TarInfo(__sa_rel)
            __sa_info.size = len(__sa_data)
            __sa_info.mtime = 0
            __sa_info.mode = 0o644
            __sa_tf.addfile(__sa_info, __sa_io.BytesIO(__sa_data))
    __sa_files_size = __sa_os.path.getsize(__sa_tmp_tar)
    __sa_meta["files_bytes"] = __sa_files_size
    __sa_meta["files_count"] = len(__sa_file_manifest)
    __sa_meta["file_manifest"] = __sa_file_manifest
    __sa_top_files = sorted(
        (
            {{"path": __sa_rel, "size": int((__sa_info or {{}}).get("size", 0) or 0)}}
            for __sa_rel, __sa_info in __sa_file_manifest.items()
        ),
        key=lambda __sa_item: __sa_item["size"],
        reverse=True,
    )[:5]
    __sa_meta["file_manifest_top"] = __sa_top_files
    __sa_meta["largest_file_bytes"] = __sa_top_files[0]["size"] if __sa_top_files else 0
    __sa_meta["skipped_files"] = __sa_skipped_files[:20]
    __sa_meta["skipped_files_count"] = len(__sa_skipped_files)
    try:
        __sa_diag_limit = {int(PYTHON_FILE_DIAGNOSTIC_TOP_N)}
        __sa_pre_manifest = globals().get("__sa_current_cell_pre_file_manifest") or {{}}
        if not isinstance(__sa_pre_manifest, dict):
            __sa_pre_manifest = {{}}
        __sa_previous_meta = globals().get("__sa_previous_meta") or {{}}
        if not isinstance(__sa_previous_meta, dict):
            __sa_previous_meta = {{}}
        __sa_previous_manifest = __sa_previous_meta.get("file_manifest") or {{}}
        if not isinstance(__sa_previous_manifest, dict):
            __sa_previous_manifest = {{}}

        def __sa_diag_size(__sa_info):
            try:
                return int((__sa_info or {{}}).get("size", 0) or 0)
            except Exception:
                return 0

        def __sa_diag_item(__sa_rel, __sa_info, __sa_extra=None):
            __sa_item = {{
                "path": __sa_rel,
                "size": __sa_diag_size(__sa_info),
            }}
            __sa_sha = str((__sa_info or {{}}).get("sha256") or "")
            if __sa_sha:
                __sa_item["sha256_prefix"] = __sa_sha[:16]
            if isinstance(__sa_extra, dict):
                __sa_item.update(__sa_extra)
            return __sa_item

        __sa_created = []
        __sa_modified = []
        __sa_deleted = []
        __sa_zero = []
        __sa_zero_created = []
        __sa_pre_nonzero_now_zero = []
        __sa_previous_nonzero_now_zero = []
        for __sa_rel, __sa_info in __sa_file_manifest.items():
            __sa_size = __sa_diag_size(__sa_info)
            __sa_pre_info = __sa_pre_manifest.get(__sa_rel)
            __sa_previous_info = __sa_previous_manifest.get(__sa_rel)
            if __sa_size == 0:
                __sa_zero.append(__sa_diag_item(__sa_rel, __sa_info))
            if __sa_pre_info is None:
                __sa_created.append(__sa_diag_item(__sa_rel, __sa_info))
                if __sa_size == 0:
                    __sa_zero_created.append(__sa_diag_item(__sa_rel, __sa_info))
            else:
                __sa_pre_size = __sa_diag_size(__sa_pre_info)
                if __sa_pre_size != __sa_size:
                    __sa_modified.append(
                        __sa_diag_item(
                            __sa_rel,
                            __sa_info,
                            {{"previous_size": __sa_pre_size}},
                        )
                    )
                if __sa_pre_size > 0 and __sa_size == 0:
                    __sa_pre_nonzero_now_zero.append(
                        __sa_diag_item(
                            __sa_rel,
                            __sa_info,
                            {{"previous_size": __sa_pre_size}},
                        )
                    )
            __sa_previous_size = __sa_diag_size(__sa_previous_info)
            if __sa_previous_size > 0 and __sa_size == 0:
                __sa_previous_nonzero_now_zero.append(
                    __sa_diag_item(
                        __sa_rel,
                        __sa_info,
                        {{"checkpoint_previous_size": __sa_previous_size}},
                    )
                )

        for __sa_rel, __sa_info in __sa_pre_manifest.items():
            if __sa_rel == "__snapshot_error__":
                continue
            if __sa_rel not in __sa_file_manifest:
                __sa_deleted.append(__sa_diag_item(__sa_rel, __sa_info))

        __sa_pdf_like_suffixes = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
        __sa_zero_binary_like = [
            __sa_item for __sa_item in __sa_zero
            if str(__sa_item.get("path", "")).lower().endswith(__sa_pdf_like_suffixes)
        ]
        __sa_meta["current_cell_file_diagnostics"] = {{
            "before_count": len([__sa_key for __sa_key in __sa_pre_manifest if __sa_key != "__snapshot_error__"]),
            "after_count": len(__sa_file_manifest),
            "created_count": len(__sa_created),
            "modified_count": len(__sa_modified),
            "deleted_count": len(__sa_deleted),
            "zero_byte_count": len(__sa_zero),
            "zero_byte_binary_like_count": len(__sa_zero_binary_like),
            "created": sorted(__sa_created, key=lambda __sa_item: __sa_item["size"], reverse=True)[:__sa_diag_limit],
            "modified": sorted(__sa_modified, key=lambda __sa_item: __sa_item["size"], reverse=True)[:__sa_diag_limit],
            "deleted": sorted(__sa_deleted, key=lambda __sa_item: __sa_item["size"], reverse=True)[:__sa_diag_limit],
            "zero_byte_files": sorted(__sa_zero, key=lambda __sa_item: __sa_item["path"])[:__sa_diag_limit],
            "zero_byte_binary_like_files": sorted(__sa_zero_binary_like, key=lambda __sa_item: __sa_item["path"])[:__sa_diag_limit],
            "zero_byte_created_files": sorted(__sa_zero_created, key=lambda __sa_item: __sa_item["path"])[:__sa_diag_limit],
            "pre_nonzero_now_zero": sorted(__sa_pre_nonzero_now_zero, key=lambda __sa_item: __sa_item["path"])[:__sa_diag_limit],
            "checkpoint_previous_nonzero_now_zero": sorted(
                __sa_previous_nonzero_now_zero,
                key=lambda __sa_item: __sa_item["path"],
            )[:__sa_diag_limit],
            "zero_binary_restored_files": __sa_meta.get("zero_binary_restored_files", []),
            "zero_binary_dropped_files": __sa_meta.get("zero_binary_dropped_files", []),
            "invalid_pdf_restored_files": __sa_meta.get("invalid_pdf_restored_files", []),
            "invalid_pdf_dropped_files": __sa_meta.get("invalid_pdf_dropped_files", []),
        }}
        if "__snapshot_error__" in __sa_pre_manifest:
            __sa_meta["current_cell_file_diagnostics"]["pre_snapshot_error"] = __sa_pre_manifest["__snapshot_error__"]
    except Exception as __sa_exc:
        __sa_meta["current_cell_file_diagnostics"] = {{"error": repr(__sa_exc)}}
    __sa_verify_errors = []
    try:
        with __sa_tarfile.open(__sa_tmp_tar, "r:gz") as __sa_tf:
            __sa_seen = set()
            for __sa_member in __sa_tf.getmembers():
                if __sa_member.isdir():
                    continue
                __sa_rel = __sa_os.path.normpath(str(__sa_member.name))
                if (
                    not __sa_rel
                    or __sa_rel == "."
                    or __sa_os.path.isabs(__sa_rel)
                    or __sa_rel.startswith("..")
                    or __sa_rel != __sa_member.name
                ):
                    __sa_verify_errors.append(f"unsafe tar member: {{__sa_member.name}}")
                    continue
                if not __sa_member.isfile():
                    __sa_verify_errors.append(f"non-regular tar member: {{__sa_rel}}")
                    continue
                __sa_seen.add(__sa_rel)
                __sa_expected = __sa_file_manifest.get(__sa_rel)
                if __sa_expected is None:
                    __sa_verify_errors.append(f"unexpected tar member: {{__sa_rel}}")
                    continue
                __sa_extracted = __sa_tf.extractfile(__sa_member)
                __sa_actual_data = __sa_extracted.read() if __sa_extracted is not None else b""
                __sa_actual_size = len(__sa_actual_data)
                __sa_expected_size = int((__sa_expected or {{}}).get("size", -1))
                if __sa_expected_size >= 0 and __sa_actual_size != __sa_expected_size:
                    __sa_verify_errors.append(f"size mismatch: {{__sa_rel}} {{__sa_actual_size}} != {{__sa_expected_size}}")
                    continue
                __sa_expected_sha = str((__sa_expected or {{}}).get("sha256") or "")
                if __sa_expected_sha:
                    __sa_actual_sha = __sa_hashlib.sha256(__sa_actual_data).hexdigest()
                    if __sa_actual_sha != __sa_expected_sha:
                        __sa_verify_errors.append(f"sha256 mismatch: {{__sa_rel}} {{__sa_actual_sha}} != {{__sa_expected_sha}}")
            for __sa_rel in sorted(set(__sa_file_manifest) - __sa_seen):
                __sa_verify_errors.append(f"missing tar member: {{__sa_rel}}")
    except Exception as __sa_exc:
        __sa_verify_errors.append("file checkpoint verify: " + repr(__sa_exc))
    __sa_meta["file_verify_errors"] = __sa_verify_errors
    if __sa_verify_errors:
        try:
            __sa_os.remove(__sa_tmp_tar)
        except Exception:
            pass
        __sa_meta["errors"].append("file checkpoint verification failed")
    elif __sa_files_size <= __sa_max_files_bytes:
        __sa_attempts, __sa_retry_errors = __sa_retry_replace(__sa_tmp_tar, __sa_files_tar, "files")
        __sa_meta["checkpoint_save_attempts"] = max(
            int(__sa_meta.get("checkpoint_save_attempts", 0) or 0),
            __sa_attempts,
        )
        __sa_meta["checkpoint_save_retry_reasons"].extend(__sa_retry_errors)
        __sa_meta["files_ok"] = True
        __sa_meta["files_checkpoint_ok"] = True
        __sa_meta["manifest_ok"] = True
        __sa_meta["file_save_verified"] = True
    else:
        __sa_os.remove(__sa_tmp_tar)
        __sa_meta["errors"].append(f"file checkpoint too large: {{__sa_files_size}} > {{__sa_max_files_bytes}}")
except Exception as __sa_exc:
    __sa_meta["errors"].append("files_save: " + repr(__sa_exc))
try:
    for __sa_meta_attempt in range(max(1, int(__sa_checkpoint_save_retries))):
        try:
            __sa_meta["checkpoint_save_attempts"] = max(
                int(__sa_meta.get("checkpoint_save_attempts", 0) or 0),
                __sa_meta_attempt + 1,
            )
            with open(__sa_meta_file, "w", encoding="utf-8") as __sa_handle:
                __sa_json.dump(__sa_meta, __sa_handle, ensure_ascii=False, sort_keys=True)
            break
        except Exception as __sa_exc:
            __sa_meta["checkpoint_save_retry_reasons"].append(
                f"meta_write_attempt_{{__sa_meta_attempt + 1}}: {{repr(__sa_exc)}}"
            )
            if __sa_meta_attempt >= max(1, int(__sa_checkpoint_save_retries)) - 1:
                raise
            try:
                import time as __sa_time
                __sa_time.sleep(0.05)
            except Exception:
                pass
except Exception as __sa_exc:
    try:
        __sa_meta["errors"].append("manifest_save: " + repr(__sa_exc))
    except Exception:
        pass
""".strip()


def normalize_python_code_input(params: Any) -> str:
    code = params
    if isinstance(params, dict):
        code = params.get("code") or params.get("raw") or ""
    if not isinstance(code, str):
        code = str(code)

    triple_match = re.search(r"```[^\n]*\n(.+?)```", code, re.DOTALL)
    if triple_match:
        code = triple_match.group(1)
    return code


def python_request_blocked(params: Any, code: str) -> bool:
    return is_value_blocked(params) or is_text_blocked(code)


def build_jupyter_payload(
    cells: List[str],
    timeout: int,
    *,
    files: Optional[dict[str, str]] = None,
    fetch_files: Optional[list[str]] = None,
) -> dict:
    return {
        "cells": cells,
        "kernel": PYTHON_JUPYTER_KERNEL,
        "cell_timeout": 0,
        "total_timeout": timeout,
        "files": files or {},
        "fetch_files": fetch_files or [],
    }


def post_jupyter_request(
    cells: List[str],
    timeout: int,
    endpoint: str,
    *,
    files: Optional[dict[str, str]] = None,
    fetch_files: Optional[list[str]] = None,
) -> dict:
    url = normalize_sandbox_fusion_endpoint(endpoint) + "/run_jupyter"
    response = requests.post(
        url,
        json=build_jupyter_payload(
            cells,
            timeout,
            files=files,
            fetch_files=fetch_files,
        ),
        timeout=timeout + 10,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _stringify_display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def format_jupyter_display(display_item: Any) -> str:
    if not isinstance(display_item, dict):
        return _stringify_display_value(display_item)

    if isinstance(display_item.get("data"), dict):
        display_item = display_item["data"]

    for key in ("text/plain", "text/markdown", "text/html"):
        value = display_item.get(key)
        if value:
            return _stringify_display_value(value)

    media_keys = [key for key in display_item if key.startswith("image/")]
    if media_keys:
        return ", ".join(f"[{key} display data]" for key in media_keys)

    return json.dumps(display_item, ensure_ascii=False)


def format_jupyter_error(error_item: Any) -> str:
    if not isinstance(error_item, dict):
        return _stringify_display_value(error_item)

    ename = str(error_item.get("ename") or "").strip()
    evalue = str(error_item.get("evalue") or "").strip()
    header = f"{ename}: {evalue}".strip(": ")
    traceback = error_item.get("traceback") or []
    if isinstance(traceback, str):
        traceback_lines = [traceback]
    elif isinstance(traceback, list):
        traceback_lines = [str(line) for line in traceback]
    else:
        traceback_lines = [json.dumps(traceback, ensure_ascii=False)]

    lines = [line for line in [header, *traceback_lines] if line]
    return "\n".join(lines) if lines else json.dumps(error_item, ensure_ascii=False)


def _relative_path_policy_active() -> bool:
    return should_enforce_relative_paths(
        "PYTHON_TOOL_PATH_POLICY",
        default=PATH_POLICY_RELATIVE,
    )


def _normalize_sandbox_tmp_path(path: str) -> str:
    path = str(path or "")
    prefix_match = re.match(r"^/tmp/tmp[A-Za-z0-9_.-]+(?:/(.*))?$", path)
    if not prefix_match:
        return path
    relative = prefix_match.group(1) or ""
    relative = relative.strip("/")
    return relative or "."


def normalize_sandbox_tmp_paths(text: Any) -> str:
    output = str(text or "")
    if not output:
        return output

    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2)
        trailing = ""
        while value and value[-1] in SANDBOX_TMP_TRAILING_PUNCTUATION:
            trailing = value[-1] + trailing
            value = value[:-1]
        return f"{prefix}{_normalize_sandbox_tmp_path(value)}{trailing}"

    return SANDBOX_TMP_PATH_RE.sub(replace, output)


def sanitize_sandbox_paths_for_relative_policy(text: Any) -> str:
    if not _relative_path_policy_active():
        return str(text or "")
    return normalize_sandbox_tmp_paths(text)


def python_checkpoint_legacy_path_compat_enabled() -> bool:
    return bool(
        PYTHON_CHECKPOINT_LEGACY_PATH_COMPAT
        and _checkpoint_backend_enabled()
        and not _relative_path_policy_active()
    )


def sanitize_sandbox_paths_for_checkpoint_legacy(text: Any) -> str:
    if not python_checkpoint_legacy_path_compat_enabled():
        return str(text or "")
    return normalize_sandbox_tmp_paths(text)


def normalize_checkpoint_legacy_code_paths(code: str) -> str:
    if not python_checkpoint_legacy_path_compat_enabled():
        return code
    return normalize_sandbox_tmp_paths(code)


def _coerce_jupyter_cells(response_data: dict) -> list:
    cells = response_data.get("cells")
    if isinstance(cells, list):
        return cells
    run_result = response_data.get("run_result")
    if isinstance(run_result, dict):
        for key in ("cells", "cell_results"):
            cells = run_result.get(key)
            if isinstance(cells, list):
                return cells
    if isinstance(run_result, list):
        return run_result
    return []


def _coerce_current_jupyter_cell(response_data: dict, current_cell_index: int) -> dict:
    cells = _coerce_jupyter_cells(response_data)
    if 0 <= current_cell_index < len(cells) and isinstance(cells[current_cell_index], dict):
        return cells[current_cell_index]
    if cells and isinstance(cells[-1], dict):
        return cells[-1]
    return {}


def _has_jupyter_cell_result(response_data: dict, current_cell_index: int) -> bool:
    cells = _coerce_jupyter_cells(response_data)
    return 0 <= current_cell_index < len(cells)


def format_jupyter_tool_output(response_data: dict, current_cell_index: int, timeout: int) -> str:
    result = []
    cell = _coerce_current_jupyter_cell(response_data, current_cell_index)

    stdout = cell.get("stdout", "")
    stderr = cell.get("stderr", "")
    stdout_chars = len(str(stdout or ""))
    stderr_chars = len(str(stderr or ""))
    if stdout:
        result.append(f"stdout:\n{stdout}")
    if stderr:
        result.append(f"stderr:\n{stderr}")

    display_items = cell.get("display") or []
    if not isinstance(display_items, list):
        display_items = [display_items]
    display_chars = 0
    for display_item in display_items:
        display_text = format_jupyter_display(display_item).strip()
        display_chars += len(display_text)
        if display_text:
            result.append(f"display:\n{display_text}")

    error_items = cell.get("error") or []
    if not isinstance(error_items, list):
        error_items = [error_items]
    error_chars = 0
    for error_item in error_items:
        error_text = format_jupyter_error(error_item).strip()
        error_chars += len(error_text)
        if error_text:
            result.append(f"error:\n{error_text}")

    run_result = response_data.get("run_result")
    driver = response_data.get("driver") or {}
    if not driver and isinstance(run_result, dict):
        driver = run_result.get("driver") or {}
    status = response_data.get("status")
    message = response_data.get("message")
    execution_time = float(driver.get("execution_time", 0.0) or 0.0) if isinstance(driver, dict) else 0.0
    driver_status = driver.get("status") if isinstance(driver, dict) else None
    driver_return_code = driver.get("return_code") if isinstance(driver, dict) else None
    driver_stderr_tail = ""
    if status and status != "Success" and isinstance(driver, dict):
        driver_stderr_tail = str(driver.get("stderr") or "")[-2000:]
    if status and status != "Success":
        suffix = f": {message}" if message else ""
        result.append(f"[Python Interpreter Error]: Jupyter execution status={status}{suffix}")
    if driver_status == "TimeLimitExceeded" or execution_time >= timeout - 1:
        result.append("[Python Interpreter Error] TimeoutError: Execution timed out.")

    combined = "\n".join(result)
    output = combined or PYTHON_TOOL_NO_OUTPUT_MESSAGE
    output = sanitize_sandbox_paths_for_relative_policy(output)
    output = sanitize_sandbox_paths_for_checkpoint_legacy(output)
    emit_trace(
        "python_tool_output_collected",
        layer="tool_python",
        current_cell_index=current_cell_index,
        status=status,
        driver_status=driver_status,
        driver_return_code=driver_return_code,
        driver_stderr_tail=driver_stderr_tail,
        execution_time=execution_time,
        stdout_chars=stdout_chars,
        stderr_chars=stderr_chars,
        display_items=len(display_items),
        display_chars=display_chars,
        error_items=len(error_items),
        error_chars=error_chars,
        output_chars=len(output),
    )
    if is_text_blocked(output):
        return blocked_python_message()
    return output


def jupyter_current_cell_succeeded(
    response_data: dict,
    current_cell_index: int,
    timeout: int,
    output: str,
) -> bool:
    if output == blocked_python_message():
        return False
    if not _has_jupyter_cell_result(response_data, current_cell_index):
        return False

    run_result = response_data.get("run_result")
    driver = response_data.get("driver") or {}
    if not driver and isinstance(run_result, dict):
        driver = run_result.get("driver") or {}

    status = response_data.get("status")
    if status and status != "Success":
        return False

    execution_time = float(driver.get("execution_time", 0.0) or 0.0) if isinstance(driver, dict) else 0.0
    driver_status = driver.get("status") if isinstance(driver, dict) else None
    if driver_status == "TimeLimitExceeded" or execution_time >= timeout - 1:
        return False

    cell = _coerce_current_jupyter_cell(response_data, current_cell_index)
    error_items = cell.get("error") or []
    if error_items:
        return False

    lowered = str(output or "").lower()
    failure_markers = (
        "[python interpreter error]",
        "[visual python interpreter error]",
        "traceback",
        "filenotfounderror",
        "timeouterror",
    )
    return not any(marker in lowered for marker in failure_markers)


def python_path_policy_error(code: str) -> str:
    if not should_enforce_relative_paths(
        "PYTHON_TOOL_PATH_POLICY",
        default=PATH_POLICY_RELATIVE,
    ):
        return ""
    violations = find_relative_path_policy_violations(code)
    if not violations:
        return ""
    return format_path_policy_error("Python Interpreter", violations)


def python_relative_path_policy_enabled() -> bool:
    return _relative_path_policy_active()


def python_interpreter_description() -> str:
    if python_relative_path_policy_enabled():
        return PYTHON_INTERPRETER_DESCRIPTION_RELATIVE
    return PYTHON_INTERPRETER_DESCRIPTION_LEGACY


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = PYTHON_INTERPRETER_DESCRIPTION_RELATIVE
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A Python code cell to execute.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        if self.name == "PythonInterpreter":
            self.description = python_interpreter_description()
        self._jupyter_cells: List[str] = []
        self._checkpoint_recovery_cells: List[str] = []
        self._checkpoint_files: dict[str, str] = {}
        self._checkpoint_version = 0
        self._checkpoint_last_meta: dict[str, Any] = {}
        self._checkpoint_degraded_reason: Optional[str] = None
        self._checkpoint_last_restore_failed = False
        self._checkpoint_last_state_restore_failed = False
        self._sandbox_endpoint: Optional[str] = None
        self._artifact_dir: Optional[Path] = None
        self._last_artifact_entries: list[dict[str, Any]] = []
        self._agent_sandbox: AgentSandboxRun | None = None
        # self.summary_mapping = SummaryMapping()

    def reset_jupyter_history(self) -> None:
        self._jupyter_cells.clear()
        self._checkpoint_recovery_cells.clear()
        self._checkpoint_files.clear()
        self._checkpoint_version = 0
        self._checkpoint_last_meta = {}
        self._checkpoint_degraded_reason = None
        self._checkpoint_last_restore_failed = False
        self._checkpoint_last_state_restore_failed = False
        self._sandbox_endpoint = None
        self._last_artifact_entries = []
        if self._agent_sandbox is not None:
            self._agent_sandbox.close()
        self._agent_sandbox = None

    def close_sandbox(self) -> None:
        """Release the run-scoped gateway Sandbox, if this backend owns one."""
        if self._agent_sandbox is not None:
            self._agent_sandbox.close()
            self._agent_sandbox = None

    def set_task_context(
        self,
        item: dict[str, Any],
        *,
        base_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> str:
        del item, base_dir
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None
        return ""

    def _checkpoint_enabled(self) -> bool:
        return _checkpoint_backend_enabled()

    def _checkpoint_trace(self, event: str, **fields: Any) -> None:
        emit_trace(
            event,
            layer="tool_python",
            backend=PYTHON_SESSION_BACKEND,
            tool=self.name,
            checkpoint_version=self._checkpoint_version,
            checkpoint_tail_cells=len(self._checkpoint_recovery_cells),
            **fields,
        )

    def _checkpoint_payload_files(
        self,
        extra_files: Optional[dict[str, str]] = None,
        *,
        include_checkpoint: bool = True,
    ) -> dict[str, str]:
        payload: dict[str, str] = {}
        if include_checkpoint:
            payload.update(self._checkpoint_files)
        payload.update(extra_files or {})
        return payload

    def _artifact_local_rel_for_sandbox_rel(self, rel: str) -> str:
        if rel == TOOL_OUTPUTS_DIR or rel.startswith(f"{TOOL_OUTPUTS_DIR}/"):
            return rel
        return os.path.normpath(os.path.join(TOOL_OUTPUTS_DIR, "_sandbox_root", rel))

    def _safe_artifact_target(self, local_rel: str) -> Path:
        if self._artifact_dir is None:
            raise ValueError("artifact_dir is not set")
        rel = _safe_checkpoint_rel_path(local_rel)
        root = self._artifact_dir.resolve()
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe artifact path: {local_rel}")
        return target

    def _append_tool_artifact_manifest(self, entries: list[dict[str, Any]]) -> None:
        if not entries or self._artifact_dir is None:
            return
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._artifact_dir / "tool_outputs_manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_local_tool_artifact(
        self,
        *,
        sandbox_rel_path: str,
        local_rel_path: str,
        data: bytes,
        source: str,
        list_for_model: bool = True,
    ) -> Optional[dict[str, Any]]:
        if self._artifact_dir is None or not _tool_outputs_sync_enabled():
            return None
        target = self._safe_artifact_target(local_rel_path)
        sha = hashlib.sha256(data).hexdigest()
        unchanged = False
        if target.exists() and target.is_file():
            try:
                unchanged = hashlib.sha256(target.read_bytes()).hexdigest() == sha
            except Exception:
                unchanged = False
        if not unchanged:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        else:
            return None
        entry = {
            "ts": time.time(),
            "tool": self.name,
            "source": source,
            "sandbox_rel_path": sandbox_rel_path,
            "local_rel_path": local_rel_path,
            "local_path": str(target),
            "size": len(data),
            "sha256": sha,
            "checkpoint_version": self._checkpoint_version,
        }
        self._append_tool_artifact_manifest([entry])
        return entry if list_for_model else None

    def _sync_checkpoint_artifacts_from_files(
        self,
        files: dict[str, str],
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self._artifact_dir is None or not TOOL_OUTPUTS_SYNC_LOCAL:
            return []
        tar_b64 = files.get(PYTHON_CHECKPOINT_FILES_TAR)
        if not isinstance(tar_b64, str) or not tar_b64:
            return []
        manifest = meta.get("file_manifest") or {}
        if not isinstance(manifest, dict):
            manifest = {}

        entries: list[dict[str, Any]] = []
        skipped_count = 0
        try:
            tar_bytes = base64.b64decode(tar_b64)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if member.isdir():
                        continue
                    rel = _safe_checkpoint_rel_path(member.name)
                    if rel.startswith(".science_agent_state/") or rel.startswith("input_images/"):
                        continue
                    if not member.isfile():
                        continue
                    extracted = tf.extractfile(member)
                    data = extracted.read() if extracted is not None else b""
                    expected = manifest.get(rel) or {}
                    expected_sha = str((expected or {}).get("sha256") or "")
                    if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
                        continue
                    local_rel = self._artifact_local_rel_for_sandbox_rel(rel)
                    if not _should_sync_local_tool_artifact(local_rel, source="checkpoint"):
                        skipped_count += 1
                        continue
                    entry = self._write_local_tool_artifact(
                        sandbox_rel_path=rel,
                        local_rel_path=local_rel,
                        data=data,
                        source="checkpoint",
                        list_for_model=True,
                    )
                    if entry:
                        entries.append(entry)
            if entries or skipped_count:
                self._checkpoint_trace(
                    "python_tool_artifacts_synced",
                    artifacts_count=len(entries),
                    artifacts_skipped_count=skipped_count,
                    artifacts_sync_mode=TOOL_OUTPUTS_SYNC_MODE,
                    artifact_dir=str(self._artifact_dir),
                    artifacts=[
                        {
                            "sandbox_rel_path": entry.get("sandbox_rel_path"),
                            "local_rel_path": entry.get("local_rel_path"),
                            "size": entry.get("size"),
                        }
                        for entry in entries[:TOOL_OUTPUTS_RETURN_MAX_FILES or 20]
                    ],
                )
        except Exception as exc:
            self._checkpoint_trace(
                "python_tool_artifact_sync_failed",
                artifact_dir=str(self._artifact_dir),
                error=repr(exc),
            )
        return entries

    def _saved_artifacts_note(self) -> str:
        if not self._last_artifact_entries or TOOL_OUTPUTS_RETURN_MAX_FILES <= 0:
            return ""
        entries = self._last_artifact_entries[:TOOL_OUTPUTS_RETURN_MAX_FILES]
        lines = []
        for entry in entries:
            sandbox_rel = str(entry.get("sandbox_rel_path") or "")
            local_path = str(entry.get("local_path") or "")
            if sandbox_rel:
                lines.append(f"- sandbox: `{sandbox_rel}`; local: `{local_path}`")
            elif local_path:
                lines.append(f"- local: `{local_path}`")
        if len(self._last_artifact_entries) > len(entries):
            lines.append(
                f"- ... {len(self._last_artifact_entries) - len(entries)} more file(s) saved; "
                "see tool_outputs_manifest.jsonl"
            )
        return "saved_artifacts:\n" + "\n".join(lines) if lines else ""

    def _checkpoint_meta_restore_failed(self, meta: Optional[dict[str, Any]] = None) -> bool:
        meta = self._checkpoint_last_meta if meta is None else meta
        if not isinstance(meta, dict):
            return True
        restore_meta = meta.get("restore_meta") or {}
        if not isinstance(restore_meta, dict):
            restore_meta = {}
        file_restore_failed = restore_meta.get("file_restore_failed")
        if file_restore_failed is not None:
            return bool(file_restore_failed)
        restore_file_errors = restore_meta.get("file_errors") or []
        if not isinstance(restore_file_errors, list):
            restore_file_errors = [str(restore_file_errors)]
        restore_meta_errors = restore_meta.get("meta_errors") or []
        if not isinstance(restore_meta_errors, list):
            restore_meta_errors = [str(restore_meta_errors)]
        return (not bool(meta.get("restore_ok", True))) and bool(restore_file_errors or restore_meta_errors)

    def _checkpoint_update_from_response(self, response_data: dict) -> bool:
        files = response_data.get("files") or {}
        if not isinstance(files, dict):
            files = {}
        meta_b64 = files.get(PYTHON_CHECKPOINT_META_FILE)
        meta: dict[str, Any] = {}
        if isinstance(meta_b64, str) and meta_b64:
            try:
                meta = _safe_json_loads(base64.b64decode(meta_b64).decode("utf-8"))
            except Exception as exc:
                meta = {"errors": [f"meta_decode: {exc!r}"]}

        next_checkpoint: dict[str, str] = {}
        for filename in PYTHON_CHECKPOINT_FETCH_FILES:
            value = files.get(filename)
            if isinstance(value, str) and value:
                next_checkpoint[filename] = value

        missing_files = sorted(PYTHON_CHECKPOINT_REQUIRED_FILES - set(next_checkpoint))
        missing_file_checkpoint_files = sorted(
            {PYTHON_CHECKPOINT_FILES_TAR, PYTHON_CHECKPOINT_META_FILE} - set(next_checkpoint)
        )
        missing_state_checkpoint_files = sorted(
            {PYTHON_CHECKPOINT_STATE_FILE} - set(next_checkpoint)
        )
        state_ok = bool(meta.get("state_ok")) and PYTHON_CHECKPOINT_STATE_FILE in next_checkpoint
        files_ok = bool(meta.get("files_ok")) and PYTHON_CHECKPOINT_FILES_TAR in next_checkpoint
        manifest_ok = (
            bool(meta.get("manifest_ok"))
            and isinstance(meta.get("file_manifest"), dict)
            and PYTHON_CHECKPOINT_META_FILE in next_checkpoint
        )
        restore_meta = meta.get("restore_meta") or {}
        if not isinstance(restore_meta, dict):
            restore_meta = {}
        restore_ok = bool(meta.get("restore_ok", True))
        errors = meta.get("errors") or []
        if not isinstance(errors, list):
            errors = [str(errors)]
        restore_errors = restore_meta.get("errors") or []
        if not isinstance(restore_errors, list):
            restore_errors = [str(restore_errors)]
        restore_file_errors = restore_meta.get("file_errors") or []
        if not isinstance(restore_file_errors, list):
            restore_file_errors = [str(restore_file_errors)]
        restore_state_errors = restore_meta.get("state_errors") or []
        if not isinstance(restore_state_errors, list):
            restore_state_errors = [str(restore_state_errors)]
        restore_module_errors = restore_meta.get("module_errors") or []
        if not isinstance(restore_module_errors, list):
            restore_module_errors = [str(restore_module_errors)]
        restore_meta_errors = restore_meta.get("meta_errors") or []
        if not isinstance(restore_meta_errors, list):
            restore_meta_errors = [str(restore_meta_errors)]
        file_restore_failed = bool(restore_meta.get("file_restore_failed")) or (
            not restore_ok and bool(restore_errors or restore_file_errors or restore_meta_errors)
        )
        state_restore_failed = bool(restore_meta.get("state_restore_failed")) or bool(
            restore_state_errors or restore_module_errors
        )
        self._checkpoint_last_restore_failed = file_restore_failed
        self._checkpoint_last_state_restore_failed = state_restore_failed
        file_save_verified = meta.get("file_save_verified")
        file_verify_errors = meta.get("file_verify_errors") or []
        if not isinstance(file_verify_errors, list):
            file_verify_errors = [str(file_verify_errors)]
        current_file_diag = meta.get("current_cell_file_diagnostics") or {}
        if not isinstance(current_file_diag, dict):
            current_file_diag = {"invalid": str(current_file_diag)[:200]}
        file_guard_ok = bool(meta.get("file_guard_ok", True))
        file_guard_errors = meta.get("file_guard_errors") or []
        if not isinstance(file_guard_errors, list):
            file_guard_errors = [str(file_guard_errors)]
        zero_binary_restored_files = meta.get("zero_binary_restored_files") or []
        if not isinstance(zero_binary_restored_files, list):
            zero_binary_restored_files = [str(zero_binary_restored_files)]
        zero_binary_dropped_files = meta.get("zero_binary_dropped_files") or []
        if not isinstance(zero_binary_dropped_files, list):
            zero_binary_dropped_files = [str(zero_binary_dropped_files)]
        invalid_pdf_restored_files = meta.get("invalid_pdf_restored_files") or []
        if not isinstance(invalid_pdf_restored_files, list):
            invalid_pdf_restored_files = [str(invalid_pdf_restored_files)]
        invalid_pdf_dropped_files = meta.get("invalid_pdf_dropped_files") or []
        if not isinstance(invalid_pdf_dropped_files, list):
            invalid_pdf_dropped_files = [str(invalid_pdf_dropped_files)]
        checkpoint_save_attempts = meta.get("checkpoint_save_attempts")
        checkpoint_save_retry_reasons = meta.get("checkpoint_save_retry_reasons") or []
        if not isinstance(checkpoint_save_retry_reasons, list):
            checkpoint_save_retry_reasons = [str(checkpoint_save_retry_reasons)]
        payload_verify_errors: list[str] = []
        payload_verify_summary: dict[str, Any] = {}
        if files_ok and manifest_ok:
            payload_verify_errors, payload_verify_summary = _verify_checkpoint_tar_payload(files, meta)
        file_checkpoint_ok = (
            files_ok
            and manifest_ok
            and file_save_verified is not False
            and not file_verify_errors
            and not payload_verify_errors
            and not file_restore_failed
            and file_guard_ok
        )
        state_checkpoint_ok = bool(state_ok and not state_restore_failed)
        state_degraded_reasons = []
        if not state_ok:
            state_degraded_reasons.append("state_save_failed")
        if state_restore_failed:
            state_degraded_reasons.append("state_restore_failed")
        if missing_state_checkpoint_files:
            state_degraded_reasons.append("missing_state_checkpoint_files")
        state_error_details = [
            str(item)
            for item in errors
            if str(item).startswith("state_") or "state checkpoint" in str(item)
        ]
        if state_error_details:
            state_degraded_reasons.append("state_save_errors")
        reject_reasons = []
        if missing_file_checkpoint_files:
            reject_reasons.append("missing_checkpoint_files")
        if not files_ok:
            reject_reasons.append("files_save_failed")
        if not manifest_ok:
            reject_reasons.append("manifest_save_failed")
        if file_restore_failed:
            reject_reasons.append("restore_failed")
            reject_reasons.append("files_restore_failed")
        if file_save_verified is False:
            reject_reasons.append("file_save_verify_failed")
        if file_verify_errors:
            reject_reasons.append("file_save_verify_errors")
        if payload_verify_errors:
            reject_reasons.append("checkpoint_payload_verify_failed")
        if not file_guard_ok:
            reject_reasons.append("file_guard_failed")
        file_error_details = [
            str(item)
            for item in errors
            if not (str(item).startswith("state_") or "state checkpoint" in str(item))
        ]
        if file_error_details and not file_checkpoint_ok:
            reject_reasons.append("save_errors")
        checkpoint_update_failed_category = _checkpoint_update_failed_category(
            reject_reasons,
            errors,
            file_restore_failed=file_restore_failed,
            state_restore_failed=state_restore_failed,
        )

        if file_checkpoint_ok and not reject_reasons:
            previous_version = self._checkpoint_version
            self._checkpoint_files = next_checkpoint
            self._checkpoint_version += 1
            self._checkpoint_last_meta = meta
            self._checkpoint_degraded_reason = None
            self._jupyter_cells.clear()
            self._checkpoint_recovery_cells.clear()
            artifact_entries = self._sync_checkpoint_artifacts_from_files(next_checkpoint, meta)
            self._last_artifact_entries.extend(artifact_entries)
            self._checkpoint_trace(
                "python_checkpoint_updated",
                previous_checkpoint_version=previous_version,
                checkpoint_accepted=True,
                file_checkpoint_ok=file_checkpoint_ok,
                state_checkpoint_ok=state_checkpoint_ok,
                state_degraded_reasons=state_degraded_reasons,
                state_bytes=meta.get("state_bytes"),
                files_bytes=meta.get("files_bytes"),
                files_ok=files_ok,
                manifest_ok=manifest_ok,
                restore_ok=restore_ok,
                file_restore_failed=file_restore_failed,
                state_restore_failed=state_restore_failed,
                files_count=meta.get("files_count"),
                file_save_verified=file_save_verified,
                file_manifest_top=meta.get("file_manifest_top") or payload_verify_summary.get("file_manifest_top"),
                largest_file_bytes=meta.get("largest_file_bytes") or payload_verify_summary.get("largest_file_bytes"),
                current_cell_file_diagnostics=current_file_diag,
                zero_byte_count=current_file_diag.get("zero_byte_count"),
                zero_byte_binary_like_count=current_file_diag.get("zero_byte_binary_like_count"),
                pre_nonzero_now_zero=current_file_diag.get("pre_nonzero_now_zero"),
                checkpoint_previous_nonzero_now_zero=current_file_diag.get("checkpoint_previous_nonzero_now_zero"),
                file_guard_ok=file_guard_ok,
                file_guard_errors=file_guard_errors,
                zero_binary_restored_files=zero_binary_restored_files,
                zero_binary_dropped_files=zero_binary_dropped_files,
                invalid_pdf_restored_files=invalid_pdf_restored_files,
                invalid_pdf_dropped_files=invalid_pdf_dropped_files,
                skipped_globals_count=len(meta.get("skipped_globals") or []),
                skipped_files_count=meta.get("skipped_files_count"),
                checkpoint_save_attempts=checkpoint_save_attempts,
                checkpoint_save_retry_reasons=checkpoint_save_retry_reasons,
                errors_count=len(errors),
                restore_errors_count=len(restore_errors),
                fetched_files=sorted(next_checkpoint),
                missing_state_checkpoint_files=missing_state_checkpoint_files,
                artifacts_synced_count=len(artifact_entries),
            )
            if not state_checkpoint_ok:
                self._checkpoint_trace(
                    "python_checkpoint_partial_accepted",
                    previous_checkpoint_version=previous_version,
                    checkpoint_accepted=True,
                    file_checkpoint_ok=file_checkpoint_ok,
                    state_checkpoint_ok=state_checkpoint_ok,
                    state_degraded_reasons=state_degraded_reasons,
                    state_bytes=meta.get("state_bytes"),
                    files_bytes=meta.get("files_bytes"),
                    files_count=meta.get("files_count"),
                    skipped_globals_count=len(meta.get("skipped_globals") or []),
                    errors=errors,
                    restore_errors=restore_errors,
                    missing_state_checkpoint_files=missing_state_checkpoint_files,
                )
            if state_restore_failed:
                self._checkpoint_trace(
                    "python_state_restore_failed_continue",
                    checkpoint_accepted=True,
                    file_checkpoint_ok=file_checkpoint_ok,
                    state_checkpoint_ok=state_checkpoint_ok,
                    restore_errors=restore_errors,
                    restore_state_errors=restore_state_errors,
                    restore_module_errors=restore_module_errors,
                )
            return True

        self._checkpoint_last_meta = meta
        self._checkpoint_degraded_reason = ",".join(reject_reasons)
        if file_restore_failed:
            self._checkpoint_trace(
                "python_files_restore_failed",
                restore_errors=restore_errors,
                restore_file_errors=restore_file_errors,
                restore_meta_errors=restore_meta_errors,
                file_manifest_top=meta.get("file_manifest_top") or payload_verify_summary.get("file_manifest_top"),
            )
        self._checkpoint_trace(
            "python_checkpoint_update_failed",
            reject_reason=self._checkpoint_degraded_reason,
            checkpoint_update_failed_category=checkpoint_update_failed_category,
            checkpoint_accepted=False,
            file_checkpoint_ok=file_checkpoint_ok,
            state_checkpoint_ok=state_checkpoint_ok,
            state_ok=state_ok,
            files_ok=files_ok,
            manifest_ok=manifest_ok,
            restore_ok=restore_ok,
            file_restore_failed=file_restore_failed,
            state_restore_failed=state_restore_failed,
            missing_files=missing_files,
            missing_file_checkpoint_files=missing_file_checkpoint_files,
            missing_state_checkpoint_files=missing_state_checkpoint_files,
            fetched_files=sorted(next_checkpoint),
            errors=errors,
            restore_errors=restore_errors,
            restore_file_errors=restore_file_errors,
            restore_state_errors=restore_state_errors,
            restore_module_errors=restore_module_errors,
            restore_meta_errors=restore_meta_errors,
            file_verify_errors=file_verify_errors,
            payload_verify_errors=payload_verify_errors,
            checkpoint_save_attempts=checkpoint_save_attempts,
            checkpoint_save_retry_reasons=checkpoint_save_retry_reasons,
            file_guard_ok=file_guard_ok,
            file_guard_errors=file_guard_errors,
            file_manifest_top=meta.get("file_manifest_top") or payload_verify_summary.get("file_manifest_top"),
            largest_file_bytes=meta.get("largest_file_bytes") or payload_verify_summary.get("largest_file_bytes"),
            current_cell_file_diagnostics=current_file_diag,
            zero_byte_count=current_file_diag.get("zero_byte_count"),
            zero_byte_binary_like_count=current_file_diag.get("zero_byte_binary_like_count"),
            pre_nonzero_now_zero=current_file_diag.get("pre_nonzero_now_zero"),
            checkpoint_previous_nonzero_now_zero=current_file_diag.get("checkpoint_previous_nonzero_now_zero"),
            zero_binary_restored_files=zero_binary_restored_files,
            zero_binary_dropped_files=zero_binary_dropped_files,
            invalid_pdf_restored_files=invalid_pdf_restored_files,
            invalid_pdf_dropped_files=invalid_pdf_dropped_files,
        )
        return False

    def _checkpoint_cells(
        self,
        code: str,
        *,
        setup_cell: str = "",
        include_recovery: bool = False,
    ) -> tuple[list[str], int]:
        restore_cell = _build_checkpoint_restore_cell(
            setup_cell,
            legacy_path_compat=python_checkpoint_legacy_path_compat_enabled(),
        )
        pre_current_cell = _build_checkpoint_pre_current_cell()
        save_cell = _build_checkpoint_save_cell()
        recovery_cells = self._checkpoint_recovery_cells if include_recovery else self._jupyter_cells
        if python_checkpoint_legacy_path_compat_enabled():
            recovery_cells = [normalize_checkpoint_legacy_code_paths(cell) for cell in recovery_cells]
        cells = [restore_cell, *recovery_cells, pre_current_cell, code, save_cell]
        current_cell_index = len(cells) - 2
        return cells, current_cell_index

    def _append_checkpoint_tail(self, code: str, **trace_fields: Any) -> None:
        if PYTHON_CHECKPOINT_MAX_TAIL_CELLS <= 0:
            self._checkpoint_trace(
                "python_checkpoint_fallback_tail_disabled",
                max_tail_cells=PYTHON_CHECKPOINT_MAX_TAIL_CELLS,
                **trace_fields,
            )
            return
        if len(self._checkpoint_recovery_cells) >= PYTHON_CHECKPOINT_MAX_TAIL_CELLS:
            removed = self._checkpoint_recovery_cells.pop(0)
            self._checkpoint_trace(
                "python_checkpoint_recovery_tail_trimmed",
                tail_cells=len(self._checkpoint_recovery_cells),
                max_tail_cells=PYTHON_CHECKPOINT_MAX_TAIL_CELLS,
                removed_code_hash=_sha256_text(removed),
                **trace_fields,
            )
        self._checkpoint_recovery_cells.append(code)
        self._checkpoint_trace(
            "python_checkpoint_recovery_tail_append",
            tail_cells=len(self._checkpoint_recovery_cells),
            **trace_fields,
        )

    def _untrusted_restore_error(self) -> str:
        errors = []
        restore_meta = (self._checkpoint_last_meta or {}).get("restore_meta") or {}
        if isinstance(restore_meta, dict):
            raw_errors = restore_meta.get("errors") or []
            if isinstance(raw_errors, list):
                errors.extend(str(item) for item in raw_errors[:3])
            elif raw_errors:
                errors.append(str(raw_errors))
        detail = "; ".join(errors) if errors else self._checkpoint_degraded_reason or "unknown restore failure"
        return (
            "[Python Interpreter Error]: Checkpoint restore failed before the current "
            f"cell could be trusted, and no recovery history is available. {detail}"
        )

    def _execute_checkpoint_jupyter_code(
        self,
        code: str,
        endpoint: str,
        timeout: int,
        *,
        update_history: bool,
        setup_cell: str = "",
        files: Optional[dict[str, str]] = None,
        fetch_files: Optional[list[str]] = None,
    ) -> str:
        del update_history
        self._last_artifact_entries = []
        original_code = code
        code = normalize_checkpoint_legacy_code_paths(code)
        if code != original_code:
            self._checkpoint_trace(
                "python_checkpoint_legacy_path_code_normalized",
                original_code_hash=_sha256_text(original_code),
                normalized_code_hash=_sha256_text(code),
            )
        use_recovery = bool(self._checkpoint_degraded_reason)
        cells, current_cell_index = self._checkpoint_cells(
            code,
            setup_cell=setup_cell,
            include_recovery=use_recovery,
        )
        request_files = self._checkpoint_payload_files(files, include_checkpoint=not use_recovery)
        request_fetch_files = _merge_fetch_files(fetch_files, PYTHON_CHECKPOINT_FETCH_FILES)
        self._checkpoint_trace(
            "python_checkpoint_request_start",
            endpoint=endpoint,
            cells_count=len(cells),
            current_cell_index=current_cell_index,
            code_hash=_sha256_text(code),
            code_chars=len(code),
            checkpoint_files=sorted(self._checkpoint_files),
            recovery_tail_cells=len(self._checkpoint_recovery_cells),
            recovery_mode=use_recovery,
            upload_files_count=len(request_files),
            fetch_files=request_fetch_files,
        )
        response_data = post_jupyter_request(
            cells,
            timeout,
            endpoint,
            files=request_files,
            fetch_files=request_fetch_files,
        )
        result = format_jupyter_tool_output(response_data, current_cell_index, timeout)
        checkpoint_ok = self._checkpoint_update_from_response(response_data)
        restore_failed = self._checkpoint_last_restore_failed
        _drop_checkpoint_response_files(response_data)
        if restore_failed and not use_recovery:
            if self._checkpoint_recovery_cells:
                self._checkpoint_trace(
                    "python_checkpoint_fallback_replay",
                    endpoint=endpoint,
                    reason="restore_failed",
                    recovery_tail_cells=len(self._checkpoint_recovery_cells),
                    code_hash=_sha256_text(code),
                )
                cells, current_cell_index = self._checkpoint_cells(
                    code,
                    setup_cell=setup_cell,
                    include_recovery=True,
                )
                request_files = self._checkpoint_payload_files(files, include_checkpoint=False)
                response_data = post_jupyter_request(
                    cells,
                    timeout,
                    endpoint,
                    files=request_files,
                    fetch_files=request_fetch_files,
                )
                result = format_jupyter_tool_output(response_data, current_cell_index, timeout)
                checkpoint_ok = self._checkpoint_update_from_response(response_data)
                restore_failed = self._checkpoint_last_restore_failed
                _drop_checkpoint_response_files(response_data)
            else:
                self._checkpoint_trace(
                    "python_checkpoint_untrusted_restore",
                    endpoint=endpoint,
                    reason="restore_failed_no_recovery_tail",
                    code_hash=_sha256_text(code),
                )
                return self._untrusted_restore_error()
        current_ok = jupyter_current_cell_succeeded(
            response_data,
            current_cell_index,
            timeout,
            result,
        )
        if current_ok and not checkpoint_ok:
            self._append_checkpoint_tail(code)
        file_error_diag = _extract_file_error_diagnostics(result)
        file_error_diag = _refine_file_error_diagnostics(file_error_diag, self._checkpoint_last_meta)
        current_file_diag = (self._checkpoint_last_meta or {}).get("current_cell_file_diagnostics") or {}
        if not isinstance(current_file_diag, dict):
            current_file_diag = {"invalid": str(current_file_diag)[:200]}
        guard_warning = _checkpoint_file_guard_warning(self._checkpoint_last_meta or {})
        if guard_warning and guard_warning not in result:
            result = f"{result}\n{guard_warning}" if result else guard_warning
        artifact_note = self._saved_artifacts_note()
        if artifact_note:
            result = f"{result}\n\n{artifact_note}" if result else artifact_note
        if file_error_diag:
            self._checkpoint_trace(
                "python_file_error_diagnostic",
                endpoint=endpoint,
                code_hash=_sha256_text(code),
                checkpoint_ok=checkpoint_ok,
                restore_failed=restore_failed,
                current_cell_ok=current_ok,
                file_error=file_error_diag,
                file_error_category=file_error_diag.get("diagnostic_category"),
                checkpoint_infra_suspect=file_error_diag.get("checkpoint_infra_suspect"),
                current_cell_file_diagnostics=current_file_diag,
                file_manifest_top=(self._checkpoint_last_meta or {}).get("file_manifest_top"),
                largest_file_bytes=(self._checkpoint_last_meta or {}).get("largest_file_bytes"),
            )
        self._checkpoint_trace(
            "python_checkpoint_request_end",
            endpoint=endpoint,
            checkpoint_ok=checkpoint_ok,
            restore_failed=restore_failed,
            current_cell_ok=current_ok,
            file_error=file_error_diag,
            file_error_category=file_error_diag.get("diagnostic_category") if file_error_diag else None,
            checkpoint_infra_suspect=file_error_diag.get("checkpoint_infra_suspect") if file_error_diag else False,
            zero_byte_count=current_file_diag.get("zero_byte_count"),
            zero_byte_binary_like_count=current_file_diag.get("zero_byte_binary_like_count"),
            pre_nonzero_now_zero=current_file_diag.get("pre_nonzero_now_zero"),
            checkpoint_previous_nonzero_now_zero=current_file_diag.get("checkpoint_previous_nonzero_now_zero"),
            zero_binary_restored_files=(self._checkpoint_last_meta or {}).get("zero_binary_restored_files"),
            zero_binary_dropped_files=(self._checkpoint_last_meta or {}).get("zero_binary_dropped_files"),
            invalid_pdf_restored_files=(self._checkpoint_last_meta or {}).get("invalid_pdf_restored_files"),
            invalid_pdf_dropped_files=(self._checkpoint_last_meta or {}).get("invalid_pdf_dropped_files"),
            output_chars=len(result),
        )
        return _truncate_python_tool_return(
            result,
            code_hash=_sha256_text(code),
            checkpoint_ok=checkpoint_ok,
        )

    def _normalized_sandbox_endpoints(self) -> list[str]:
        return [endpoint for endpoint in (normalize_sandbox_fusion_endpoint(x) for x in SANDBOX_FUSION_ENDPOINTS) if endpoint]

    def _choose_sandbox_endpoint(self) -> str:
        endpoints = self._normalized_sandbox_endpoints()
        if not endpoints:
            return ""
        if self._sandbox_endpoint in endpoints:
            return str(self._sandbox_endpoint)
        self._sandbox_endpoint = random.choice(endpoints)
        return self._sandbox_endpoint

    def _rotate_sandbox_endpoint(self, failed_endpoint: str) -> None:
        endpoints = self._normalized_sandbox_endpoints()
        if not endpoints:
            self._sandbox_endpoint = None
            return
        alternatives = [endpoint for endpoint in endpoints if endpoint != failed_endpoint]
        self._sandbox_endpoint = random.choice(alternatives or endpoints)

    @property
    def jupyter_history_length(self) -> int:
        return len(self._jupyter_cells)
    
    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'

            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def observation(self, tool: dict, tool_dict: dict, tool_results, empty_mode: bool=False, readpage: bool=False, max_observation_length: int=None, tokenizer=None):
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results
    
    @property
    def function(self) -> dict:  
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def _execute_jupyter_code(
        self,
        code: str,
        endpoint: str,
        timeout: int,
        *,
        update_history: bool,
        files: Optional[dict[str, str]] = None,
        fetch_files: Optional[list[str]] = None,
    ) -> str:
        if self._checkpoint_enabled():
            setup_cell = PYTHON_SETUP_CELL if python_relative_path_policy_enabled() else ""
            return self._execute_checkpoint_jupyter_code(
                code,
                endpoint,
                timeout,
                update_history=update_history,
                setup_cell=setup_cell,
                files=files,
                fetch_files=fetch_files,
            )
        setup_cells = [PYTHON_SETUP_CELL] if python_relative_path_policy_enabled() else []
        cells = [*setup_cells, *self._jupyter_cells, code]
        current_cell_index = len(cells) - 1
        if files or fetch_files:
            response_data = post_jupyter_request(
                cells,
                timeout,
                endpoint,
                files=files,
                fetch_files=fetch_files,
            )
        else:
            response_data = post_jupyter_request(cells, timeout, endpoint)
        result = format_jupyter_tool_output(response_data, current_cell_index, timeout)
        if update_history and jupyter_current_cell_succeeded(
            response_data,
            current_cell_index,
            timeout,
            result,
        ):
            self._jupyter_cells.append(code)
        return _truncate_python_tool_return(result, code_hash=_sha256_text(code))

    def call(self, params, files= None, timeout = None, **kwargs) -> str:
        try:
            code = normalize_python_code_input(params)
            if python_request_blocked(params, code):
                return blocked_python_message()
            path_policy_error = python_path_policy_error(code)
            if path_policy_error:
                return path_policy_error
            if not code.strip():
                return '[Python Interpreter Error]: Empty code.'
            if PYTHON_SESSION_BACKEND == SANDBOX_GATEWAY_BACKEND:
                run = shared_sandbox_run()
                self._agent_sandbox = run
                try:
                    result = run.run_python(
                        code,
                        timeout=PYTHON_TOOL_TIMEOUT_SEC if timeout is None else int(timeout),
                    )
                    return _truncate_python_tool_return(result, code_hash=_sha256_text(code)) or PYTHON_TOOL_NO_OUTPUT_MESSAGE
                except SandboxGatewayError as exc:
                    return f"[Python Interpreter Error]: {exc}"
            last_error = None
            timeout = PYTHON_TOOL_TIMEOUT_SEC if timeout is None else int(timeout)
            max_attempts = _effective_python_tool_max_attempts(timeout)
            if not SANDBOX_FUSION_ENDPOINTS:
                return missing_sandbox_fusion_endpoint_message()
            if max_attempts != PYTHON_TOOL_MAX_RETRIES:
                self._checkpoint_trace(
                    "python_retry_decision",
                    reason="total_timeout_budget",
                    configured_max_attempts=PYTHON_TOOL_MAX_RETRIES,
                    effective_max_attempts=max_attempts,
                    per_attempt_timeout_sec=timeout,
                    total_timeout_sec=PYTHON_TOOL_TOTAL_TIMEOUT_SEC,
                    retry=True,
                )
            for attempt in range(max_attempts):
                endpoint = self._choose_sandbox_endpoint()
                try:
                    print(f"Attempt {attempt + 1}/{max_attempts} using endpoint: {endpoint!r}")

                    result = self._execute_jupyter_code(
                        code,
                        endpoint,
                        timeout,
                        update_history=True,
                    )
                    if result == blocked_python_message():
                        print("[Python] Code Result blocked by evaluation URL/keyword guard")
                        return result
                    print("[Python] Jupyter Result", result)
                    print('SUCCESS RUNNING TOOL')
                    return result

                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint!r}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    self._rotate_sandbox_endpoint(endpoint)
                    self._checkpoint_trace(
                        "python_retry_decision",
                        reason="transport_timeout",
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        effective_max_attempts=max_attempts,
                        per_attempt_timeout_sec=timeout,
                        total_timeout_sec=PYTHON_TOOL_TOTAL_TIMEOUT_SEC,
                        retry=attempt < max_attempts - 1,
                        error=str(e),
                    )
                    if attempt == max_attempts - 1:
                        return last_error
                    continue
                
                except (RequestException, RuntimeError, ValueError) as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint!r}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    self._rotate_sandbox_endpoint(endpoint)
                    self._checkpoint_trace(
                        "python_retry_decision",
                        reason="transport_or_runtime_error",
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        effective_max_attempts=max_attempts,
                        per_attempt_timeout_sec=timeout,
                        total_timeout_sec=PYTHON_TOOL_TOTAL_TIMEOUT_SEC,
                        retry=attempt < max_attempts - 1,
                        error=str(e),
                    )
                    if attempt == max_attempts - 1:
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = None, **kwargs) -> tuple:
        """Test a specific endpoint directly"""
        try:
            if type(params) is str:
                params = json5.loads(params)
            code = normalize_python_code_input(params)
        except Exception:
            code = normalize_python_code_input(extract_code(params))

        if python_request_blocked(params, code):
            return False, blocked_python_message(), None

        path_policy_error = python_path_policy_error(code)
        if path_policy_error:
            return False, path_policy_error, None

        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.'

        try:
            endpoint = normalize_sandbox_fusion_endpoint(endpoint)
            if not endpoint:
                return False, missing_sandbox_fusion_endpoint_message(), None
            timeout = PYTHON_TOOL_TIMEOUT_SEC if timeout is None else int(timeout)
            start_time = time.time()
            result = self._execute_jupyter_code(
                code,
                endpoint,
                timeout,
                update_history=True,
            )
            execution_time = time.time() - start_time
            if result == blocked_python_message():
                return False, result, None
            return True, result, execution_time

        except Timeout as e:
            return False, f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint!r}.', None
        except (RequestException, RuntimeError, ValueError) as e:
            return False, f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint!r}', None
