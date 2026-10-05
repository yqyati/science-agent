import base64
import hashlib
from io import BytesIO
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Optional

from qwen_agent.tools.base import register_tool

from multimodal_input import resolve_item_image_paths, resolve_item_image_urls
from tool_path_policy import find_relative_path_policy_violations, format_path_policy_error
from tool_python import (
    PYTHON_CHECKPOINT_FETCH_FILES,
    PYTHON_TOOL_MAX_RETRIES,
    PYTHON_TOOL_NO_OUTPUT_MESSAGE,
    PYTHON_TOOL_TIMEOUT_SEC,
    SANDBOX_FUSION_ENDPOINTS,
    TOOL_OUTPUTS_DIR,
    PythonInterpreter,
    _coerce_current_jupyter_cell,
    blocked_python_message,
    _drop_checkpoint_response_files,
    format_jupyter_tool_output,
    jupyter_current_cell_succeeded,
    missing_sandbox_fusion_endpoint_message,
    normalize_checkpoint_legacy_code_paths,
    normalize_python_code_input,
    post_jupyter_request,
    python_relative_path_policy_enabled,
    python_request_blocked,
    _merge_fetch_files,
    _sha256_text,
)


DATA_IMAGE_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
LEGACY_VISUAL_INPUT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])/mnt/data/input_images"
    r"(?=/|[^A-Za-z0-9_.-]|$)"
)
VISUAL_SETUP_CELL = (
    "import os\n"
    "os.makedirs('input_images', exist_ok=True)\n"
    f"os.makedirs({TOOL_OUTPUTS_DIR!r}, exist_ok=True)"
)


def _normalize_visual_input_image_paths(code: str) -> str:
    """Map only the historical visual-input mount to its staged relative path."""

    return LEGACY_VISUAL_INPUT_PATH_RE.sub("input_images", str(code or ""))


def _mime_to_extension(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    guessed = mimetypes.guess_extension(mime_type) or ".png"
    return ".jpg" if guessed == ".jpe" else guessed


def _guess_image_mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_relative_component(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    cleaned = cleaned.strip("._/")
    return cleaned or default


def _advanced_visual_session_dir() -> str:
    raw = str(os.getenv("ADVANCED_VISUAL_SESSION_DIR") or "input_images").strip().strip("/")
    parts = [
        _safe_relative_component(part, "")
        for part in raw.split("/")
        if _safe_relative_component(part, "")
    ]
    return "/".join(parts) or "input_images"


def _advanced_visual_name_prefix() -> str:
    return _safe_relative_component(os.getenv("ADVANCED_VISUAL_SESSION_NAME_PREFIX") or "", "advanced_visual")


def _image_content_part(mime_type: str, image_b64: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_b64.strip()}",
        },
    }


def _extract_data_image(image_url: str) -> tuple[str, str] | None:
    match = DATA_IMAGE_RE.match(str(image_url).strip())
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _display_item_data(display_item: Any) -> dict[str, Any]:
    if not isinstance(display_item, dict):
        return {}
    data = display_item.get("data")
    if isinstance(data, dict):
        return data
    return display_item


def _extract_display_images(cell: dict[str, Any]) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    display_items = cell.get("display") or []
    if not isinstance(display_items, list):
        display_items = [display_items]
    for display_item in display_items:
        data = _display_item_data(display_item)
        for mime_type in ("image/png", "image/jpeg"):
            image_b64 = data.get(mime_type)
            if isinstance(image_b64, str) and image_b64.strip():
                images.append((mime_type, image_b64.strip()))
                break
    return images


def _visual_response_mode() -> str:
    mode = str(os.getenv("VISUAL_TOOL_RESPONSE_MODE") or "auto").strip().lower()
    if mode not in {"auto", "multimodal", "text"}:
        return "auto"
    return mode


def _visual_max_return_images() -> int:
    try:
        return max(1, int(os.getenv("VISUAL_TOOL_MAX_RETURN_IMAGES", "1")))
    except ValueError:
        return 1


def _visual_max_image_edge() -> int:
    try:
        return max(1, int(os.getenv("VISUAL_TOOL_MAX_IMAGE_EDGE", "1600")))
    except ValueError:
        return 1600


def _decode_image_bytes(image_b64: str) -> bytes:
    return base64.b64decode(str(image_b64).strip())


def _image_digest(image_b64: str) -> str:
    try:
        image_bytes = _decode_image_bytes(image_b64)
    except Exception:
        image_bytes = str(image_b64).encode("utf-8", errors="replace")
    return hashlib.sha256(image_bytes).hexdigest()


def _resize_image_for_response(mime_type: str, image_b64: str) -> tuple[str, str, bool]:
    max_edge = _visual_max_image_edge()
    try:
        image_bytes = _decode_image_bytes(image_b64)
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if max(width, height) <= max_edge:
                return mime_type, image_b64, False

            resized = image.copy()
            resized.thumbnail((max_edge, max_edge))
            output = BytesIO()
            if mime_type == "image/jpeg":
                if resized.mode not in {"RGB", "L"}:
                    resized = resized.convert("RGB")
                resized.save(output, format="JPEG")
            else:
                resized.save(output, format="PNG")
                mime_type = "image/png"
            resized_b64 = base64.b64encode(output.getvalue()).decode("ascii")
            return mime_type, resized_b64, True
    except Exception:
        return mime_type, image_b64, False


@register_tool("VisualPythonInterpreter", allow_overwrite=True)
class VisualPythonInterpreter(PythonInterpreter):
    name = "VisualPythonInterpreter"
    description = (
        "Execute Python code in a stateful Jupyter-style notebook environment for "
        "quantitative visual processing. The model can already see the original "
        "problem images; do not use this tool just to look at an image. Use it "
        "only when code-based OCR, crop/zoom inspection, curve digitization, "
        "pixel/coordinate measurement, scale conversion, or geometric estimation "
        "would materially help. Current problem images are available inside the "
        "sandbox under `input_images/`. Start from these files; only open "
        "generated page, crop, or rendered files after creating them in this "
        f"notebook. Save generated files under `{TOOL_OUTPUTS_DIR}/`. Prefer "
        "matplotlib `plt.show()` or IPython display; do not use PIL "
        "`Image.show()`. Images displayed this way will be returned to you. "
        "Every call should print a "
        "`visual_measurement_summary:` with the useful OCR text or measured "
        "quantities, including crop/pixel coordinates, scale conversion, final "
        "values, and uncertainty; if nothing useful was measured, print "
        "`visual_measurement_summary: no useful visual measurement found`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "A Python code cell to execute for quantitative visual "
                    "processing. Use it for OCR, crop/zoom inspection, curve "
                    "digitization, pixel/coordinate measurement, scale conversion, "
                    "or geometric estimation. End by printing "
                    "`visual_measurement_summary:` with the image path processed, "
                    "crop or pixel coordinates, scale reference, conversion factor, "
                    "measured values or OCR text, and uncertainty. If no useful "
                    "measurement was found, print `visual_measurement_summary: no "
                    "useful visual measurement found`."
                ),
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._input_files: dict[str, str] = {}
        self._input_image_paths: list[str] = []
        self._artifact_dir: Optional[Path] = None
        self._artifact_counter = 0
        self._force_text_response = False
        self._returned_image_hashes: set[str] = set()
        self._input_image_hashes: dict[str, str] = {}

    def reset_jupyter_history(self) -> None:
        super().reset_jupyter_history()
        self._artifact_counter = 0
        self._force_text_response = False
        self._returned_image_hashes.clear()
        self._input_image_hashes.clear()

    @property
    def input_image_paths(self) -> list[str]:
        return list(self._input_image_paths)

    def set_text_response_fallback(self, enabled: bool) -> None:
        self._force_text_response = bool(enabled)

    def set_task_context(
        self,
        item: dict[str, Any],
        *,
        base_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> str:
        self._input_files = {}
        self._input_image_paths = []
        self._artifact_counter = 0
        self._force_text_response = False
        self._returned_image_hashes.clear()
        self._input_image_hashes.clear()
        self._artifact_dir = Path(artifact_dir) if artifact_dir else None

        image_entries: list[tuple[str, str, str]] = []
        try:
            image_paths = resolve_item_image_paths(item, base_dir=base_dir)
        except FileNotFoundError:
            image_paths = []

        for image_path_text in image_paths:
            image_path = Path(image_path_text)
            mime_type = _guess_image_mime(image_path)
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            image_entries.append((mime_type, image_b64, image_path.suffix or _mime_to_extension(mime_type)))

        if not image_entries:
            for image_url in resolve_item_image_urls(item, base_dir=base_dir):
                parsed = _extract_data_image(image_url)
                if parsed is None:
                    continue
                mime_type, image_b64 = parsed
                image_entries.append((mime_type, image_b64, _mime_to_extension(mime_type)))

        for index, (mime_type, image_b64, suffix) in enumerate(image_entries, start=1):
            ext = suffix if suffix.startswith(".") else f".{suffix}"
            sandbox_path = f"input_images/image_{index}{ext.lower()}"
            self._input_files[sandbox_path] = image_b64
            self._input_image_paths.append(sandbox_path)
            self._input_image_hashes[_image_digest(image_b64)] = sandbox_path

        if not self._input_image_paths:
            return ""

        paths = ", ".join(f"`{path}`" for path in self._input_image_paths)
        return (
            "For VisualPythonInterpreter only, the current problem image file(s) "
            f"are available inside the sandbox at: {paths}. Start from these "
            "relative paths with PIL/OpenCV; do not guess page or crop filenames "
            "unless you created them earlier in this notebook. If unsure, list "
            f"`input_images/` or `{TOOL_OUTPUTS_DIR}/` instead of "
            "guessing. Do not use /mnt/data or host paths. Use this tool only for "
            "quantitative visual processing, not to simply view the original "
            "image. Each call should produce a text `visual_measurement_summary:` "
            "with the useful OCR result, measured value, pixel/crop coordinates, "
            "scale conversion, and uncertainty; if none is found, say no useful "
            "visual measurement found."
        )

    def add_input_image_data_urls(self, images: list[dict[str, Any]] | list[str]) -> dict[str, Any]:
        paths: list[str] = []
        errors: list[str] = []
        for index, image in enumerate(images or [], start=1):
            if isinstance(image, str):
                data_url = image
                source_tool = ""
            elif isinstance(image, dict):
                data_url = str(image.get("data_url") or image.get("url") or "").strip()
                source_tool = str(image.get("source_tool") or image.get("tool") or "").strip()
            else:
                errors.append(f"image {index}: unsupported payload")
                continue
            parsed = _extract_data_image(data_url)
            if parsed is None:
                errors.append(f"image {index}: not a data image")
                continue
            mime_type, image_b64 = parsed
            try:
                image_bytes = _decode_image_bytes(image_b64)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"image {index}: invalid base64: {exc}")
                continue
            digest = hashlib.sha256(image_bytes).hexdigest()
            existing = self._input_image_hashes.get(digest)
            if existing:
                paths.append(existing)
                continue
            ext = _mime_to_extension(mime_type)
            directory = _advanced_visual_session_dir()
            prefix = _advanced_visual_name_prefix()
            next_index = len([path for path in self._input_image_paths if path.startswith(f"{directory}/{prefix}_")]) + 1
            sandbox_path = f"{directory}/{prefix}_{next_index:04d}{ext.lower()}"
            while sandbox_path in self._input_files:
                next_index += 1
                sandbox_path = f"{directory}/{prefix}_{next_index:04d}{ext.lower()}"
            self._input_files[sandbox_path] = image_b64
            self._input_image_paths.append(sandbox_path)
            self._input_image_hashes[digest] = sandbox_path
            paths.append(sandbox_path)
            if self._artifact_dir and _env_bool("ADVANCED_VISUAL_SESSION_SAVE_LOCAL_COPY", True):
                try:
                    local_dir = self._artifact_dir / "advanced_visual_inputs"
                    local_dir.mkdir(parents=True, exist_ok=True)
                    source_suffix = f"_{_safe_relative_component(source_tool, 'source')}" if source_tool else ""
                    (local_dir / f"{prefix}_{next_index:04d}{source_suffix}{ext.lower()}").write_bytes(image_bytes)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"image {index}: local copy warning: {exc}")

        hint = ""
        if paths:
            joined = ", ".join(f"`{path}`" for path in paths)
            hint = (
                "For VisualPythonInterpreter only, advanced visual evidence has "
                f"been added inside the sandbox at: {joined}. Use these relative "
                "paths for focused visual inspection, OCR-like checks, cropping, "
                "measurement, or figure analysis."
            )
        return {"paths": paths, "errors": errors, "hint": hint}

    def _execute_visual_code(
        self,
        code: str,
        endpoint: str,
        timeout: int,
        *,
        update_history: bool,
    ) -> tuple[str, list[tuple[str, str]], list[str]]:
        self._last_artifact_entries = []
        if self._checkpoint_enabled():
            original_code = code
            code = normalize_checkpoint_legacy_code_paths(code)
            if code != original_code:
                self._checkpoint_trace(
                    "python_checkpoint_legacy_path_code_normalized",
                    original_code_hash=_sha256_text(original_code),
                    normalized_code_hash=_sha256_text(code),
                    visual=True,
                )
            use_recovery = bool(self._checkpoint_degraded_reason)
            cells, current_cell_index = self._checkpoint_cells(
                code,
                setup_cell=VISUAL_SETUP_CELL,
                include_recovery=use_recovery,
            )
            request_files = self._checkpoint_payload_files(
                self._input_files,
                include_checkpoint=not use_recovery,
            )
            request_fetch_files = _merge_fetch_files(PYTHON_CHECKPOINT_FETCH_FILES)
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
                visual=True,
            )
            response_data = post_jupyter_request(
                cells,
                timeout,
                endpoint,
                files=request_files,
                fetch_files=request_fetch_files,
            )
            text_output = format_jupyter_tool_output(response_data, current_cell_index, timeout)
            cell = _coerce_current_jupyter_cell(response_data, current_cell_index)
            images = _extract_display_images(cell)
            artifact_paths = self._save_image_artifacts(images)
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
                        visual=True,
                    )
                    cells, current_cell_index = self._checkpoint_cells(
                        code,
                        setup_cell=VISUAL_SETUP_CELL,
                        include_recovery=True,
                    )
                    request_files = self._checkpoint_payload_files(
                        self._input_files,
                        include_checkpoint=False,
                    )
                    response_data = post_jupyter_request(
                        cells,
                        timeout,
                        endpoint,
                        files=request_files,
                        fetch_files=request_fetch_files,
                    )
                    text_output = format_jupyter_tool_output(response_data, current_cell_index, timeout)
                    cell = _coerce_current_jupyter_cell(response_data, current_cell_index)
                    images = _extract_display_images(cell)
                    artifact_paths = self._save_image_artifacts(images)
                    checkpoint_ok = self._checkpoint_update_from_response(response_data)
                    restore_failed = self._checkpoint_last_restore_failed
                    _drop_checkpoint_response_files(response_data)
                else:
                    self._checkpoint_trace(
                        "python_checkpoint_untrusted_restore",
                        endpoint=endpoint,
                        reason="restore_failed_no_recovery_tail",
                        code_hash=_sha256_text(code),
                        visual=True,
                    )
                    return self._untrusted_restore_error(), [], []
            current_ok = jupyter_current_cell_succeeded(
                response_data,
                current_cell_index,
                timeout,
                text_output,
            )
            if current_ok and not checkpoint_ok:
                self._append_checkpoint_tail(code, visual=True)
            self._checkpoint_trace(
                "python_checkpoint_request_end",
                endpoint=endpoint,
                checkpoint_ok=checkpoint_ok,
                restore_failed=restore_failed,
                current_cell_ok=current_ok,
                output_chars=len(text_output),
                display_images=len(images),
                visual=True,
            )
            return text_output, images, artifact_paths

        cells = [VISUAL_SETUP_CELL, *self._jupyter_cells, code]
        current_cell_index = len(cells) - 1
        response_data = post_jupyter_request(
            cells,
            timeout,
            endpoint,
            files=self._input_files,
        )
        text_output = format_jupyter_tool_output(response_data, current_cell_index, timeout)
        cell = _coerce_current_jupyter_cell(response_data, current_cell_index)
        images = _extract_display_images(cell)
        artifact_paths = self._save_image_artifacts(images)
        if update_history and jupyter_current_cell_succeeded(
            response_data,
            current_cell_index,
            timeout,
            text_output,
        ):
            self._jupyter_cells.append(code)
        return text_output, images, artifact_paths

    def _save_image_artifacts(self, images: list[tuple[str, str]]) -> list[str]:
        if not images or self._artifact_dir is None:
            return []

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_counter += 1
        paths: list[str] = []
        for index, (mime_type, image_b64) in enumerate(images, start=1):
            ext = _mime_to_extension(mime_type)
            filename = f"visual_{self._artifact_counter:04d}_{index:02d}{ext}"
            local_rel = f"{TOOL_OUTPUTS_DIR}/vision_display/{filename}"
            try:
                target = self._safe_artifact_target(local_rel)
                self._write_local_tool_artifact(
                    sandbox_rel_path="",
                    local_rel_path=local_rel,
                    data=base64.b64decode(image_b64),
                    source="vision_display",
                    list_for_model=False,
                )
                paths.append(str(target))
            except Exception:
                continue
        return paths

    def _format_response(
        self,
        text_output: str,
        images: list[tuple[str, str]],
        artifact_paths: list[str],
    ):
        text = text_output or PYTHON_TOOL_NO_OUTPUT_MESSAGE
        text = self._maybe_append_file_not_found_hint(text)
        if artifact_paths:
            artifacts_text = "\n".join(f"- {path}" for path in artifact_paths)
            text = f"{text}\n\nvisual_artifacts:\n{artifacts_text}"

        artifact_note = self._saved_artifacts_note()
        if artifact_note:
            text = f"{text}\n\n{artifact_note}"

        if not images:
            return text

        mode = _visual_response_mode()
        if mode == "text" or self._force_text_response:
            return text

        max_images = _visual_max_return_images()
        response_images: list[tuple[str, str]] = []
        duplicate_count = 0
        limited_count = 0
        resized_count = 0
        for mime_type, image_b64 in images:
            digest = _image_digest(image_b64)
            if digest in self._returned_image_hashes:
                duplicate_count += 1
                continue
            if len(response_images) >= max_images:
                limited_count += 1
                continue
            self._returned_image_hashes.add(digest)
            response_mime_type, response_b64, resized = _resize_image_for_response(
                mime_type,
                image_b64,
            )
            resized_count += int(resized)
            response_images.append((response_mime_type, response_b64))

        text = self._append_visual_response_note(
            text,
            duplicate_count=duplicate_count,
            limited_count=limited_count,
            resized_count=resized_count,
            max_images=max_images,
        )
        if not response_images:
            return text

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for mime_type, image_b64 in response_images:
            content.append(_image_content_part(mime_type, image_b64))
        return content

    def _maybe_append_file_not_found_hint(self, text: str) -> str:
        if "filenotfounderror" not in str(text).lower():
            return text
        input_paths = (
            ", ".join(f"`{path}`" for path in self._input_image_paths)
            if self._input_image_paths
            else "none"
        )
        hint = (
            "visual_file_hint:\n"
            f"- Current input image files: {input_paths}\n"
            f"- Save generated files under `{TOOL_OUTPUTS_DIR}/`.\n"
            "- Do not guess page or crop filenames unless your code created them earlier in this notebook."
        )
        return f"{text}\n\n{hint}"

    def _append_visual_response_note(
        self,
        text: str,
        *,
        duplicate_count: int,
        limited_count: int,
        resized_count: int,
        max_images: int,
    ) -> str:
        notes: list[str] = []
        if duplicate_count:
            notes.append(f"{duplicate_count} duplicate visual image(s) omitted.")
        if limited_count:
            notes.append(
                f"{limited_count} additional visual image(s) omitted by "
                f"VISUAL_TOOL_MAX_RETURN_IMAGES={max_images}."
            )
        if resized_count:
            notes.append(
                f"{resized_count} visual image(s) resized for the tool response "
                f"to long edge <= {_visual_max_image_edge()}."
            )
        if not notes:
            return text
        note_text = "\n".join(f"- {note}" for note in notes)
        return f"{text}\n\nvisual_response_note:\n{note_text}"

    def call(self, params, files=None, timeout=None, **kwargs):
        del files, kwargs
        try:
            code = normalize_python_code_input(params)
            code = _normalize_visual_input_image_paths(code)
            if python_request_blocked(params, code):
                return blocked_python_message()
            if python_relative_path_policy_enabled():
                violations = find_relative_path_policy_violations(code)
                if violations:
                    return format_path_policy_error("Visual Python Interpreter", violations)
            if not code.strip():
                return "[Visual Python Interpreter Error]: Empty code."
            if not SANDBOX_FUSION_ENDPOINTS:
                return missing_sandbox_fusion_endpoint_message()

            timeout = PYTHON_TOOL_TIMEOUT_SEC if timeout is None else int(timeout)
            last_error = None
            for attempt in range(PYTHON_TOOL_MAX_RETRIES):
                endpoint = self._choose_sandbox_endpoint()
                try:
                    print(f"Attempt {attempt + 1}/{PYTHON_TOOL_MAX_RETRIES} using endpoint: {endpoint!r}")
                    text_output, images, artifact_paths = self._execute_visual_code(
                        code,
                        endpoint,
                        timeout,
                        update_history=True,
                    )
                    if text_output == blocked_python_message():
                        print("[VisualPython] Code Result blocked by evaluation URL/keyword guard")
                        return text_output
                    result = self._format_response(text_output, images, artifact_paths)
                    print("[VisualPython] Jupyter Result", result)
                    print("SUCCESS RUNNING VISUAL TOOL")
                    return result
                except Exception as exc:
                    last_error = (
                        f"[Visual Python Interpreter Error]: {str(exc)} "
                        f"on endpoint {endpoint!r}"
                    )
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    self._rotate_sandbox_endpoint(endpoint)
                    if attempt == PYTHON_TOOL_MAX_RETRIES - 1:
                        return last_error
                    continue
            return last_error or "[Visual Python Interpreter Error]: All attempts failed."
        except Exception as exc:
            return f"[Visual Python Interpreter Error]: {str(exc)}"
