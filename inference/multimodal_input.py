import json
import os
from typing import Any, Optional

from file_tools.utils import encode_image_as_base64
from prompt import SCIENCE_USER_TEMPLATE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)


VISION_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "mm_vision_tower",
    "vision_tower",
)

MULTIMODAL_RUNTIME_MARKERS = (
    "at most 0 image",
    "may be provided in one prompt",
    "image_url",
    "image input",
    "multimodal",
    "vision",
    "language-model-only",
    "unsupported content",
    "expected a string",
    "content type",
    "does not support image",
    "does not support vision",
)


def model_image_inputs_enabled() -> bool:
    raw_value = str(os.getenv("AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS", "1")).strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _content_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            return True
    return False


def messages_have_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if _content_has_image(message.get("content")):
            return True
    return False


def _resolve_config_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        return config_path if os.path.exists(config_path) else None
    if model_path.endswith("config.json") and os.path.exists(model_path):
        return model_path
    return None


def infer_model_vision_support(model_path: Optional[str]) -> tuple[Optional[bool], str]:
    config_path = _resolve_config_path(model_path)
    if not config_path:
        return None, f"config_unavailable model_path={model_path!r}"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        return None, f"config_read_failed path={config_path!r} error={exc}"

    if any(key in config for key in VISION_CONFIG_KEYS):
        return True, f"config_path={config_path!r}"

    return False, f"config_path={config_path!r}"


def build_static_multimodal_support_error(
    messages: list[dict[str, Any]],
    *,
    model_path: Optional[str],
    served_model_name: str,
) -> Optional[str]:
    if not messages_have_image(messages):
        return None

    supports_vision, detail = infer_model_vision_support(model_path)
    if supports_vision is False:
        return (
            "Image input is enabled for this sample, but the configured model does not "
            "appear to expose vision support. "
            f"served_model={served_model_name}, model_path={model_path!r}, {detail}"
        )
    return None


def build_runtime_multimodal_support_error(
    messages: list[dict[str, Any]],
    *,
    model_path: Optional[str],
    served_model_name: str,
    error_text: str,
    base_url: Optional[str] = None,
) -> Optional[str]:
    if not messages_have_image(messages):
        return None

    supports_vision, detail = infer_model_vision_support(model_path)
    lowered = error_text.lower()
    if supports_vision is False or any(marker in lowered for marker in MULTIMODAL_RUNTIME_MARKERS):
        return (
            "Image input is enabled for this sample, but the backend rejected multimodal "
            "messages. "
            f"served_model={served_model_name}, model_path={model_path!r}, "
            f"base_url={base_url!r}, config_support={supports_vision}, {detail}, "
            f"error={error_text}"
        )
    return None


def _add_candidate(candidates: list[str], candidate: Optional[str]) -> None:
    if not candidate:
        return
    candidates.append(os.path.normpath(candidate))


def _should_use_hipho_image_path(item: dict[str, Any]) -> bool:
    file_name = str(item.get("file_name") or "").replace("\\", "/").strip()
    if file_name.startswith("image_question/"):
        return True

    image_question = item.get("image_question")
    if isinstance(image_question, str):
        return bool(image_question.strip())
    if isinstance(image_question, list):
        return any(isinstance(entry, str) and entry.strip() for entry in image_question)
    return False


def _iter_image_base_dirs(
    base_dir: Optional[str],
    *,
    include_hipho_env: bool = False,
) -> list[str]:
    base_dirs: list[str] = []
    if base_dir:
        base_dirs.append(base_dir)
        base_dirs.append(os.path.dirname(base_dir))

    if include_hipho_env:
        hipho_image_path = str(os.getenv("HIPHO_IMAGE_PATH") or "").strip()
        if hipho_image_path:
            base_dirs.append(hipho_image_path)
            base_dirs.append(os.path.dirname(hipho_image_path))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in base_dirs:
        normalized = os.path.normpath(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _build_image_candidates(
    file_name: str,
    base_dir: Optional[str],
    *,
    include_hipho_env: bool = False,
) -> list[str]:
    candidates: list[str] = []
    normalized = file_name.replace("\\", "/").strip()

    _add_candidate(candidates, file_name)
    _add_candidate(candidates, os.path.abspath(file_name))
    _add_candidate(candidates, os.path.join(PROJECT_DIR, normalized))

    for candidate_base_dir in _iter_image_base_dirs(
        base_dir,
        include_hipho_env=include_hipho_env,
    ):
        _add_candidate(candidates, os.path.join(candidate_base_dir, normalized))

    if normalized.startswith("datasets/hle/"):
        remapped = normalized.replace("datasets/hle/", "datasets/data/hle/", 1)
        _add_candidate(candidates, remapped)
        _add_candidate(candidates, os.path.join(PROJECT_DIR, remapped))
        for candidate_base_dir in _iter_image_base_dirs(
            base_dir,
            include_hipho_env=include_hipho_env,
        ):
            _add_candidate(candidates, os.path.join(candidate_base_dir, remapped))

    return candidates


def _iter_image_question_names(item: dict[str, Any]) -> list[str]:
    declared: list[str] = []
    seen: set[str] = set()

    image_question = item.get("image_question")
    if isinstance(image_question, str):
        image_entries = [image_question]
    elif isinstance(image_question, list):
        image_entries = image_question
    else:
        image_entries = []

    for raw_path in image_entries:
        if not isinstance(raw_path, str):
            continue
        normalized = raw_path.strip()
        if not normalized or normalized in seen:
            continue
        declared.append(normalized)
        seen.add(normalized)

    return declared


def _iter_declared_image_names(item: dict[str, Any]) -> list[str]:
    declared = _iter_image_question_names(item)
    seen = set(declared)

    single_file = str(item.get("file_name") or "").strip()
    if single_file and single_file not in seen:
        declared.append(single_file)

    return declared


def _resolve_declared_image_paths(
    declared_image_names: list[str],
    *,
    base_dir: Optional[str] = None,
    include_hipho_env: bool = False,
) -> list[str]:
    resolved_paths: list[str] = []
    for file_name in declared_image_names:
        candidates = _build_image_candidates(
            file_name,
            base_dir,
            include_hipho_env=include_hipho_env,
        )

        seen: set[str] = set()
        resolved_path: Optional[str] = None
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate):
                resolved_path = candidate
                break

        if resolved_path is None:
            raise FileNotFoundError(
                f"Image file declared by sample was not found: file_name={file_name!r}, "
                f"base_dir={base_dir!r}, candidates={sorted(seen)}"
            )
        resolved_paths.append(resolved_path)

    return resolved_paths


def _resolve_inline_image_urls(item: dict[str, Any]) -> list[str]:
    image_data = str(item.get("image") or "").strip()
    if image_data.startswith("data:image"):
        return [image_data]
    return []


def resolve_item_image_paths(
    item: dict[str, Any],
    *,
    base_dir: Optional[str] = None,
) -> list[str]:
    declared_image_names = _iter_declared_image_names(item)
    return _resolve_declared_image_paths(
        declared_image_names,
        base_dir=base_dir,
        include_hipho_env=_should_use_hipho_image_path(item),
    )


def resolve_item_image_path(item: dict[str, Any], *, base_dir: Optional[str] = None) -> Optional[str]:
    image_paths = resolve_item_image_paths(item, base_dir=base_dir)
    if not image_paths:
        return None
    return image_paths[0]


def resolve_item_image_urls(
    item: dict[str, Any],
    *,
    base_dir: Optional[str] = None,
) -> list[str]:
    image_question_names = _iter_image_question_names(item)
    if image_question_names:
        return [
            encode_image_as_base64(path)
            for path in _resolve_declared_image_paths(
                image_question_names,
                base_dir=base_dir,
                include_hipho_env=_should_use_hipho_image_path(item),
            )
        ]

    inline_image_urls = _resolve_inline_image_urls(item)
    if inline_image_urls:
        return inline_image_urls

    file_name = str(item.get("file_name") or "").strip()
    if not file_name:
        return []

    return [
        encode_image_as_base64(path)
        for path in _resolve_declared_image_paths(
            [file_name],
            base_dir=base_dir,
            include_hipho_env=_should_use_hipho_image_path(item),
        )
    ]


def build_initial_user_message(
    item: dict[str, Any],
    question: str,
    *,
    question_suffix: str = "",
    model_path: Optional[str],
    served_model_name: str,
    base_dir: Optional[str] = None,
) -> dict[str, Any]:
    text = question + question_suffix
    if not text.startswith(SCIENCE_USER_TEMPLATE.format(question="").rstrip()):
        text = SCIENCE_USER_TEMPLATE.format(question=text)
    if not model_image_inputs_enabled():
        return {"role": "user", "content": text}

    image_urls = resolve_item_image_urls(item, base_dir=base_dir)
    if not image_urls:
        return {"role": "user", "content": text}

    content_blocks = [{"type": "text", "text": text}]
    for image_url in image_urls:
        content_blocks.append(
            {"type": "image_url", "image_url": {"url": image_url}}
        )

    message = {
        "role": "user",
        "content": content_blocks,
    }

    static_error = build_static_multimodal_support_error(
        [message],
        model_path=model_path,
        served_model_name=served_model_name,
    )
    if static_error:
        raise RuntimeError(static_error)

    return message
