import os
import json
from pathlib import Path
from typing import Any

import react_agent_qwen_api as base_module
from prompt import OPENAI_COMPAT_SYSTEM_PROMPT, OPENAI_COMPAT_SYSTEM_PROMPT_SUFFIX
from react_agent_qwen_api import MultiTurnReactAgent as BaseMultiTurnReactAgent

from runtime_env import get_env_bool
from tool_runtime_registry import (
    TOOL_ALIASES,
    build_tool_runtime,
    call_runtime_tool,
    canonicalize_tool_name,
    schema_tool_name,
)
from tool_http_client import close_sandbox_http_run


ORIGINAL_TODAY_DATE = base_module.today_date


def openai_compat_current_date() -> str:
    if not get_env_bool("OPENAI_COMPAT_FREEZE_DATE", default=False):
        return ORIGINAL_TODAY_DATE()
    return (os.getenv("OPENAI_COMPAT_FIXED_DATE") or "2026-05-15").strip() or "2026-05-15"


class MultiTurnReactAgent(BaseMultiTurnReactAgent):
    def __init__(self, function_list=None, llm=None, **kwargs):
        if not function_list and os.getenv("SCIENCE_AGENT_TOOLS", "").strip():
            function_list = [name.strip() for name in os.environ["SCIENCE_AGENT_TOOLS"].split(",") if name.strip()]
        super().__init__(function_list=function_list, llm=llm, **kwargs)
        runtime = build_tool_runtime(function_list, default_tools=["search", "visit"])
        schema_snapshot = os.getenv("SCIENCE_AGENT_TOOL_SCHEMAS_FILE")
        if schema_snapshot:
            schemas = json.loads(Path(schema_snapshot).read_text(encoding="utf-8"))
            names = [schema["function"]["name"] for schema in schemas]
            if names != runtime.active_tool_names or any(schema["type"] != "function" for schema in schemas):
                raise ValueError("Tool snapshot must match this run's active tool names and order")
            runtime.active_tool_schemas = schemas
        self.tool_runtime = runtime
        self.requested_tool_keys = runtime.requested_tool_keys
        self.active_tool_keys = runtime.active_tool_keys
        self.active_tool_names = runtime.active_tool_names
        self.active_tools = runtime.active_tools
        self.active_tool_map = runtime.active_tool_map
        self.active_tool_schemas = runtime.active_tool_schemas
        self.dynamic_tool_prompt = runtime.prompt_append
        self._tool_key_to_tool = dict(zip(runtime.active_tool_keys, runtime.active_tools))
        self._tool_key_to_schema = dict(zip(runtime.active_tool_keys, runtime.active_tool_schemas))
        self.current_active_tool_keys = list(runtime.active_tool_keys)
        self.current_active_tool_map = dict(runtime.active_tool_map)
        self.current_active_tool_schemas = list(runtime.active_tool_schemas)
        self._remote_sandbox_task_context: dict[str, Any] = {}
        self._remote_sandbox_session_id = ""
        self._remote_initial_vision_inputs_attempted = False
        self._remote_initial_vision_inputs_staged = False

        base_module.TOOL_CLASS = self.active_tools
        base_module.TOOL_MAP = self.active_tool_map
        base_module.TOOLS_DEEPSEEK_V32 = self.active_tool_schemas

    def _build_system_prompt(self) -> str:
        if not get_env_bool("OPENAI_COMPAT_USE_SYSTEM_PROMPT", default=True):
            return ""

        snapshot = os.getenv("SCIENCE_AGENT_SYSTEM_PROMPT_FILE")
        if snapshot:
            return Path(snapshot).read_text(encoding="utf-8")

        current_date = openai_compat_current_date()
        prompt = f"{OPENAI_COMPAT_SYSTEM_PROMPT}{current_date}"
        prompt_suffix = OPENAI_COMPAT_SYSTEM_PROMPT_SUFFIX.strip()
        if prompt_suffix:
            return f"{prompt}\n\n{prompt_suffix}"
        return prompt

    def _get_api_key(self) -> str:
        return (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
            or "EMPTY"
        )

    def reset_tool_runtime_state(self) -> None:
        close_sandbox_http_run()
        for tool in self.active_tools:
            close_sandbox = getattr(tool, "close_sandbox", None)
            if callable(close_sandbox):
                close_sandbox()
            reset = getattr(tool, "reset_jupyter_history", None)
            if callable(reset):
                reset()
        self.current_active_tool_keys = list(self.active_tool_keys)
        self.current_active_tool_map = dict(self.active_tool_map)
        self.current_active_tool_schemas = list(self.active_tool_schemas)
        self._remote_sandbox_task_context = {}
        self._remote_sandbox_session_id = ""
        self._remote_initial_vision_inputs_attempted = False
        self._remote_initial_vision_inputs_staged = False

    @staticmethod
    def _initial_vision_inputs_from_user_message(
        item: dict,
        user_message: dict | None,
    ) -> list[dict[str, str]]:
        inline_image = str(item.get("image") or "").strip()
        has_declared_image = bool(item.get("image_question") or item.get("file_name"))
        if inline_image.startswith("data:image") and not has_declared_image:
            return []

        content = user_message.get("content") if isinstance(user_message, dict) else None
        if not isinstance(content, list):
            return []

        images: list[dict[str, str]] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            data_url = str(image_url or "").strip()
            if not data_url.startswith("data:image"):
                continue
            images.append(
                {
                    "data_url": data_url,
                    "source_tool": "problem_input",
                    "label": f"problem_image_{len(images) + 1}",
                }
            )
        return images

    def _configure_task_tools(
        self,
        item: dict,
        item_base_dir: str,
        data: dict,
        *,
        user_message: dict | None = None,
    ) -> str:
        super()._configure_task_tools(
            item,
            item_base_dir,
            data,
            user_message=user_message,
        )
        active_keys = list(self.active_tool_keys)
        tool_hint = ""
        advanced_visual_active = any(key in active_keys for key in ("advanced_search", "advanced_visit"))
        self._remote_sandbox_task_context = {
            "item": item,
            "item_base_dir": item_base_dir,
            "artifact_dir": data.get("tool_artifact_dir"),
            "initial_vision_inputs": [],
        }
        self._remote_sandbox_session_id = ""
        self._remote_initial_vision_inputs_attempted = False
        self._remote_initial_vision_inputs_staged = False

        if "vision" in active_keys:
            benchmark = str(item.get("eval_benchmark") or item.get("source") or "").strip()
            vision_allowed = benchmark in {
                "hle_all",
                "hle_all_mini",
                "hle_all_mini_100",
                "hle_all_mm",
                "hle_subject",
                "hipho",
                "hipho_mm",
            }
            vision_tool = self._tool_key_to_tool.get("vision")
            vision_is_remote = bool(getattr(vision_tool, "is_remote_tool_proxy", False))
            if vision_allowed and vision_is_remote:
                self._remote_sandbox_task_context["initial_vision_inputs"] = (
                    self._initial_vision_inputs_from_user_message(item, user_message)
                )
            if (
                vision_allowed
                and vision_tool is not None
                and not vision_is_remote
            ):
                set_context = getattr(vision_tool, "set_task_context", None)
                if callable(set_context):
                    tool_hint = set_context(
                        item,
                        base_dir=item_base_dir,
                        artifact_dir=data.get("tool_artifact_dir"),
                    )
            if not vision_allowed or (not vision_is_remote and not tool_hint and not advanced_visual_active):
                active_keys = [key for key in active_keys if key != "vision"]

        self.current_active_tool_keys = active_keys
        self.current_active_tool_schemas = [
            self._tool_key_to_schema[key]
            for key in active_keys
            if key in self._tool_key_to_schema
        ]
        self.current_active_tool_map = {}
        for key in active_keys:
            tool = self._tool_key_to_tool.get(key)
            if tool is None:
                continue
            self.current_active_tool_map[tool.name] = tool
            for alias, canonical in TOOL_ALIASES.items():
                if canonical == key:
                    self.current_active_tool_map[alias] = tool
        return tool_hint

    def _active_tool_schemas_for_current_task(self):
        return self.current_active_tool_schemas

    def _schema_tool_name(self, tool_name: str) -> str:
        try:
            return schema_tool_name(tool_name)
        except Exception:
            return tool_name

    def _set_tool_text_response_fallback(self, enabled: bool) -> None:
        for tool in self.active_tools:
            setter = getattr(tool, "set_text_response_fallback", None)
            if callable(setter):
                setter(enabled)

    def _remote_sandbox_session_keys(self) -> list[str]:
        keys: list[str] = []
        for key in ("python", "vision"):
            if key not in self.current_active_tool_keys:
                continue
            tool = self._tool_key_to_tool.get(key)
            if getattr(tool, "is_remote_tool_proxy", False):
                keys.append(key)
        return keys

    def _stage_initial_remote_vision_inputs(self) -> tuple[list[str], list[str]]:
        if self._remote_initial_vision_inputs_attempted:
            return [], []

        context = self._remote_sandbox_task_context or {}
        images = context.get("initial_vision_inputs") or []
        if not isinstance(images, list) or not images:
            self._remote_initial_vision_inputs_attempted = True
            return [], []

        self._remote_initial_vision_inputs_attempted = True
        vision_tool = self._tool_key_to_tool.get("vision")
        append_inputs = getattr(vision_tool, "append_vision_inputs", None)
        if vision_tool is None or not callable(append_inputs):
            errors = ["remote vision tool does not support appending problem images"]
            print(f"[remote_vision] Warning: {errors[0]}")
            return [], errors

        try:
            response = append_inputs(images)
        except Exception as exc:  # noqa: BLE001
            errors = [f"{type(exc).__name__}: {exc}"]
            sample = context.get("item") or {}
            sample_id = sample.get("id") or sample.get("uid") or sample.get("question_id") or "unknown"
            print(
                "[remote_vision] Warning: failed to stage problem images; "
                f"sample_id={sample_id}, session_id={self._remote_sandbox_session_id or 'unknown'}, "
                f"images={len(images)}, errors={errors}"
            )
            return [], errors

        paths = response.get("paths") if isinstance(response, dict) else []
        errors = response.get("errors") if isinstance(response, dict) else []
        normalized_paths = (
            [str(path) for path in paths if str(path).strip()]
            if isinstance(paths, list)
            else []
        )
        normalized_errors = (
            [str(error) for error in errors if str(error).strip()]
            if isinstance(errors, list)
            else []
        )
        self._remote_initial_vision_inputs_staged = bool(normalized_paths)
        context["initial_vision_paths"] = normalized_paths
        if normalized_errors or not normalized_paths:
            sample = context.get("item") or {}
            sample_id = sample.get("id") or sample.get("uid") or sample.get("question_id") or "unknown"
            print(
                "[remote_vision] Warning: problem image staging was incomplete; "
                f"sample_id={sample_id}, session_id={self._remote_sandbox_session_id or 'unknown'}, "
                f"images={len(images)}, paths={len(normalized_paths)}, errors={normalized_errors}"
            )
        return normalized_paths, normalized_errors

    def _ensure_remote_sandbox_session(self, *, stage_initial_vision: bool = False) -> str:
        session_keys = self._remote_sandbox_session_keys()
        if not session_keys:
            return ""

        existing_session_id = self._remote_sandbox_session_id
        if not existing_session_id:
            for key in session_keys:
                tool = self._tool_key_to_tool.get(key)
                existing_session_id = str(getattr(tool, "session_id", "") or "")
                if existing_session_id:
                    break
        if existing_session_id:
            self._remote_sandbox_session_id = existing_session_id
            for key in session_keys:
                tool = self._tool_key_to_tool.get(key)
                bind_session = getattr(tool, "bind_session", None)
                if callable(bind_session):
                    bind_session(existing_session_id)
            if stage_initial_vision:
                self._stage_initial_remote_vision_inputs()
            return existing_session_id

        first_remote_tool = self._tool_key_to_tool.get(session_keys[0])
        create_shared_session = getattr(first_remote_tool, "create_shared_session", None)
        if not callable(create_shared_session):
            return ""
        context = self._remote_sandbox_task_context or {}
        session_data = create_shared_session(
            context.get("item") or {},
            tools=session_keys,
            base_dir=context.get("item_base_dir"),
            artifact_dir=context.get("artifact_dir"),
        )
        session_id = str(session_data.get("session_id") or "")
        if not session_id:
            return ""
        self._remote_sandbox_session_id = session_id
        for key in session_keys:
            tool = self._tool_key_to_tool.get(key)
            bind_session = getattr(tool, "bind_session", None)
            if callable(bind_session):
                bind_session(session_id)
        if stage_initial_vision:
            self._stage_initial_remote_vision_inputs()
        return session_id

    def _advanced_visual_session_enabled(self) -> bool:
        return get_env_bool("ADVANCED_VISUAL_SESSION_ARTIFACTS", default=True)

    def _advanced_visual_inputs_from_result(self, tool_name: str, result) -> list[dict[str, str]]:
        if not base_module.tool_content_has_image(result):
            return []
        images: list[dict[str, str]] = []
        for index, part in enumerate(result, start=1):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            data_url = str(image_url.get("url") or "").strip()
            if not data_url.startswith("data:image/"):
                continue
            images.append(
                {
                    "data_url": data_url,
                    "source_tool": tool_name,
                    "label": f"{tool_name}_{index}",
                }
            )
        return images

    def _append_advanced_visual_to_vision_session(self, tool_name: str, result) -> tuple[list[str], list[str]]:
        if "vision" not in self.current_active_tool_keys:
            return [], []
        if not self._advanced_visual_session_enabled():
            return [], []
        images = self._advanced_visual_inputs_from_result(tool_name, result)
        if not images:
            return [], []

        vision_tool = self._tool_key_to_tool.get("vision")
        if vision_tool is None:
            return [], []

        if getattr(vision_tool, "is_remote_tool_proxy", False):
            try:
                if not self._ensure_remote_sandbox_session():
                    return [], ["failed to create remote sandbox session"]
                append_inputs = getattr(vision_tool, "append_vision_inputs", None)
                if not callable(append_inputs):
                    return [], ["remote vision tool does not support appending visual inputs"]
                response = append_inputs(images)
            except Exception as exc:  # noqa: BLE001
                return [], [f"{type(exc).__name__}: {exc}"]
        else:
            add_inputs = getattr(vision_tool, "add_input_image_data_urls", None)
            if not callable(add_inputs):
                return [], ["local vision tool does not support appending visual inputs"]
            try:
                response = add_inputs(images)
            except Exception as exc:  # noqa: BLE001
                return [], [f"{type(exc).__name__}: {exc}"]

        paths = response.get("paths") if isinstance(response, dict) else []
        errors = response.get("errors") if isinstance(response, dict) else []
        return (
            [str(path) for path in paths if str(path).strip()] if isinstance(paths, list) else [],
            [str(error) for error in errors if str(error).strip()] if isinstance(errors, list) else [],
        )

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        try:
            key = canonicalize_tool_name(tool_name)
        except Exception:
            key = ""
        if key in {"python", "vision"}:
            self._ensure_remote_sandbox_session(stage_initial_vision=key == "vision")
        result = call_runtime_tool(self.current_active_tool_map, tool_name, tool_args, **kwargs)
        if key in {"advanced_search", "advanced_visit"}:
            paths, errors = self._append_advanced_visual_to_vision_session(key, result)
            if errors:
                print(f"[advanced_visual] Warning: failed to append some visual inputs to vision session: {errors}")
            return self._maybe_append_advanced_visual_followup_hint(tool_name, result, session_paths=paths)
        return result

    def _maybe_append_advanced_visual_followup_hint(self, tool_name: str, result, *, session_paths: list[str] | None = None):
        try:
            key = canonicalize_tool_name(tool_name)
        except Exception:
            return result
        if key not in {"advanced_search", "advanced_visit"}:
            return result
        if not base_module.tool_content_has_image(result):
            return result
        if not get_env_bool("ADVANCED_VISUAL_TOOL_HINT", default=True):
            return result

        active = set(self.current_active_tool_keys)
        paths = [str(path) for path in (session_paths or []) if str(path).strip()]
        if "vision" not in active or not paths:
            return result

        formatted_paths = ", ".join(f"`{path}`" for path in paths)
        hint_text = (
            "\n\navailable_followup_tools:\n"
            "- VisualPythonInterpreter can inspect the visual evidence saved in "
            f"the current visual session at: {formatted_paths}."
        )
        if not isinstance(result, list):
            return result
        updated = []
        appended = False
        for part in result:
            if isinstance(part, dict) and part.get("type") == "text" and not appended:
                new_part = dict(part)
                new_part["text"] = str(new_part.get("text") or "").rstrip() + hint_text
                updated.append(new_part)
                appended = True
            else:
                updated.append(part)
        if not appended:
            updated.insert(0, {"type": "text", "text": hint_text.strip()})
        return updated
