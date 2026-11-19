"""Separate opt-in progress-verifier agent; the original agent is unchanged."""

import copy
import os
from contextvars import ContextVar

from progress_verifier import PROGRESS_INSTRUCTION, PROGRESS_RE, progress_verifier_scope, start_progress_verifier
from react_agent_qwen_api import MultiTurnReactAgent as BaseAgent


_RUN_STATE = ContextVar("science_progress_agent_run", default=None)


class ProgressVerifierMixin:
    @progress_verifier_scope
    def _run(self, *args, **kwargs):
        token = _RUN_STATE.set({"manager": None, "histories": []})
        try:
            return super()._run(*args, **kwargs)
        finally:
            _RUN_STATE.reset(token)

    def _build_system_prompt(self):
        prompt = super()._build_system_prompt()
        if os.getenv("SCIENCE_AGENT_PROGRESS_VERIFIER", "0").lower() in {"1", "true", "yes"}:
            prompt += "\n" + PROGRESS_INSTRUCTION
        return prompt

    def _set_fallback_message_lists(self, *message_lists):
        super()._set_fallback_message_lists(*message_lists)
        state = _RUN_STATE.get()
        if state is None:
            return
        # The base loop supplies full history first, model/compacted view second.
        histories = [messages for messages in message_lists if isinstance(messages, list)]
        state["histories"] = histories
        if not histories:
            return
        if state["manager"] is None:
            state["manager"] = start_progress_verifier(
                histories[0][1], endpoint=getattr(self, "progress_verifier_endpoint", None))
        manager = state["manager"]
        if manager is None:
            return
        feedback = manager.poll()
        seen = set()
        for messages in histories:
            if id(messages) not in seen:
                # This hook runs before a policy request, after tool execution.
                messages.extend(copy.deepcopy(feedback))
                seen.add(id(messages))

    def _build_assistant_request_message(self, reasoning, content, serialized_tool_calls,
                                         response_output_items=None):
        message = super()._build_assistant_request_message(
            reasoning, content, serialized_tool_calls, response_output_items)
        state = _RUN_STATE.get()
        if state is not None and state["manager"] is not None and state["histories"]:
            if not serialized_tool_calls:
                if PROGRESS_RE.search(content or ""):
                    state["manager"]._event(
                        "progress_checkpoint_skipped", reason="no_same_turn_tool_call")
                return message
            # Base loop has appended the assistant to full history, but has not
            # dispatched its tools yet. This is precisely the evidence boundary.
            history = state["histories"][0]
            state["manager"].submit(
                history, content, sum(m.get("role") == "assistant" for m in history))
        return message


class MultiTurnReactAgent(ProgressVerifierMixin, BaseAgent):
    pass
