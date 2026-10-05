"""Conservative, deterministic helpers for benchmark answer extraction."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


ANSWER_EXTRACTION_MODE_ENV = "AGENT_EVAL_ANSWER_EXTRACTION_MODE"
LEGACY_ANSWER_EXTRACTION_MODE = "legacy"
CONSERVATIVE_ANSWER_EXTRACTION_MODE = "conservative_v2"
ANSWER_EXTRACTION_MODES = {
    LEGACY_ANSWER_EXTRACTION_MODE,
    CONSERVATIVE_ANSWER_EXTRACTION_MODE,
}
DEFAULT_ANSWER_EXTRACTION_MODE = CONSERVATIVE_ANSWER_EXTRACTION_MODE

HIPHO_STEP_JUDGE_SOLUTION_MODE_ENV = "HIPHO_STEP_JUDGE_SOLUTION_MODE"
HIPHO_STEP_JUDGE_VISIBLE_MODE = "visible"
HIPHO_STEP_JUDGE_REASONING_FALLBACK_MODE = "answer_only_reasoning_fallback"
HIPHO_STEP_JUDGE_SOLUTION_MODES = {
    HIPHO_STEP_JUDGE_VISIBLE_MODE,
    HIPHO_STEP_JUDGE_REASONING_FALLBACK_MODE,
}
DEFAULT_HIPHO_STEP_JUDGE_SOLUTION_MODE = HIPHO_STEP_JUDGE_REASONING_FALLBACK_MODE

_ANSWER_START_TAG = "<answer>"
_ANSWER_END_TAG = "</answer>"
_FINAL_ANSWER_PREFIX_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:(?:\*\*|__)[ \t]*)?"
    r"final(?:[ _-]+)answers?\b",
    flags=re.IGNORECASE,
)
_EMPTY_THINK_RE = re.compile(r"<think\b[^>]*>\s*</think>", flags=re.IGNORECASE | re.DOTALL)
_FORMAT_TAG_RE = re.compile(
    r"</?(?:answer|final|response|assistant)>|"
    r"<\|(?:im_end|endoftext|end_of_text|end)\|>",
    flags=re.IGNORECASE,
)
_FINAL_ANSWER_LABEL_ONLY_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:(?:\*\*|__)[ \t]*)?"
    r"final(?:[ _-]+)answers?\b[ \t]*(?:(?:\*\*|__)[ \t]*)?"
    r"(?:(?::|：|=|-)[ \t]*(?:(?:\*\*|__)[ \t]*)?)?$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_BOXED_START_RE = re.compile(r"(?<![A-Za-z])\\?boxed[ \t]*\{")


@dataclass(frozen=True)
class TextRegion:
    source: str
    content: str
    region_start: int
    region_end: int
    content_start: int
    content_end: int


@dataclass(frozen=True)
class BoxedSpan:
    content: str
    canonical: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class CandidateDecision:
    value: str
    source: str
    reason: str
    changed: bool


def normalize_answer_extraction_mode(mode: str | None = None) -> str:
    value = mode
    if value is None:
        value = os.getenv(ANSWER_EXTRACTION_MODE_ENV, DEFAULT_ANSWER_EXTRACTION_MODE)
    normalized = str(value or "").strip().lower()
    if normalized not in ANSWER_EXTRACTION_MODES:
        return LEGACY_ANSWER_EXTRACTION_MODE
    return normalized


def normalize_hipho_step_judge_solution_mode(mode: str | None = None) -> str:
    value = mode
    if value is None:
        value = os.getenv(
            HIPHO_STEP_JUDGE_SOLUTION_MODE_ENV,
            DEFAULT_HIPHO_STEP_JUDGE_SOLUTION_MODE,
        )
    normalized = str(value or "").strip().lower()
    if normalized not in HIPHO_STEP_JUDGE_SOLUTION_MODES:
        return HIPHO_STEP_JUDGE_VISIBLE_MODE
    return normalized


def last_complete_answer_region(text: str) -> TextRegion | None:
    if not isinstance(text, str) or not text:
        return None

    lowered = text.lower()
    end_start = lowered.rfind(_ANSWER_END_TAG)
    if end_start < 0:
        return None
    start = lowered.rfind(_ANSWER_START_TAG, 0, end_start)
    if start < 0:
        return None

    content_start = start + len(_ANSWER_START_TAG)
    region_end = end_start + len(_ANSWER_END_TAG)
    return TextRegion(
        source="answer_tag",
        content=text[content_start:end_start].strip(),
        region_start=start,
        region_end=region_end,
        content_start=content_start,
        content_end=end_start,
    )


def _consume_markdown_decoration(line: str, cursor: int) -> int:
    while cursor < len(line) and line[cursor] in " \t":
        cursor += 1
    if line.startswith("**", cursor) or line.startswith("__", cursor):
        cursor += 2
        while cursor < len(line) and line[cursor] in " \t":
            cursor += 1
    return cursor


def last_anchored_final_answer_region(text: str) -> TextRegion | None:
    """Return the last line-anchored Final Answer(s) region.

    Prose such as ``the final answer is ...`` is intentionally rejected. A
    same-line value requires ``:``, ``：``, ``=``, or ``-`` as a separator;
    without a separator the label must occupy the line by itself.
    """

    if not isinstance(text, str) or not text:
        return None

    matches: list[TextRegion] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        match = _FINAL_ANSWER_PREFIX_RE.match(line)
        if match is None:
            offset += len(raw_line)
            continue

        cursor = _consume_markdown_decoration(line, match.end())
        if cursor < len(line):
            if line[cursor] not in {":", "：", "=", "-"}:
                offset += len(raw_line)
                continue
            cursor += 1
            cursor = _consume_markdown_decoration(line, cursor)
            content_start = offset + cursor
        else:
            content_start = offset + len(raw_line)

        matches.append(
            TextRegion(
                source="final_answer",
                content=text[content_start:].strip(),
                region_start=offset + match.start(),
                region_end=len(text),
                content_start=content_start,
                content_end=len(text),
            )
        )
        offset += len(raw_line)

    # splitlines() returns no row for an empty trailing fragment, which is fine:
    # a marker was already recorded from the preceding physical line.
    return matches[-1] if matches else None


def find_boxed_spans(text: str) -> tuple[BoxedSpan, ...]:
    if not isinstance(text, str) or not text:
        return ()

    spans: list[BoxedSpan] = []
    cursor = 0
    while cursor < len(text):
        match = _BOXED_START_RE.search(text, cursor)
        if match is None:
            break
        brace = text.find("{", match.start(), match.end())
        if brace < 0:
            break

        depth = 1
        index = brace + 1
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            break

        end = index
        content = text[brace + 1 : end - 1].strip()
        if content:
            spans.append(
                BoxedSpan(
                    content=content,
                    canonical=f"\\boxed{{{content}}}",
                    raw=text[match.start():end],
                    start=match.start(),
                    end=end,
                )
            )
        cursor = end
    return tuple(spans)


def is_strict_answer_only_output(
    text: str,
    region: TextRegion | None = None,
) -> bool:
    if not isinstance(text, str) or not text:
        return False
    answer_region = region or last_complete_answer_region(text)
    if answer_region is None or not answer_region.content:
        return False

    residual = text[: answer_region.region_start] + text[answer_region.region_end :]
    previous = None
    while residual != previous:
        previous = residual
        residual = _EMPTY_THINK_RE.sub("", residual)
    residual = _FORMAT_TAG_RE.sub("", residual)
    residual = _FINAL_ANSWER_LABEL_ONLY_RE.sub("", residual)
    residual = re.sub(r"```(?:[A-Za-z0-9_+.-]+)?", "", residual)
    residual = re.sub(r"[\s`*_#~$\\()\[\]{}<>:;,.!?=+\-|/：，。！？；]+", "", residual)
    return residual == ""


def _decision(value: str, legacy: str, source: str, reason: str) -> CandidateDecision:
    safe_value = value if value or not legacy else legacy
    if safe_value == legacy and value != legacy:
        source = "legacy"
        reason = "candidate_was_empty; used exact legacy fallback"
    return CandidateDecision(
        value=safe_value,
        source=source,
        reason=reason,
        changed=safe_value != legacy,
    )


def select_hipho_answer_content(
    text: str,
    legacy_content: str,
    mode: str | None = None,
) -> CandidateDecision:
    if normalize_answer_extraction_mode(mode) != CONSERVATIVE_ANSWER_EXTRACTION_MODE:
        return _decision(legacy_content, legacy_content, "legacy", "legacy mode")

    region = last_complete_answer_region(text)
    if region is None:
        return _decision(
            legacy_content,
            legacy_content,
            "legacy",
            "no complete answer block; used exact legacy fallback",
        )
    if not region.content:
        return _decision(
            legacy_content,
            legacy_content,
            "legacy",
            "last complete answer block was empty; used exact legacy fallback",
        )
    return _decision(
        region.content,
        legacy_content,
        region.source,
        "selected last non-empty complete answer block",
    )


def select_hle_candidate(
    text: str,
    legacy_candidate: str,
    mode: str | None = None,
) -> CandidateDecision:
    if normalize_answer_extraction_mode(mode) != CONSERVATIVE_ANSWER_EXTRACTION_MODE:
        return _decision(legacy_candidate, legacy_candidate, "legacy", "legacy mode")

    marker_region = last_anchored_final_answer_region(text)
    if marker_region is not None:
        region = marker_region
        reason = "selected last anchored Final Answer region"
    else:
        answer_region = last_complete_answer_region(text)
        if answer_region is None or not is_strict_answer_only_output(text, answer_region):
            return _decision(
                legacy_candidate,
                legacy_candidate,
                "legacy",
                "no high-confidence terminal answer region; used exact legacy fallback",
            )
        region = answer_region
        reason = "selected strict answer-only complete answer block"

    boxed = find_boxed_spans(region.content)
    if len(boxed) < 2:
        return _decision(
            legacy_candidate,
            legacy_candidate,
            "legacy",
            "terminal region had fewer than two boxed answers; used exact legacy fallback",
        )
    if legacy_candidate != boxed[-1].canonical:
        return _decision(
            legacy_candidate,
            legacy_candidate,
            "legacy",
            "terminal boxed answer did not match legacy extraction; used exact legacy fallback",
        )

    candidate = "\n".join(span.canonical for span in boxed)
    return _decision(candidate, legacy_candidate, region.source, reason)


def select_frontierscience_olympiad_candidate(
    text: str,
    legacy_candidate: str,
    mode: str | None = None,
) -> CandidateDecision:
    if normalize_answer_extraction_mode(mode) != CONSERVATIVE_ANSWER_EXTRACTION_MODE:
        return _decision(legacy_candidate, legacy_candidate, "legacy", "legacy mode")

    regions = [
        region
        for region in (
            last_complete_answer_region(text),
            last_anchored_final_answer_region(text),
        )
        if region is not None and region.content
    ]
    if not regions:
        return _decision(
            legacy_candidate,
            legacy_candidate,
            "legacy",
            "no high-confidence terminal answer region; used exact legacy fallback",
        )
    region = max(regions, key=lambda item: item.region_start)
    boxed = find_boxed_spans(region.content)
    legacy_boxed = find_boxed_spans(legacy_candidate)
    if len(boxed) < 2 or not legacy_boxed:
        return _decision(
            legacy_candidate,
            legacy_candidate,
            "legacy",
            "terminal region lacked multiple retained boxed answers; used exact legacy fallback",
        )

    legacy_terminal_box = legacy_boxed[-1].canonical
    if legacy_terminal_box not in {span.canonical for span in boxed}:
        return _decision(
            legacy_candidate,
            legacy_candidate,
            "legacy",
            "terminal region did not retain the legacy boxed answer; used exact legacy fallback",
        )

    return _decision(
        region.content,
        legacy_candidate,
        region.source,
        "selected complete terminal region with multiple boxed answers",
    )


def select_hipho_step_judge_solution(
    output: str,
    reasoning_content: str,
    mode: str | None = None,
) -> str:
    if (
        normalize_hipho_step_judge_solution_mode(mode)
        != HIPHO_STEP_JUDGE_REASONING_FALLBACK_MODE
    ):
        return output
    reasoning = str(reasoning_content or "")
    if not reasoning.strip() or not is_strict_answer_only_output(output):
        return output
    return f"{reasoning.rstrip()}\n\n{output}"
