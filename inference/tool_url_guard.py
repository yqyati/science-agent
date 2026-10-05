import hashlib
import json
import os
import re
import threading
import unicodedata
import zlib
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Callable, Iterator, Mapping, TypeVar
from urllib.parse import unquote, urlparse


TRUE_VALUES = {"1", "true", "yes", "on"}
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BLOCKED_URLS_FILE = os.path.join(PROJECT_DIR, "blocked_urls_keywords_0513.txt")
GUARD_SNAPSHOT_SCHEMA_VERSION = 1
GUARD_CAPTURE_SCHEMA_VERSION = 1
GUARD_EVALUATOR_VERSION = "url-guard-2026-07-23-v1"
GUARD_FORMAT_SERPER_SEARCH_V1 = "serper-search-v1"
GUARD_FORMAT_SERPER_SCHOLAR_V1 = "serper-scholar-v1"
GUARD_FORMAT_NORMALIZED_SEARCH_V1 = "normalized-search-v1"
GUARD_FORMAT_NORMALIZED_SCHOLAR_V1 = "normalized-scholar-v1"
GUARD_FORMAT_VISIT_FINAL_V1 = "visit-final-v1"
SHORT_KEYWORD_MAX_LEN = 4
DATASET_YEAR_SUFFIX_RE = re.compile(r"^(?:\d{4}|\d{2})$")
ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
URL_CANDIDATE_RE = re.compile(
    r"(?i)\b(?:https?://|[a-z][a-z0-9+.-]*://|www\.)[^\s<>\[\]{}\"']+"
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class UrlGuardRule:
    kind: str
    value: str
    label: str = ""
    host: str = ""
    path: str = ""
    query: str = ""
    fragment: str = ""
    compact_value: str = ""
    short_token: bool = False


@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    host: str
    path: str
    query: str = ""
    fragment: str = ""


@dataclass(frozen=True)
class SearchResultGuardDecision:
    blocked: bool
    reason: str = ""


@dataclass(frozen=True)
class UrlGuardMatch:
    reason: str
    rule_kind: str
    rule_label: str


@dataclass(frozen=True)
class GuardPolicySnapshot:
    """Immutable, environment-independent URL guard policy."""

    schema_version: int
    evaluator_version: str
    status: str
    benchmark: str
    rules: tuple[UrlGuardRule, ...]
    rules_sha256: str

    @property
    def enabled(self) -> bool:
        return self.status in {"active", "active_empty", "degraded"}

    @property
    def cacheable(self) -> bool:
        return self.status != "degraded"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "evaluator_version": self.evaluator_version,
            "status": self.status,
            "benchmark": self.benchmark,
            "rules": [_guard_rule_dict(rule) for rule in self.rules],
            "rules_sha256": self.rules_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class GuardContext:
    """A captured guard policy plus non-fingerprinted diagnostic state."""

    snapshot: GuardPolicySnapshot
    source_error: str = ""

    @property
    def enabled(self) -> bool:
        return self.snapshot.enabled

    @property
    def cacheable(self) -> bool:
        return self.snapshot.cacheable

    @property
    def fingerprint(self) -> str:
        return guard_fingerprint(self)


@dataclass(frozen=True)
class GuardUrlScope:
    url: str
    goal: str = ""
    item_index: int = -1

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "goal": self.goal,
            "item_index": self.item_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GuardUrlScope":
        return cls(
            url=str(value.get("url") or ""),
            goal=str(value.get("goal") or ""),
            item_index=int(value.get("item_index", -1)),
        )


@dataclass(frozen=True)
class GuardCheckedValue:
    sequence: int
    stage: str
    value_json: str
    item_index: int = -1
    url_scope: GuardUrlScope | None = None

    @property
    def value(self) -> object:
        return json.loads(self.value_json)

    def to_dict(self) -> dict:
        value = {
            "sequence": self.sequence,
            "stage": self.stage,
            "value_json": self.value_json,
            "item_index": self.item_index,
        }
        if self.url_scope is not None:
            value["url_scope"] = self.url_scope.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GuardCheckedValue":
        scope = value.get("url_scope")
        return cls(
            sequence=int(value.get("sequence", 0)),
            stage=str(value.get("stage") or "value"),
            value_json=str(value.get("value_json") or "null"),
            item_index=int(value.get("item_index", -1)),
            url_scope=GuardUrlScope.from_dict(scope) if isinstance(scope, Mapping) else None,
        )


@dataclass(frozen=True)
class GuardStructuredBasis:
    sequence: int
    tool: str
    stage: str
    formatter: str
    request_json: str
    payload_json: str
    item_index: int = -1
    url_scope: GuardUrlScope | None = None

    @property
    def request(self) -> object:
        return json.loads(self.request_json)

    @property
    def payload(self) -> object:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict:
        value = {
            "sequence": self.sequence,
            "tool": self.tool,
            "stage": self.stage,
            "formatter": self.formatter,
            "request_json": self.request_json,
            "payload_json": self.payload_json,
            "item_index": self.item_index,
        }
        if self.url_scope is not None:
            value["url_scope"] = self.url_scope.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GuardStructuredBasis":
        scope = value.get("url_scope")
        return cls(
            sequence=int(value.get("sequence", 0)),
            tool=str(value.get("tool") or ""),
            stage=str(value.get("stage") or "provider"),
            formatter=str(value.get("formatter") or ""),
            request_json=str(value.get("request_json") or "{}"),
            payload_json=str(value.get("payload_json") or "null"),
            item_index=int(value.get("item_index", -1)),
            url_scope=GuardUrlScope.from_dict(scope) if isinstance(scope, Mapping) else None,
        )


@dataclass(frozen=True)
class GuardReplayEnvelope:
    schema_version: int
    policy_fingerprint: str
    structured_bases: tuple[GuardStructuredBasis, ...]
    checked_values: tuple[GuardCheckedValue, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_fingerprint": self.policy_fingerprint,
            "structured_bases": [item.to_dict() for item in self.structured_bases],
            "checked_values": [item.to_dict() for item in self.checked_values],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_bytes(self, *, compression_level: int = 6) -> bytes:
        level = max(0, min(9, int(compression_level)))
        return zlib.compress(self.to_json().encode("utf-8"), level)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GuardReplayEnvelope":
        schema_version = int(value.get("schema_version", 0))
        if schema_version != GUARD_CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported guard replay envelope schema: {schema_version}")
        structured = value.get("structured_bases")
        checked = value.get("checked_values")
        return cls(
            schema_version=schema_version,
            policy_fingerprint=str(value.get("policy_fingerprint") or ""),
            structured_bases=tuple(
                GuardStructuredBasis.from_dict(item)
                for item in structured or ()
                if isinstance(item, Mapping)
            ),
            checked_values=tuple(
                GuardCheckedValue.from_dict(item)
                for item in checked or ()
                if isinstance(item, Mapping)
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "GuardReplayEnvelope":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ValueError("guard replay envelope must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def from_bytes(cls, value: bytes) -> "GuardReplayEnvelope":
        try:
            payload = zlib.decompress(value)
        except zlib.error as exc:
            raise ValueError("invalid compressed guard replay envelope") from exc
        return cls.from_json(payload.decode("utf-8"))


class GuardCaptureCollector:
    """Thread-safe collector shared by copied ContextVar contexts."""

    def __init__(self, context: GuardContext):
        self.context = context
        self._lock = threading.Lock()
        self._sequence = 0
        self._structured_bases: list[GuardStructuredBasis] = []
        self._checked_values: list[GuardCheckedValue] = []

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def record_checked_value(
        self,
        value: object,
        *,
        stage: str,
        item_index: int,
        url_scope: GuardUrlScope | None,
    ) -> None:
        value_json = _guard_value_json(value)
        with self._lock:
            self._checked_values.append(
                GuardCheckedValue(
                    sequence=self._next_sequence(),
                    stage=str(stage or "value"),
                    value_json=value_json,
                    item_index=item_index,
                    url_scope=url_scope,
                )
            )

    def record_structured_basis(
        self,
        tool: str,
        payload: object,
        *,
        stage: str,
        formatter: str,
        request: object,
        item_index: int,
        url_scope: GuardUrlScope | None,
    ) -> None:
        request_json = _guard_value_json(request)
        payload_json = _guard_value_json(payload)
        with self._lock:
            self._structured_bases.append(
                GuardStructuredBasis(
                    sequence=self._next_sequence(),
                    tool=_canonical_guard_tool(tool),
                    stage=str(stage or "provider"),
                    formatter=str(formatter or ""),
                    request_json=request_json,
                    payload_json=payload_json,
                    item_index=item_index,
                    url_scope=url_scope,
                )
            )

    def envelope(self) -> GuardReplayEnvelope:
        with self._lock:
            def stable_key(item):
                index = item.item_index if item.item_index >= 0 else 2**31 - 1
                return index, item.sequence

            structured = tuple(
                replace(item, sequence=index)
                for index, item in enumerate(
                    sorted(self._structured_bases, key=stable_key),
                    start=1,
                )
            )
            checked = tuple(
                replace(item, sequence=index)
                for index, item in enumerate(
                    sorted(self._checked_values, key=stable_key),
                    start=1,
                )
            )
        return GuardReplayEnvelope(
            schema_version=GUARD_CAPTURE_SCHEMA_VERSION,
            # Raw replay material is guard-independent and can be evaluated
            # under many policy fingerprints over its lifetime.
            policy_fingerprint="",
            structured_bases=structured,
            checked_values=checked,
        )


_ACTIVE_GUARD_CONTEXT: ContextVar[GuardContext | None] = ContextVar(
    "science_agent_active_guard_context",
    default=None,
)
_ACTIVE_GUARD_COLLECTOR: ContextVar[GuardCaptureCollector | None] = ContextVar(
    "science_agent_active_guard_collector",
    default=None,
)
_ACTIVE_GUARD_URL_SCOPE: ContextVar[GuardUrlScope | None] = ContextVar(
    "science_agent_active_guard_url_scope",
    default=None,
)
_ACTIVE_GUARD_ITEM_INDEX: ContextVar[int] = ContextVar(
    "science_agent_active_guard_item_index",
    default=-1,
)


def _guard_rule_dict(rule: UrlGuardRule) -> dict:
    return {
        "kind": rule.kind,
        "value": rule.value,
        "label": rule.label,
        "host": rule.host,
        "path": rule.path,
        "query": rule.query,
        "fragment": rule.fragment,
        "compact_value": rule.compact_value,
        "short_token": rule.short_token,
    }


def _guard_rule_from_dict(value: Mapping[str, object]) -> UrlGuardRule:
    return UrlGuardRule(
        kind=str(value.get("kind") or ""),
        value=str(value.get("value") or ""),
        label=str(value.get("label") or ""),
        host=str(value.get("host") or ""),
        path=str(value.get("path") or ""),
        query=str(value.get("query") or ""),
        fragment=str(value.get("fragment") or ""),
        compact_value=str(value.get("compact_value") or ""),
        short_token=bool(value.get("short_token", False)),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _rules_digest(rules: tuple[UrlGuardRule, ...]) -> str:
    payload = [_guard_rule_dict(rule) for rule in rules]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def guard_policy_snapshot_from_dict(value: Mapping[str, object]) -> GuardPolicySnapshot:
    schema_version = int(value.get("schema_version", 0))
    if schema_version != GUARD_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported guard snapshot schema: {schema_version}")
    rules_value = value.get("rules")
    rules = tuple(
        _guard_rule_from_dict(item)
        for item in rules_value or ()
        if isinstance(item, Mapping)
    )
    digest = _rules_digest(rules)
    stored_digest = str(value.get("rules_sha256") or "")
    if stored_digest and stored_digest != digest:
        raise ValueError("guard snapshot rules digest mismatch")
    status = str(value.get("status") or "")
    if status not in {"disabled", "active", "active_empty", "degraded"}:
        raise ValueError(f"invalid guard snapshot status: {status!r}")
    return GuardPolicySnapshot(
        schema_version=schema_version,
        evaluator_version=str(value.get("evaluator_version") or ""),
        status=status,
        benchmark=str(value.get("benchmark") or ""),
        rules=rules,
        rules_sha256=digest,
    )


def guard_context_from_snapshot(
    snapshot: GuardPolicySnapshot | Mapping[str, object],
) -> GuardContext:
    if isinstance(snapshot, Mapping):
        snapshot = guard_policy_snapshot_from_dict(snapshot)
    return GuardContext(snapshot=snapshot)


def guard_fingerprint(context: GuardContext | GuardPolicySnapshot) -> str:
    snapshot = context.snapshot if isinstance(context, GuardContext) else context
    if not snapshot.cacheable:
        return ""
    payload = {
        "schema_version": snapshot.schema_version,
        "evaluator_version": snapshot.evaluator_version,
        "status": snapshot.status,
        "benchmark": snapshot.benchmark,
        "rules": [_guard_rule_dict(rule) for rule in snapshot.rules],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_guard_tool(tool: object) -> str:
    normalized = str(tool or "").strip()
    aliases = {
        "search": "search",
        "google_search-google_search": "search",
        "scholar": "google_scholar",
        "google_scholar": "google_scholar",
        "visit": "visit",
        "read-page-server-read_page": "visit",
    }
    return aliases.get(normalized, normalized)


def _guard_json_normalize(value: object, seen: set[int] | None = None) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        try:
            json.dumps(value, allow_nan=False)
        except ValueError:
            return str(value)
        return value

    seen = seen if seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        return "<recursive>"
    seen.add(object_id)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return _guard_json_normalize(asdict(value), seen)
        if isinstance(value, Mapping):
            return {
                str(key): _guard_json_normalize(item, seen)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_guard_json_normalize(item, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [_guard_json_normalize(item, seen) for item in value]
            return sorted(normalized, key=_canonical_json)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    finally:
        seen.discard(object_id)


def _guard_value_json(value: object) -> str:
    return _canonical_json(_guard_json_normalize(value))


def current_guard_context() -> GuardContext | None:
    return _ACTIVE_GUARD_CONTEXT.get()


def current_guard_collector() -> GuardCaptureCollector | None:
    return _ACTIVE_GUARD_COLLECTOR.get()


@contextmanager
def guard_context_scope(context: GuardContext) -> Iterator[GuardContext]:
    token = _ACTIVE_GUARD_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_GUARD_CONTEXT.reset(token)


@contextmanager
def guard_capture_scope(
    context: GuardContext | None = None,
) -> Iterator[GuardCaptureCollector]:
    resolved_context = context or current_guard_context() or capture_guard_context()
    collector = GuardCaptureCollector(resolved_context)
    context_token = _ACTIVE_GUARD_CONTEXT.set(resolved_context)
    collector_token = _ACTIVE_GUARD_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_GUARD_COLLECTOR.reset(collector_token)
        _ACTIVE_GUARD_CONTEXT.reset(context_token)


@contextmanager
def guard_capture_item_scope(item_index: int) -> Iterator[int]:
    token = _ACTIVE_GUARD_ITEM_INDEX.set(int(item_index))
    try:
        yield int(item_index)
    finally:
        _ACTIVE_GUARD_ITEM_INDEX.reset(token)


@contextmanager
def guard_capture_url_scope(
    url: object,
    *,
    goal: object = "",
    item_index: int | None = None,
) -> Iterator[GuardUrlScope]:
    resolved_index = _ACTIVE_GUARD_ITEM_INDEX.get() if item_index is None else int(item_index)
    scope = GuardUrlScope(
        url=str(url or ""),
        goal=str(goal or ""),
        item_index=resolved_index,
    )
    token = _ACTIVE_GUARD_URL_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE_GUARD_URL_SCOPE.reset(token)


def bind_guard_capture_context(callback: Callable[..., _T]) -> Callable[..., _T]:
    """Bind the current ContextVars for one future/thread invocation."""

    context: Context = copy_context()

    def bound(*args, **kwargs):
        return context.run(callback, *args, **kwargs)

    return bound


def capture_structured_basis(
    tool: str,
    payload: object,
    *,
    request: object | None = None,
    stage: str = "provider",
    formatter: str = "",
    item_index: int | None = None,
) -> object:
    """Record a replayable provider/formatter input and return it unchanged."""

    collector = current_guard_collector()
    if collector is None:
        return payload
    resolved_index = _ACTIVE_GUARD_ITEM_INDEX.get() if item_index is None else int(item_index)
    collector.record_structured_basis(
        tool,
        payload,
        stage=stage,
        formatter=formatter,
        request=request if request is not None else {},
        item_index=resolved_index,
        url_scope=_ACTIVE_GUARD_URL_SCOPE.get(),
    )
    return payload


def _capture_checked_value(value: object, *, stage: str) -> None:
    collector = current_guard_collector()
    if collector is None:
        return
    collector.record_checked_value(
        value,
        stage=stage,
        item_index=_ACTIVE_GUARD_ITEM_INDEX.get(),
        url_scope=_ACTIVE_GUARD_URL_SCOPE.get(),
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _canonical_benchmark_name(name: str) -> str:
    normalized = str(name or "").strip()
    aliases = {
        "hle_subject": "hle_all",
        "hle_all_mini": "hle_all",
        "hle_all_mini_100": "hle_all",
        "hle_all_mm": "hle_all",
        "hle_all_text": "hle_all",
        "hipho_mm": "hipho",
        "cmt_benchmark": "cmt",
    }
    return aliases.get(normalized, normalized)


def _url_guard_enabled_from_env(env: Mapping[str, str]) -> bool:
    if str(env.get("AGENT_EVAL_URL_GUARD", "")).strip().lower() not in TRUE_VALUES:
        return False

    benchmark_allowlist = _split_csv(str(env.get("AGENT_EVAL_URL_GUARD_BENCHMARKS", "")))
    if not benchmark_allowlist:
        return True

    benchmark = _canonical_benchmark_name(str(env.get("AGENT_EVAL_BENCHMARK", "")))
    allowed = {_canonical_benchmark_name(item) for item in benchmark_allowlist}
    return benchmark in allowed


def url_guard_enabled(context: GuardContext | None = None) -> bool:
    resolved_context = context or current_guard_context()
    if resolved_context is not None:
        return resolved_context.enabled
    return _url_guard_enabled_from_env(os.environ)


def _blocked_urls_file(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return str(values.get("AGENT_EVAL_BLOCKED_URLS_FILE", "")).strip() or DEFAULT_BLOCKED_URLS_FILE


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = unquote(text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value)


def _looks_like_dataset_token(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9]*", value)) and any(char.isalpha() for char in value)


def _dataset_year_candidate_matches(keyword: str, candidate: str) -> bool:
    if not candidate.startswith(keyword):
        return False
    return bool(DATASET_YEAR_SUFFIX_RE.fullmatch(candidate[len(keyword) :]))


def _dataset_year_suffix_matches(rule: UrlGuardRule, text: str, max_parts: int) -> bool:
    keywords = []
    if _looks_like_dataset_token(rule.value):
        keywords.append(rule.value)
    if rule.compact_value and rule.compact_value not in keywords and _looks_like_dataset_token(rule.compact_value):
        keywords.append(rule.compact_value)
    if not keywords:
        return False

    parts = _tokens(text)
    for keyword in keywords:
        for start in range(len(parts)):
            joined = ""
            for part in parts[start : start + max_parts]:
                joined += part
                if _dataset_year_candidate_matches(keyword, joined):
                    return True
                if len(joined) > len(keyword) + 4:
                    break
    return False


def _canonical_host(host: str) -> str:
    normalized = _normalize_text(host).split("@")[-1].split(":")[0]
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _clean_url_candidate(value: object) -> str:
    text = _normalize_text(value)
    return text.rstrip(".,;:!?)]}>")


def _parse_url(value: object) -> ParsedUrl | None:
    normalized = _clean_url_candidate(value)
    if not normalized:
        return None

    try:
        parsed = urlparse(normalized)
        if not parsed.netloc and "://" not in normalized:
            parsed = urlparse("//" + normalized)
    except ValueError:
        return None

    host = _canonical_host(parsed.netloc)
    if not host or "." not in host and parsed.scheme not in {"huggingface"}:
        return None

    scheme = (parsed.scheme or "").casefold()
    if scheme in {"http", "https"}:
        scheme = "web"

    return ParsedUrl(
        scheme=scheme,
        host=host,
        path=parsed.path or "/",
        query=parsed.query,
        fragment=parsed.fragment,
    )


def _looks_like_url_rule(line: str, parsed: ParsedUrl | None) -> bool:
    if parsed is None:
        return False
    normalized = _normalize_text(line)
    return "://" in normalized or normalized.startswith("www.") or bool(parsed.host)


def _benchmark_rule_sections(benchmark: str | None = None) -> set[str]:
    if benchmark is None:
        benchmark = str(os.environ.get("AGENT_EVAL_BENCHMARK", "") or "").strip()
    canonical = _canonical_benchmark_name(benchmark)
    if not canonical:
        return set()

    sections = {canonical}
    if canonical == "hle_all":
        sections.update(
            {"hle_subject", "hle_all", "hle_all_mini", "hle_all_mini_100", "hle_all_mm", "hle_all_text"}
        )
    if canonical == "cmt":
        sections.update({"cmt", "cmt_benchmark"})
    return sections


def _section_is_active(section: str | None, active_sections: set[str], saw_section: bool) -> bool:
    if section is None:
        return not saw_section
    if not active_sections:
        return False
    return _canonical_benchmark_name(section) in active_sections


def _make_keyword_rule(line: str) -> UrlGuardRule | None:
    label = str(line or "").strip()
    normalized = _normalize_text(line)
    if not normalized:
        return None

    arxiv_match = ARXIV_ID_RE.fullmatch(normalized)
    if arxiv_match:
        return UrlGuardRule("arxiv", arxiv_match.group(1), label=label)

    compact = _compact_alnum(normalized)
    short_token = bool(re.fullmatch(r"[a-z0-9]+", normalized)) and len(normalized) <= SHORT_KEYWORD_MAX_LEN
    return UrlGuardRule(
        "keyword",
        normalized,
        label=label,
        compact_value=compact,
        short_token=short_token,
    )


def _make_rule(line: str) -> UrlGuardRule | None:
    label = str(line or "").strip()
    normalized = _normalize_text(line)
    if not normalized:
        return None

    if normalized.startswith("domain:"):
        domain = _canonical_host(normalized.split(":", 1)[1])
        if domain:
            return UrlGuardRule("domain", domain, label=label)
        return None

    arxiv_match = ARXIV_ID_RE.fullmatch(normalized)
    if arxiv_match:
        return UrlGuardRule("arxiv", arxiv_match.group(1), label=label)

    parsed = _parse_url(normalized)
    if _looks_like_url_rule(normalized, parsed):
        assert parsed is not None
        return UrlGuardRule(
            "url",
            normalized,
            label=label,
            host=parsed.host,
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment,
        )

    return _make_keyword_rule(line)


def _parse_rules_text(text: str, *, benchmark: str) -> list[UrlGuardRule]:
    rules: list[UrlGuardRule] = []
    current_section: str | None = None
    active_sections = _benchmark_rule_sections(benchmark)
    saw_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        section_match = re.fullmatch(r"\[([^\]]+)\]", line)
        if section_match:
            saw_section = True
            current_section = _canonical_benchmark_name(section_match.group(1).strip())
            continue

        if not _section_is_active(current_section, active_sections, saw_section):
            continue

        rule = _make_rule(line)
        if rule is not None:
            rules.append(rule)
    return rules


def _make_policy_snapshot(
    *,
    status: str,
    benchmark: str,
    rules: tuple[UrlGuardRule, ...] = (),
) -> GuardPolicySnapshot:
    return GuardPolicySnapshot(
        schema_version=GUARD_SNAPSHOT_SCHEMA_VERSION,
        evaluator_version=GUARD_EVALUATOR_VERSION,
        status=status,
        benchmark=_canonical_benchmark_name(benchmark),
        rules=rules,
        rules_sha256=_rules_digest(rules),
    )


def capture_guard_context(env: Mapping[str, str] | None = None) -> GuardContext:
    """Capture one stable guard policy without retaining env or file references."""

    values = dict(os.environ if env is None else env)
    benchmark = _canonical_benchmark_name(str(values.get("AGENT_EVAL_BENCHMARK", "")))
    if not _url_guard_enabled_from_env(values):
        return GuardContext(
            snapshot=_make_policy_snapshot(
                status="disabled",
                benchmark=benchmark,
            )
        )

    path = _blocked_urls_file(values)
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = os.stat(path)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if (
            before_identity
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or before_identity
            != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        ):
            raise RuntimeError("guard rules file changed while it was captured")
        text = payload.decode("utf-8")
        rules = tuple(_parse_rules_text(text, benchmark=benchmark))
    except (OSError, UnicodeError, RuntimeError) as exc:
        return GuardContext(
            snapshot=_make_policy_snapshot(
                status="degraded",
                benchmark=benchmark,
            ),
            source_error=f"{type(exc).__name__}: {exc}",
        )

    return GuardContext(
        snapshot=_make_policy_snapshot(
            status="active" if rules else "active_empty",
            benchmark=benchmark,
            rules=rules,
        )
    )


def _load_rules() -> list[UrlGuardRule]:
    active_context = current_guard_context()
    if active_context is not None:
        return list(active_context.snapshot.rules)

    path = _blocked_urls_file()
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _parse_rules_text(
                handle.read(),
                benchmark=str(os.environ.get("AGENT_EVAL_BENCHMARK", "")),
            )
    except OSError as exc:
        print(f"[url-guard] Warning: failed to read AGENT_EVAL_BLOCKED_URLS_FILE={path!r}: {exc}")
        return []


def _url_rule_path_matches(rule_path: str, parsed_path: str) -> bool:
    rule_prefix = (rule_path or "/").rstrip("/") or "/"
    parsed_normalized = (parsed_path or "/").rstrip("/") or "/"
    if parsed_normalized == rule_prefix:
        return True
    if rule_prefix == "/":
        return False
    return parsed_normalized.startswith(rule_prefix + "/")


def _url_matches_rule(parsed: ParsedUrl, rule: UrlGuardRule) -> bool:
    if rule.kind == "domain":
        return parsed.host == rule.value or parsed.host.endswith("." + rule.value)

    if rule.kind != "url":
        return False

    if rule.host != parsed.host:
        return False
    if not _url_rule_path_matches(rule.path, parsed.path):
        return False
    if rule.query and rule.query != parsed.query:
        return False
    if rule.fragment and rule.fragment != parsed.fragment:
        return False
    return True


def _extract_url_candidates(text: str) -> list[str]:
    candidates = []
    parsed = _parse_url(text)
    if parsed is not None:
        candidates.append(text)
    candidates.extend(match.group(0) for match in URL_CANDIDATE_RE.finditer(text))
    return candidates


def _arxiv_ids(value: str) -> set[str]:
    return {match.group(1) for match in ARXIV_ID_RE.finditer(value)}


def _short_keyword_matches(rule: UrlGuardRule, text: str) -> bool:
    if re.search(rf"(?<![a-z0-9]){re.escape(rule.value)}(?![a-z0-9])", text):
        return True
    if _dataset_year_suffix_matches(rule, text, max_parts=3):
        return True

    if not rule.compact_value:
        return False

    parts = _tokens(text)
    for start in range(len(parts)):
        joined = ""
        for part in parts[start : start + 3]:
            joined += part
            if joined == rule.compact_value:
                return True
            if len(joined) >= len(rule.compact_value):
                break
    return False


def _keyword_matches(rule: UrlGuardRule, text: str) -> bool:
    if rule.short_token:
        return _short_keyword_matches(rule, text)

    if rule.value and rule.value in text:
        return True
    if _dataset_year_suffix_matches(rule, text, max_parts=6):
        return True

    if not rule.compact_value or len(rule.compact_value) < 3:
        return False

    parts = _tokens(text)
    for start in range(len(parts)):
        joined = ""
        for part in parts[start : start + 6]:
            joined += part
            if joined == rule.compact_value:
                return True
            if len(joined) >= len(rule.compact_value):
                break
    return False


def _rule_label(rule: UrlGuardRule) -> str:
    return rule.label or rule.value


def _string_guard_matches(
    value: object,
    *,
    rules: tuple[UrlGuardRule, ...] | list[UrlGuardRule] | None = None,
) -> list[UrlGuardMatch]:
    text = _normalize_text(value)
    if not text:
        return []

    resolved_rules = _load_rules() if rules is None else rules
    matches: list[UrlGuardMatch] = []

    for candidate in _extract_url_candidates(text):
        parsed = _parse_url(candidate)
        if parsed is None:
            continue
        for rule in resolved_rules:
            if _url_matches_rule(parsed, rule):
                matches.append(UrlGuardMatch("url_rule", rule.kind, _rule_label(rule)))

    text_arxiv_ids = _arxiv_ids(text)
    for rule in resolved_rules:
        if rule.kind == "arxiv" and rule.value in text_arxiv_ids:
            matches.append(UrlGuardMatch("arxiv_rule", rule.kind, _rule_label(rule)))
        if rule.kind == "keyword" and _keyword_matches(rule, text):
            matches.append(UrlGuardMatch("keyword_rule", rule.kind, _rule_label(rule)))

    return matches


def _string_guard_decision(
    value: object,
    *,
    rules: tuple[UrlGuardRule, ...] | list[UrlGuardRule] | None = None,
) -> SearchResultGuardDecision:
    matches = _string_guard_matches(value, rules=rules)
    if matches:
        return SearchResultGuardDecision(True, matches[0].reason)
    return SearchResultGuardDecision(False)


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)


def _value_guard_decision_with_rules(
    value: object,
    rules: tuple[UrlGuardRule, ...] | list[UrlGuardRule],
) -> SearchResultGuardDecision:
    for text in _iter_strings(value):
        decision = _string_guard_decision(text, rules=rules)
        if decision.blocked:
            return decision
    return SearchResultGuardDecision(False)


def evaluate_guard_value(
    value: object,
    context: GuardContext,
) -> SearchResultGuardDecision:
    """Pure evaluation against an immutable context; does not capture values."""

    if not context.enabled:
        return SearchResultGuardDecision(False)
    return _value_guard_decision_with_rules(value, context.snapshot.rules)


def value_guard_decision(
    value: object,
    *,
    context: GuardContext | None = None,
    stage: str = "value",
) -> SearchResultGuardDecision:
    # Capture before the enabled check so a non-guard raw execution still
    # contains enough checked material for a later guarded cache replay.
    _capture_checked_value(value, stage=stage)
    resolved_context = context or current_guard_context()
    if resolved_context is not None:
        return evaluate_guard_value(value, resolved_context)
    if not url_guard_enabled():
        return SearchResultGuardDecision(False)
    return _value_guard_decision_with_rules(value, _load_rules())


def value_guard_matches(
    value: object,
    *,
    context: GuardContext | None = None,
) -> list[UrlGuardMatch]:
    resolved_context = context or current_guard_context()
    if resolved_context is not None:
        if not resolved_context.enabled:
            return []
        rules: tuple[UrlGuardRule, ...] | list[UrlGuardRule] = resolved_context.snapshot.rules
    else:
        if not url_guard_enabled():
            return []
        rules = _load_rules()

    if not rules:
        return []

    matches: list[UrlGuardMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for text in _iter_strings(value):
        for match in _string_guard_matches(text, rules=rules):
            key = (match.reason, match.rule_kind, match.rule_label)
            if key in seen:
                continue
            seen.add(key)
            matches.append(match)
    return matches


def active_guard_rule_labels(context: GuardContext | None = None) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    resolved_context = context or current_guard_context()
    rules = resolved_context.snapshot.rules if resolved_context is not None else _load_rules()
    for rule in rules:
        label = _rule_label(rule)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def is_value_blocked(
    value: object,
    *,
    context: GuardContext | None = None,
    stage: str = "value",
) -> bool:
    return value_guard_decision(value, context=context, stage=stage).blocked


def is_text_blocked(
    text: object,
    *,
    context: GuardContext | None = None,
    stage: str = "text",
) -> bool:
    return is_value_blocked(text, context=context, stage=stage)


def is_url_blocked(
    url: str,
    *,
    context: GuardContext | None = None,
    stage: str = "request_url",
) -> bool:
    return is_value_blocked(url, context=context, stage=stage)


def search_result_guard_decision(
    result: dict,
    *,
    context: GuardContext | None = None,
    stage: str = "provider",
) -> SearchResultGuardDecision:
    if not isinstance(result, dict):
        return SearchResultGuardDecision(False)
    return value_guard_decision(result, context=context, stage=stage)


def search_result_guard_reason(
    result: dict,
    *,
    context: GuardContext | None = None,
) -> str:
    return search_result_guard_decision(result, context=context).reason


def is_search_result_blocked(
    result: dict,
    *,
    context: GuardContext | None = None,
    stage: str = "provider",
) -> bool:
    return search_result_guard_decision(result, context=context, stage=stage).blocked


def blocked_search_message(query: str | None = None) -> str:
    query_text = str(query or "").strip().replace("\n", " ")
    if query_text:
        return (
            f"Search completed, but all retrieved results for query {query_text!r} "
            "were excluded by the evaluation URL guard for this benchmark. Do not "
            "try to bypass this restriction or repeatedly search for the same "
            "public solution source. Continue using the problem statement, "
            "calculations, and any non-blocked sources/tools."
        )

    return (
        "Search completed, but all retrieved results were excluded by the evaluation "
        "URL guard for this benchmark. Do not try to bypass this restriction or "
        "repeatedly search for the same public solution source. Continue using the "
        "problem statement, calculations, and any non-blocked sources/tools."
    )


def partial_search_note() -> str:
    return "Note: Some search results were excluded by the evaluation URL guard."


def blocked_visit_message() -> str:
    return (
        "[read_page] Blocked by evaluation URL guard: this URL is not accessible "
        "in this benchmark. Do not attempt to bypass this restriction; use the "
        "problem statement, calculations, or non-blocked sources instead."
    )


def blocked_python_message() -> str:
    return (
        "[PythonInterpreter] Blocked by evaluation URL/keyword guard: this Python "
        "tool request or output is not accessible in this benchmark. Do not attempt "
        "to bypass this restriction; use the problem statement, calculations, or "
        "non-blocked sources/tools instead."
    )
