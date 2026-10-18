# Sandbox backend

This harness copy uses one Sandbox per Agent run. Python, search, visit, and
readpage summary HTTP requests reuse that instance. The gateway mode is opt-in;
other backends keep their existing behavior.

## Multiple final answers

ReAct preserves all boxed answers in an explicit final `Final Answer:` region,
including when the model omits `<answer>` tags. It retains the region marker so
the downstream HLE selector does not fall back to the last box. Multiple boxes
inside an explicit `<answer>` block receive the same marker when needed.
Single answers and unmarked last-box fallbacks retain their existing behavior.

`tests/test_terminal_answer_extraction.py` covers the ReAct-to-HLE extraction
chain using the actual adapter function when the adjacent Molt checkout exists.
Changing extraction does not update already loaded processes or saved grades;
existing runs require extraction from saved full responses and rejudging.

## Local configuration and login

`sandbox.local.env` stores local API settings and is ignored by Git. If it does
not exist, copy `sandbox.local.env.example` and fill `API_KEY` and
`SUMMARY_MODEL_NAME`. `API_BASE` is the OpenAI-compatible base URL including
`/v1`, for example `http://35.220.164.252:3888/v1`. Configure `SERPER_API_KEY`
for real search and optional `JINA_API_KEY` for the page reader.

The approved resource is `10.12.111.143/ailab-ma4agismall`.
The local configuration explicitly sets
`SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION=allow` for online tools.
The client expresses this as default-deny with explicit allow rules for
`0.0.0.0/0` and `::/0`. This avoids the platform's invalid Cilium policy
generation for default-allow with an empty egress list.
Without this setting, the platform chooses the default egress policy.

```bash
python3 platform_ad_auth.py -- bash
bash run_sandbox_harness.sh python3 sandbox_smoke.py
bash run_sandbox_harness.sh python3 sandbox_smoke.py --llm
bash run_sandbox_harness.sh python3 sandbox_smoke.py --llm --tools
```

`sandbox_smoke.py` retries the whole preflight up to three times by default,
waiting 5 then 15 seconds after failed checks. Each attempt cleans up its
Sandbox before creating a fresh instance and repeats all requested checks.
Use `--attempts 1` to disable whole-preflight retries. Missing configuration
fails before any attempt; unconfirmed cleanup stops without another instance.
Evaluation and training launchers inherit this behavior through the smoke script.

Login once before starting the workload. Child Agents inherit the gateway Token;
creating an instance does not require another login. The platform documentation
specifies a 15-day Token lifetime; reauthenticate after expiry.

For rjob evaluation, save the Token in the shared local config before
submission (the file is ignored by Git and saved with mode 600):

```bash
python3 platform_ad_auth.py --save-token sandbox.local.env
```

Inside the rjob container, launch the evaluation entrypoint through
`bash /absolute/project/path/run_sandbox_harness.sh <evaluation command>`.
The wrapper exports the saved `OPEN_SANDBOX_GATEWAY_TOKEN`; no interactive
login is needed in the job. The same shared path must be mounted in the job.
Repeat the save command before expiry. The AD password is never saved.

## Runtime

Evaluation and training tool Sandboxes default to
`registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:science-agent-cpu-20261010`.
The gateway, launcher, and example configuration use this image; an explicit
`SCIENCE_AGENT_SANDBOX_IMAGE` overrides it. Live platform validation passed
scientific computation, persistent Python, page storage/cache, web and Scholar
search, image upload/crop/plot, and cleanup. This setting selects the tool
Sandbox image; the GPU job has its own runtime image.

Instances are created lazily with ports 44772 (command), 18080 (Python), and
18081 (HTTP). Worker files are uploaded automatically to `/tmp` through the
command endpoint. No custom image build is needed. Both workers must pass
`/health` before use; initialization errors are reported without silently
switching to single-shot Python or host HTTP.

After the management API reports Running, startup also waits for the command
service's GET `/ping` to return HTTP 200 before bootstrap uploads or commands.
Transient readiness failures, including proxy HTTP 403, wait within the startup
timeout. A command POST receiving HTTP 403 still fails without replay.

The shared creation endpoint retries HTTP 403, 429, 500, 502, 503, 504 and
URL/timeout errors up to four total attempts, waiting 1, 5, then 10 seconds.
Other HTTP errors and successful responses missing an instance ID fail without
another creation attempt. This policy applies only to creating an instance;
it does not replay command or tool POSTs. Caller-level retries (search or judge)
can start another bounded creation cycle after these attempts are exhausted.
Server-side idempotency has not been established: a lost creation response can
leave an instance whose ID is unknown and cannot be explicitly cleaned up by
this client. Such an instance relies on the requested platform lifetime.

`QwenReactAgent._run` has a gateway-only run scope and deletes the instance in
`finally`. Direct tool callers must invoke `close_sandbox_http_run()` or
`close_shared_sandbox_run()` in `finally`. Python state persists within a run.
The HTTP transport preserves upstream HTTP error status and body.

## Validation status

On October 8, live create/command/delete and persistent Python (42 then 41)
passed. On October 9, offline regression checks cover run isolation, shared
Python/HTTP use, cleanup, HTTP error bodies, and Sandbox summary routing.
Live 18081 and persistent Python passed. Public DNS fallback and TCP access to
the LLM API passed. TLS verification for google.serper.dev and r.jina.ai passed;
example.com returned a Fortinet-issued certificate and failed verification.
The live HTTP smoke through `https://r.jina.ai/https://example.com` returned
200 with the expected page content. Python persistence, shared run, and cleanup
also passed on instance `cprod3-z67en409ad2vdm68ecya`. A later live smoke on
`cprod2-qt0dpqwz1wtdy20gelhn` also passed the summary LLM request using
`bailian/deepseek-v4.1-flash`, shared run, and cleanup. A prior LLM attempt
returned HTTP 503; retry succeeded without code changes. Full Search/Visit
tool calls and the rjob evaluation entrypoint remain to be verified. The current shell has no gateway Token and the local
summary model name is empty. The current Python environment also lacks
`qwen_agent`, required for the optional real Search/Visit test.

## DNS fallback

Live diagnostics showed the configured resolver `169.254.25.10` timed out,
while public-IP TCP and the LLM API TCP connection worked. The local setting
`SCIENCE_AGENT_SANDBOX_DNS_SERVERS=1.1.1.1,8.8.8.8` probes those DNS servers
and updates only the new instance's `/etc/resolv.conf` with working servers.
If none answer, initialization fails and the instance is cleaned up. Remove
this setting to use the platform resolver, especially for Kubernetes service
names requiring cluster DNS. No host DNS configuration is changed.

## TLS certificates

`SCIENCE_AGENT_SANDBOX_CA_BUNDLE=auto` combines the host Python CA bundle,
system CA bundle, and certifi certificates, removes duplicates, validates the
bundle, and uploads it to the new Sandbox. Workers use it through
`SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE`; HTTPS certificate verification
remains enabled. An explicit PEM file path can be used instead of `auto`.
Live HTTPS still needs smoke validation; a platform TLS interception CA may
need to be supplied if it is absent from all host bundles.
`--diagnose-network` also inspects the TLS certificate chains for example.com,
google.serper.dev, and r.jina.ai using openssl. It reports certificate subjects,
issuers, fingerprints, and verification errors without sending HTTP requests
or credentials. This distinguishes a missing platform CA from an incomplete
public website certificate chain; it does not disable tool TLS verification.

The default HTTP smoke URL is now `https://r.jina.ai/https://example.com`,
matching the page-reader route. A passing result validates this route only.
Use `--http-url https://example.com` to retest direct website access, which
still requires the platform's trusted Fortinet CA (issuer FG120GTK26007524).
Do not trust a certificate fetched from the failing connection as a new root;
obtain the CA from the platform administrator. Other directly accessed sites
may also be subject to the same platform TLS inspection.

## Vision backend

With `PYTHON_SESSION_BACKEND=gateway_sandbox`, VisualPythonInterpreter now
uses the same run-scoped Sandbox/Python worker as Python, search, and visit.
Input images are uploaded to relative `input_images/` paths and unchanged
files are not reuploaded. PIL images displayed with `display`, matplotlib
`plt.show()`, and final-expression images return PNG data to the existing
Vision response formatter, preserving image deduplication/resizing and local
display artifact saving. The original SandboxFusion branch remains for other
backends. No checkpoint/replay is used within a persistent gateway run.

```bash
bash run_sandbox_harness.sh python3 sandbox_smoke.py --vision --llm
```

The first live Vision smoke found matplotlib absent in the default image.
Before its first Vision call, each instance now checks Pillow, matplotlib,
and IPython and installs only missing packages using its Python pip. The
worker validates imports before executing user code. Disable with
`SCIENCE_AGENT_SANDBOX_INSTALL_VISION_DEPS=false`; a package mirror can be
configured via `SCIENCE_AGENT_SANDBOX_PIP_INDEX_URL`. Installation is bounded
to 300 seconds and errors are surfaced. For large rjob evaluations, preinstall
these packages in the image to avoid repeated downloads for each instance. Offline crop/plot/image
return, persistent state, upload deduplication, and path validation passed.
The actual Sandbox image still requires the live Vision smoke above.

The sandbox launcher defaults to `SCIENCE_AGENT_TOOLS=search,visit,python,google_scholar`. Vision is excluded unless explicitly requested. The Agent reads this setting when no explicit function_list is supplied. `sandbox_smoke.py --vision` is an explicit diagnostic and does not enable Vision for evaluation.

## Isolated Molt evaluation smoke

Use `labs-molt/examples/scripts/rjob/submit_hle3_base_sandbox_smoke.sh`.
It submits four GPUs and three selected HLE text rows (source indices 0, 4, 10).
The new entrypoint runs `evl/eval_molt_hle3_sandbox_smoke.sh`, preserving the
base model, 8192-token response budget, 65536-token trajectory budget, and
no masking/compaction. Four tools use the Sandbox transport; Vision is absent.
Judge uses the configured API at `http://35.220.164.252:3888/v1` with model
`o3-mini`. The isolated `science_legacy_react_sandbox.py` adapter sends Bearer
authentication using SCIENCE_AGENT_JUDGE_API_KEY, falling back to API_KEY.
Judge HTTP is executed by the Sandbox HTTP worker, never directly by the
rjob host. Each nonempty candidate owns a separate lazy judge Sandbox because
the Agent tool Sandbox is closed before grading. The three attempts reuse that
instance; a finally block deletes it on success or failure. Cleanup errors are
logged without replacing the grading result. Trace info includes
`science_judge_backend=gateway_sandbox`. Empty candidates and F1 mode create
no judge instance. The rjob preflight includes `sandbox_smoke.py --judge` to
validate the configured judge model and authentication before loading vLLM.
The original non-Sandbox evaluation scripts are not edited for this integration.

Before submission, configure SERPER_API_KEY for both search tools. The job
runs web search, scholar search, page read, and summary LLM preflight before
loading vLLM. Output goes under `yangqingyu/eval/molt_hle3_sandbox/`, including
eval.log, sandbox_preflight.log, sandbox_trace.jsonl, and rollout_dump.
Serper HTTP 403 was traced to a comma-separated list sent as one API key.
The list is retained in SERPER_API_KEYS and one key is selected in
SERPER_API_KEY. Live web search, scholar search, page read, summary LLM, and
o3-mini host judge preflight all passed. The first rjob
`hle3-base-sandbox-smoke-20261009-142447-88012253` exited successfully, but
its grading was invalid: two host-direct judge requests timed out after three
attempts, and one empty candidate was skipped. The reported -6.6667 was the
mean of failure rewards (0, -10, -10), not an accuracy. Page processing also
stalled downloading tiktoken on the rjob host; that is a separate issue.
The judge transport change addresses the host-direct timeouts and must still
be checked in a new complete rjob evaluation.

Judge transport validation: the actual async Sandbox adapter graded a fixed
prime-factor candidate using the configured o3-mini API, returning
`science_judge_valid=1`, reward 1, and `science_judge_backend=gateway_sandbox`.
Policy generation was stubbed; judge HTTP, parsing, and cleanup used the
production runtime. Full rjob evaluation has not been rerun.

Lifecycle: both the Agent scope and the judge use finally cleanup. Deletion
retries three times, logging the Sandbox ID on failure and retaining the ID
if all attempts fail. A cancelled judge waits for its in-flight thread before
deleting, preventing creation after cleanup. Process termination that bypasses
finally (such as SIGKILL) relies on the gateway instance timeout, configured
by SCIENCE_AGENT_SANDBOX_TIMEOUT (default 1800 seconds).

Final local validation: 23 Sandbox tests passed, including async judge
cancellation ordering and transient-delete retry. Molt compileall and bash
syntax checks passed. The host has no pytest installed, so the complete Molt
pytest suite could not run. A combined live smoke reached Jina HTTP 200 but
its expected-page-text assertion failed before judge; its finally cleanup
still completed. This does not invalidate the separate successful live
judge-adapter test described above.

The subsequent live smoke with
`--judge --http-url https://r.jina.ai/http://example.com` passed Python
persistence, outbound HTTP 200, o3-mini judge JSON grading, and cleanup.

## Page files in the Agent workspace

The Sandbox launcher defaults to SCIENCE_AGENT_READ_PAGE_MODE=file.
read_page(url, goal) requests Jina from the Sandbox HTTP worker and stores
the complete returned Markdown in /workspace/pages/<URL-hash>/content.md.
source.json records the source/reader URLs, timestamp, length, preview, and
evaluation guard fingerprint. Pages of at most 6000 Unicode characters return
the complete Markdown directly. Longer pages return metadata, the saved file
path, and the first 1000 characters as a preview. Both sizes retain the complete
file and recovery backup; cached visits follow the same length rule. The file
branch does not invoke tokenization or a summary model.
The isolated Visit module no longer imports tiktoken or downloads encodings.

The existing evaluation URL/text guard snapshot is evaluated in the worker
before saving; blocked or failed content is not saved. Repeated same-run
URLs reuse completed files under the same guard fingerprint. Changed guard
policies force a refetch. URLs get separate filenames and concurrent Agents
have distinct Sandbox filesystems. Files disappear with Sandbox deletion.

Python and HTTP workers start in /workspace. Use the returned relative path
with the existing Python tool, for example:

```python
from pathlib import Path
lines = Path("pages/<page_id>/content.md").read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    if "router freezing" in line.lower():
        for number in range(max(0, index - 2), min(len(lines), index + 4)):
            print(f"{number + 1}: {lines[number]}")
        break
```

The gateway Python branch now uses the existing observation truncation
helper, with launcher default PYTHON_TOOL_RETURN_MAX_CHARS=50000. Full files
remain available for additional slices. /workspace is a working directory,
not a filesystem security boundary; Python can still access permitted
Sandbox system paths. No host shared-storage mount is added.

Live check (qwen-agent must be installed):

```bash
bash run_sandbox_harness.sh python3 sandbox_smoke.py --page-file --http-url https://r.jina.ai/http://example.com
```

The isolated rjob preflight includes --page-file before model startup.

Page-file validation: 27 Sandbox regression tests passed. A live test with
the real Visit and PythonInterpreter tools passed page saving, reading the
same file from /workspace, repeat-URL cache reuse, 6000-character observation
truncation, and Sandbox cleanup. Syntax/compileall checks passed. The default
host lacks pytest; the full Molt pytest suite and full policy rollout have
not been run for this change.

Path protocol: page metadata exposes only the relative `path`, for example
`pages/<page_id>/content.md`. Python already runs in /workspace; pass `path`
directly to open() or Path().read_text(), without prepending /workspace.
Absolute paths are rejected by the existing Python path policy. Cached
metadata from older workers also has absolute_path removed before return.

Infrastructure retries: gateway GET lookups retry HTTP 403/502/503/504 and
transport errors up to three attempts, with 1s/2s waits. Worker proxy calls
retry only the explicit 502 "Sandbox upstream connection failed" response.
Recovery returns the normal tool result; exhausted retries return an error.
Python execution response timeouts and generic 502 errors are not replayed
because the worker may already have executed stateful code.

Budget termination: Agent results distinguish generation_truncated from
trajectory_budget_exhausted and propagate their OR through Result.truncated
in the isolated Molt adapter, allowing session capture to mark the final
trajectory segment. Both flags and science_final_answer_valid are included
in eval trace info. Here valid means a nonempty candidate that is not an
exact template placeholder, not judge correctness. Budget finalization no
longer includes a literal placeholder answer. Empty/placeholder answers
are skipped by judge with reward 0 and explicit budget termination; real
answers produced at the limit remain eligible for grading. Existing Molt
truncated_rate still averages trajectory segments; the reason flags provide
episode diagnosis without changing that metric's definition.

Validation: 34 Sandbox tests passed, including budget placeholder rejection,
real-answer grading after budget exhaustion, and propagation through judge,
empty-answer, and F1 branches. Molt compileall passed; its full pytest suite
could not run because pytest is absent on this host.

## Experimental recovery after a Sandbox becomes inactive

Recovery is enabled by default for new tasks. Set
`SCIENCE_AGENT_SANDBOX_RECOVERY=0` before startup to disable it. This does not
update already-loaded processes or pinned
runtime snapshots. `SCIENCE_AGENT_SANDBOX_RECOVERY_DIR` selects the host backup
root; otherwise it is `$OUTPUT_ROOT/sandbox_recovery` (or `/tmp/sandbox_recovery`
without an output directory). Each task gets a private UUID directory. Use a
persistent output directory for retained diagnostics. Backups contain agent code
and page/input content, not gateway tokens or HTTP authorization headers; apply
the same retention/access policy as rollout artifacts.

The gateway journals Python dispatch and completion, including calls that return
Python exceptions (they may already have modified globals). Page reads transport
full `content.md` and `source.json` to the host before returning the usual short
page description. Uploaded visual inputs are backed up too. These file and code
events are replayed in order, preserving paths and original page contents without
fetching the pages again. Files created by Python are recreated by replay, not
independently snapshotted. Backup blobs are content-addressed and checked on read.

Only a management endpoint's explicit `SANDBOX_NOT_ACTIVE` or sandbox-not-found
response initiates replacement. Ordinary 503s, Python exceptions, DNS errors and
ambiguous timeouts do not. Replacement uses the original creation options and
clears worker/input caches. At most two replacements are attempted per task
(`SCIENCE_AGENT_SANDBOX_RECOVERY_MAX`). Python replay has a 120-second budget
(`SCIENCE_AGENT_SANDBOX_REPLAY_TIMEOUT`) after creation; this is checked between
events and bounds Python requests, but management/file-transfer requests retain
their existing individual timeouts, so it is not a hard wall-clock deadline.

Each recorded Python call gets an operation ID. With recovery enabled (the
default), an unavailable execution response first triggers a result lookup. A
completed result is reused without executing code twice. Otherwise the worker
receives a cooperative interrupt request. The gateway polls for up to five
seconds; if execution still has not completed, it terminates the identified
worker process group and starts a fresh worker in the same Sandbox. Process ID,
start time and new-worker identity are checked before reporting a successful
restart. Individual network/management request timeouts can extend this wait.

A successful interrupt retains the namespace and any partial modifications. A
restart clears Python memory, retains Sandbox files (including partial writes),
and does not replay code. Interrupted calls are excluded from recovery history.
After a confirmed restart, the active code history is cleared; file backups and
new-session code remain available. The on-disk log retains the audit trail.
Subsequent whole-Sandbox recovery reports partial restoration when interrupted
state or earlier-session generated files were not backed up. Ordinary Python
exceptions remain replayable. Failed termination/startup reports unconfirmed
state and blocks new Python work, rather than reporting success.

HTTP-only requests, including search, can proceed without consuming Python
state. Pending recovery notices gate all tools until returned to the model.
Search and Scholar return state errors directly without transport retries or
host fallback. Interrupt/restart notices describe facts without instructions.
External side effects and detached subprocesses cannot be assumed undone.

Tool responses are passed to the model without an appended remaining-token
notice, matching the historical harness. Internal token accounting and configured
context/trajectory limits still apply.

During replay, a Python audit hook rejects network/process operations and writes
outside `/workspace`; a caught audit violation still marks replay unsuccessful.
This is a guard for ordinary agent code, **not a security boundary for hostile
Python or arbitrary native extensions**. Shell commands, downloads, package
installation and some library imports may therefore prevent full recovery.
Visual Python execution is a replay barrier in this initial version, although its
input files are retained. Output/error/image digests must match the original
call. Matching output does not prove identical hidden state: random numbers,
time, external state and nondeterministic computations can still differ.

On any barrier, output difference or timeout, replay stops, records incomplete
recovery and reports the number of restored Python calls. The failing call may
have partially modified state. The model receives the recovery notice at the
ReAct tool-message boundary, ahead of ordinary output; inner tool retries cannot
continue while that notice is pending. The interrupted request is not
transparently retried. Agent-facing recovery notices contain factual status only: recovery scope,
failure category, and execution outcome. They contain no instructions or
recommendations about the agent’s next action. Raw recovery exception messages
are not interpolated into these notices. Future replacements stop at a recorded prior
incomplete-recovery barrier as well. No conversation or historical tool output is
replayed into the model context.

The journal is currently used by the same live Agent process; resuming a crashed
host process from the journal is not implemented. Timeout-triggered worker
restarts create an empty session; they do not restore the previous namespace.
Judge-only replacement continues to use the existing adapter path;
it does not require restoring a solver's Python state.

Search defaults match the historical harness: 10 web results, all Scholar results,
and no additional snippet truncation. `SCIENCE_AGENT_SEARCH_SNIPPET_CHARS=0` and
`SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS=0` disable snippet limits;
`SCIENCE_AGENT_SCHOLAR_MAX_RESULTS=0` returns all provider entries.
