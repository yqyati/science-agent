#!/usr/bin/env python3
"""Evaluate HLE through a solver API and Sandbox tools, without Molt or Ray."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/science-agent-next/data/hle_subsets/HLE_all_mini.jsonl')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def artifact(args, kind, row):
    key = hashlib.sha256(row['question_id'].encode()).hexdigest()[:24]
    return args.run_dir / kind / f'{key}.json'


def load_rows(args):
    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        question = raw.get('question') or raw.get('problem') or raw.get('prompt')
        answer = raw.get('answer') or raw.get('ground_truth') or raw.get('target')
        if not isinstance(question, str) or not question.strip() or answer is None:
            raise ValueError('Each HLE row requires a nonempty question and a reference answer')
        key = str(raw.get('question_id') or raw.get('id') or raw.get('task_id') or f'hle360-{len(rows):04d}')
        rows.append({**raw, 'question_id': key, 'id': key, 'question': question,
                     'answer': str(answer), 'source': raw.get('source') or 'hle_all_mini',
                     'dataset_base_dir': str(args.dataset_base_dir)})
    if len(rows) != args.expected_rows or len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError(f'Expected {args.expected_rows} unique HLE questions; found {len(rows)} rows')
    return rows[:args.limit] if args.limit else rows


def configure(args):
    # Set before importing the agent: its loop budgets are import-time constants.
    settings = {
        'SCIENCE_TRAJECTORY_MAX_TOKENS': str(args.max_context_tokens),
        'QWEN_API_MAX_OUTPUT_TOKENS': str(args.max_output_tokens),
        'AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS': str(args.soft_context_tokens),
        'AGENT_CONTEXT_COMPACTION': '0', 'AGENT_CONTEXT_COMPACT_MAX_COMPACTIONS': '0',
        'AGENT_OBSERVATION_MASKING': '0', 'AGENT_EVAL_FORCE_FINALIZE': '1',
        'AGENT_ANSWER_TYPE': 'boxed', 'OPENAI_COMPAT_USE_SYSTEM_PROMPT': '1',
        'AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING': '0',
        'AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS': '0', 'MAX_TOOL_CALLS_PER_RUN': '0',
        'AGENT_TOOL_BACKEND': 'local', 'CONTEXT_MANAGEMENT_STRATEGY': 'none',
        'SCIENCE_AGENT_TOOLS': 'search,visit,python,google_scholar',
        'PYTHON_SESSION_BACKEND': 'gateway_sandbox',
        'SCIENCE_AGENT_TOOL_HTTP_BACKEND': 'gateway_sandbox',
        'SCIENCE_AGENT_SANDBOX_HTTP_STRICT': 'true',
        'SCIENCE_AGENT_SANDBOX_RESOURCE': args.sandbox_resource,
        'SCIENCE_AGENT_SANDBOX_TIMEOUT': '14400', 'SCIENCE_AGENT_SANDBOX_RECOVERY': '1',
        'SCIENCE_AGENT_READ_PAGE_MODE': 'file', 'PYTHON_TOOL_RETURN_MAX_CHARS': '50000',
        'SCIENCE_AGENT_READ_PAGE_MAX_CHARS': '6000', 'SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS': '4',
        'SCIENCE_AGENT_SEARCH_MAX_RESULTS': '10', 'SCIENCE_AGENT_SEARCH_SNIPPET_CHARS': '0',
        'SCIENCE_AGENT_SCHOLAR_MAX_RESULTS': '0', 'SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS': '0',
        'TASK_TIMEOUT_SEC': str(args.task_timeout), 'MAX_LLM_CALL_PER_RUN': str(args.max_llm_calls),
        'LLM_RETRY_INCOMPLETE_OUTPUT': '0',
        'LLM_AUTO_ADJUST_MAX_OUTPUT_TOKENS_ON_CONTEXT_ERROR': '1',
        'LLM_CONTEXT_ERROR_MIN_OUTPUT_TOKENS': '1',
        'OPENAI_REQUEST_TIMEOUT': str(args.request_timeout),
        'AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS': '1',
        'SERVED_MODEL_NAME_FALLBACKS': '', 'OPENAI_COMPAT_RESPONSES_MODE': '0',
        'SCIENCE_AGENT_DEEPSEEK_V4_REASONING': '0',
        'SCIENCE_AGENT_TRACE_FILE': str(args.run_dir / 'sandbox_trace.jsonl'),
    }
    os.environ.update(settings)
    if args.solver_direct:
        hostname = urlsplit(args.base_url).hostname
        for key in ('NO_PROXY', 'no_proxy'):
            entries = [x for x in os.environ.get(key, os.environ.get('NO_PROXY', '')).split(',') if x]
            os.environ[key] = ','.join(dict.fromkeys([*entries, hostname]))
    # Preserve the independently configured Judge chain and credentials.
    defaults = {'SCIENCE_AGENT_JUDGE_MODEL': 'bailian/deepseek-v4.1-flash',
                'SCIENCE_AGENT_JUDGE_BASE_URL': 'http://35.220.164.252:3888/v1',
                'SCIENCE_AGENT_JUDGE_THINKING': 'enabled',
                'SCIENCE_AGENT_JUDGE_MAX_TOKENS': '6000', 'SCIENCE_AGENT_JUDGE_TIMEOUT': '180'}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    return settings


def make_agent_factory(args):
    from react_agent_openai_api_tools import MultiTurnReactAgent

    class APIAgent(MultiTurnReactAgent):
        def _get_api_base(self):
            return args.base_url

        def _get_api_key(self):
            return os.environ[args.api_key_env]

        def _deepseek_v4_profile_model_name(self, model_name):
            # API parameter conventions are explicit, independent of model aliases.
            return ''

        def count_tokens(self, messages, tool_schemas=None):
            # API models need not have Qwen's local vision processor. Preserve the
            # actual image blocks sent to the API; estimate only their budget cost.
            images = sum(isinstance(part, dict) and part.get('type') == 'image_url'
                         for m in messages if isinstance(m.get('content'), list)
                         for part in m['content'])
            if not images:
                return super().count_tokens(messages, tool_schemas)
            text_messages = copy.deepcopy(messages)
            for message in text_messages:
                if isinstance(message.get('content'), list):
                    message['content'] = [p for p in message['content']
                                          if not isinstance(p, dict) or p.get('type') != 'image_url']
            self.image_context_estimated = True
            return super().count_tokens(text_messages, tool_schemas) + images * args.image_tokens_per_image

        def call_server(self, messages, planning_port, **kwargs):
            self.last_messages = copy.deepcopy(messages)
            remaining = args.max_context_tokens - self.count_tokens(messages, kwargs.get('tool_schemas')) - 1
            if remaining <= 0:
                raise RuntimeError('API context budget exhausted before dispatch')
            kwargs['max_output_tokens'] = min(kwargs.get('max_output_tokens') or args.max_output_tokens,
                                               args.max_output_tokens, remaining)
            # Forced-finalization overrides (e.g. disabling thinking) take precedence.
            extra = copy.deepcopy(args.request_extra_body)
            for key, value in (kwargs.get('request_extra_body') or {}).items():
                if isinstance(value, dict) and isinstance(extra.get(key), dict):
                    extra[key].update(value)
                else:
                    extra[key] = value
            kwargs['request_extra_body'] = extra
            response, usage = super().call_server(messages, planning_port, **kwargs)
            self.context_tokens_estimated = self._context_tokenizer is None or self.image_context_estimated
            return response, usage

    def create():
        agent = APIAgent(function_list=['search', 'visit', 'python', 'google_scholar'], llm={
            'model': args.tokenizer_path or args.model, 'served_model_name': args.model,
            'generate_cfg': {'temperature': args.temperature, 'top_p': args.top_p,
                             'max_input_tokens': 320000, 'max_retries': 3, 'presence_penalty': 0},
        })
        agent.last_messages = []
        agent.context_tokens_estimated = True
        agent.image_context_estimated = False
        return agent
    return create


def hle_candidate(text):
    from science_agent.eval.answer_extraction import find_boxed_spans, select_hle_candidate
    text = str(text or '').strip()
    spans = find_boxed_spans(text)
    prediction = select_hle_candidate(text, spans[-1].canonical).value if spans else text
    return '' if prediction.casefold() in {'your answer', 'your final answer'} else prediction


def judge_messages(row, prediction):
    # Match the HLE protocol used by the Molt adapter.
    return [
        {'role': 'system', 'content': 'You are a strict grader for problems across all academic disciplines. Given a problem and a candidate solution, decide whether the candidate answer is correct. Be strict: the final answer must match the ground truth (allow equivalent forms).'},
        {'role': 'user', 'content': 'You will output ONLY valid JSON with this schema:\n{"correct": true/false, "reason": "one-sentence reason"}\n\n'
         f"Problem:\n{row['question']}\n\nCandidate answer (model output):\n{prediction}\n\nGround truth answer:\n{row['answer']}\n"},
    ]


def safe_error(exc, secrets):
    message = f'{type(exc).__name__}: {exc}'
    for secret in secrets:
        if secret:
            message = message.replace(secret, '<redacted>')
    return message[:2000]


async def run_question(row, args, create_agent, judge_answer, secrets):
    rollout_path = artifact(args, 'rollouts', row)
    grade_path = artifact(args, 'grades', row)
    rollout = json.loads(rollout_path.read_text()) if rollout_path.exists() else None
    if args.stage != 'judge' and (rollout is None or args.retry_failed and
                                 (rollout.get('error') or not rollout.get('prediction'))):
        agent = create_agent()
        started = time.monotonic()
        try:
            # The reference stays out of the solving task; only grading receives it.
            item = {key: value for key, value in row.items()
                    if key not in {'answer', 'ground_truth', 'target', 'messages'}}
            rollout = agent._run({'item': item,
                                 'tool_artifact_dir': str(args.run_dir / 'tools' / rollout_path.stem)}, args.model)
        except Exception as exc:
            rollout = {'prediction': '', 'messages': agent.last_messages,
                       'termination': 'solver exception', 'error': safe_error(exc, secrets)}
        rollout.update(question_id=row['question_id'], elapsed_seconds=time.monotonic() - started,
                       context_tokens_estimated=agent.context_tokens_estimated)
        rollout['prediction'] = hle_candidate(rollout.get('prediction'))
        save_json(rollout_path, rollout)
        # A deliberately retried rollout invalidates its old grade.
        grade_path.unlink(missing_ok=True)
    if rollout is None:
        return 'missing saved rollout'
    if args.stage == 'rollout':
        return 'rollout saved'
    if grade_path.exists():
        previous = json.loads(grade_path.read_text())
        if not (args.retry_failed and previous['status'] == 'judge_error'):
            return 'saved ' + previous['status']
    prediction = rollout.get('prediction', '')
    if not prediction:
        grade = {'status': 'empty_answer', 'correct': False, 'score': 0, 'judge': None}
    else:
        try:
            result = await judge_answer(judge_messages(row, prediction), require_score=False)
            valid = result['judge_result'] is not None
            correct = bool((result['judge_result'] or {}).get('correct'))
            grade = {'status': 'graded' if valid else 'judge_error', 'correct': correct,
                     'score': int(correct) if valid else -1, 'judge': result}
        except Exception as exc:
            grade = {'status': 'judge_error', 'correct': False, 'score': -1,
                     'error': safe_error(exc, secrets)}
    grade.update(question_id=row['question_id'], prediction=prediction,
                 termination=rollout.get('termination'), truncated=bool(rollout.get('truncated')),
                 generation_truncated=bool(rollout.get('generation_truncated')),
                 trajectory_budget_exhausted=bool(rollout.get('trajectory_budget_exhausted')),
                 completion_tokens=rollout.get('completion_tokens'),
                 context_tokens_estimated=rollout.get('context_tokens_estimated', True),
                 elapsed_seconds=rollout.get('elapsed_seconds'),
                 tool_calls=sum(len(m.get('tool_calls') or []) for m in rollout.get('messages', [])
                                if isinstance(m, dict)))
    save_json(grade_path, grade)
    return grade['status']


def summarize(rows, args):
    grades = [json.loads(p.read_text()) for row in rows
              if (p := artifact(args, 'grades', row)).exists()]
    correct = sum(g['correct'] for g in grades)
    summary = {'selected': len(rows), 'completed': len(grades), 'pending': len(rows) - len(grades),
               'correct': correct, 'pass1_completed': correct / len(grades) if grades else None,
               'pass1_overall': correct / len(rows),
               'empty_answers': sum(g['status'] == 'empty_answer' for g in grades),
               'judge_errors': sum(g['status'] == 'judge_error' for g in grades),
               'truncated': sum(g['truncated'] for g in grades),
               'completion_tokens': sum(g.get('completion_tokens') or 0 for g in grades),
               'usage_missing': sum(g.get('completion_tokens') is None for g in grades),
               'context_estimated': sum(g.get('context_tokens_estimated', True) for g in grades),
               'tool_calls': sum(g.get('tool_calls', 0) for g in grades)}
    save_json(args.run_dir / 'summary.json', summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--expected-rows', type=int, default=360)
    parser.add_argument('--limit', type=int, default=0, help='First N questions for a smoke run; 0 means all')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-url', default=os.getenv('HLE_API_BASE_URL', ''))
    parser.add_argument('--model', default=os.getenv('HLE_API_MODEL', ''))
    parser.add_argument('--api-key-env', default='HLE_API_KEY')
    parser.add_argument('--solver-direct', action='store_true', default=os.getenv('HLE_API_DIRECT') == '1',
                        help='Bypass environment proxies for the solver hostname only')
    parser.add_argument('--tokenizer-path', default='', help='Optional local solver tokenizer; otherwise context counts are estimates')
    parser.add_argument('--request-extra-body', type=json.loads, default=json.loads(os.getenv('HLE_API_EXTRA_BODY', '{}')))
    parser.add_argument('--image-tokens-per-image', type=int, default=4096,
                        help='Estimated context cost per API image; actual accounting depends on the endpoint')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--max-context-tokens', type=int, default=122880)
    parser.add_argument('--soft-context-tokens', type=int, default=110592)
    parser.add_argument('--max-output-tokens', type=int, default=122880)
    parser.add_argument('--max-llm-calls', type=int, default=100)
    parser.add_argument('--task-timeout', type=int, default=9000)
    parser.add_argument('--request-timeout', type=int, default=600)
    parser.add_argument('--sandbox-resource', default='10.12.111.143/ailab-ai4sgi')
    parser.add_argument('--dataset-base-dir', type=Path, default=Path('/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/HLE_tool_v1'))
    parser.add_argument('--stage', choices=['all', 'rollout', 'judge', 'summary'], default='all')
    parser.add_argument('--retry-failed', action='store_true', help='Retry empty/failed rollouts, or invalid grades; judge stage never regenerates')
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not 0 < args.soft_context_tokens < args.max_context_tokens:
        parser.error('Require 0 < soft context threshold < hard context limit')
    if min(args.workers, args.max_output_tokens, args.max_llm_calls, args.task_timeout, args.request_timeout, args.expected_rows, args.image_tokens_per_image) <= 0 or args.limit < 0:
        parser.error('Budgets and worker count must be positive; limit cannot be negative')
    if not isinstance(args.request_extra_body, dict) or set(args.request_extra_body) & {'messages', 'model', 'tools', 'max_tokens', 'temperature', 'top_p', 'stream'}:
        parser.error('request-extra-body must be an object of provider options, without core request fields')
    if args.base_url:
        url = urlsplit(args.base_url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            parser.error('base-url must be an HTTP(S) endpoint without embedded credentials, query or fragment')
    if args.stage != 'summary' and (not args.base_url or not args.model):
        parser.error('Set --base-url and --model (or HLE_API_BASE_URL and HLE_API_MODEL)')
    if args.stage in {'all', 'rollout'} and not os.getenv(args.api_key_env):
        parser.error(f'Set solver API key in {args.api_key_env}; use EMPTY for an unauthenticated internal service')
    return args


def main(argv=None):
    args = parse_args(argv)
    saved_dataset = args.run_dir / 'dataset.json'
    rows = (json.loads(saved_dataset.read_text()) if args.stage == 'summary' and saved_dataset.exists()
            else load_rows(args))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.run_dir / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'summary':
            print(json.dumps(summarize(rows, args), ensure_ascii=False))
            return
        settings = configure(args)
        sys.path.insert(0, str(ROOT / 'inference'))
        from judge_client import judge_answer, load_judge_chain
        from sandbox_gateway import sandbox_run_scope
        from science_agent.tracing import trace_context
        chain = load_judge_chain(default_max_tokens=6000)
        config = {key: value for key, value in vars(args).items()
                  if key not in {'stage', 'retry_failed', 'run_dir', 'workers'} }
        config.update(runtime=settings, dataset_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      judge_chain=[{k: v for k, v in service.items() if k != 'api_key'} for service in chain],
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [ROOT / 'eval_hle_api.py', *sorted((ROOT / 'inference').glob('*.py')),
                                               *sorted((ROOT / 'science_agent/eval').glob('*.py'))]})
        config = json.loads(json.dumps(config, default=str))
        manifest = args.run_dir / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError('Run configuration/source changed; use a new run directory')
        save_json(manifest, config)
        save_json(args.run_dir / 'dataset.json', rows)
        create_agent = make_agent_factory(args) if args.stage != 'judge' else None
        secrets = [os.getenv(args.api_key_env, ''), *(s['api_key'] for s in chain)]

        @sandbox_run_scope
        async def scoped(row):
            with trace_context(question_id=row['question_id'], run_id=args.run_dir.name):
                return await run_question(row, args, create_agent, judge_answer, secrets)

        def worker(row):
            return asyncio.run(scoped(row))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, row): row['question_id'] for row in rows}
            for future in as_completed(futures):
                status = future.result()
                summary = summarize(rows, args)
                print(f"[{summary['completed']}/{len(rows)}] {futures[future]}: {status}; "
                      f"correct={summary['correct']} pass1_completed={summary['pass1_completed']}", flush=True)
        print(json.dumps(summarize(rows, args), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
