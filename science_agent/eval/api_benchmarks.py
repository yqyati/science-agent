"""Dataset and answer protocols matching the Molt science evaluation recipes."""
import json
from pathlib import Path

from science_agent.eval.answer_extraction import find_boxed_spans

RESEARCH = Path('/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research')
EVAL = RESEARCH / 'jiangyichen/p1-shared-2/eval'
# Dataset sizes are validated before applying a smoke-run limit.
BENCHMARKS = {
    'hle': (RESEARCH / 'yangqingyu/science-agent-next/data/hle_subsets/HLE_all_mini.jsonl', 360),
    'amobench': (EVAL / 'math/amobench.jsonl', 50),
    'fso': (EVAL / 'frontierscience/FrontierScience_Olympiad.jsonl', 100),
    'imo_answerbench': (EVAL / 'math/imo_answerbench.jsonl', 400),
    'cmt_benchmark': (EVAL / 'cmt_benchmark/cmt_benchmark.jsonl', 50),
}


def load_benchmark_rows(args):
    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        key = str(raw['id'])
        question, answer = raw['question'], raw['answer']
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Missing question: {key}')
        answers = [answer] if isinstance(answer, str) else answer
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f'Missing reference answers: {key}')
        reference = '\n'.join(answers) if args.benchmark == 'fso' else json.dumps(answers, ensure_ascii=False)
        prompt = question
        if args.benchmark == 'imo_answerbench':
            messages = raw['prompt']
            if (not isinstance(messages, list) or len(messages) != 1
                    or messages[0].get('role') != 'user'
                    or not isinstance(messages[0].get('content'), str)
                    or not messages[0]['content'].strip()):
                raise ValueError(f'Invalid IMO prompt: {key}')
            prompt = messages[0]['content']
        elif args.benchmark == 'cmt_benchmark':
            if raw.get('image_question'):
                raise ValueError(f'Image input requires an explicit adapter: {key}')
            prompt = (
                'Use LaTeX for formulas. Show your reasoning, then write FINAL ANSWER: '
                'followed by the final answer(s) in \\boxed{}. '
                'For multiple answers, preserve their order and use separate boxes. '
                'Do not use boxes in the reasoning.\n\n'
                f"Problem:\n{question}\n\nParameters (if any):\n{raw.get('parameters') or 'N/A'}"
                f"\n\nFunctions (if any):\n{raw.get('functions') or 'N/A'}\n"
            )
        elif args.benchmark == 'fso':
            prompt += ('\n\nUse LaTeX for formulas. Put the final answer(s) inside <answer></answer>, '
                       'with each requested answer in a separate \\boxed{...}, in the order requested. '
                       'Keep labels and units clear.')
        rows.append({**raw, 'id': key, 'question_id': key, 'question': prompt,
                     'answer': reference,
                     'source': 'frontierscience_olympiad' if args.benchmark == 'fso' else args.benchmark,
                     'dataset_base_dir': str(args.dataset_base_dir)})
    if len(rows) != args.expected_rows or len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError(f'Expected {args.expected_rows} unique {args.benchmark} questions; found {len(rows)} rows')
    return rows[:args.limit] if args.limit else rows


def extract_prediction(benchmark, text):
    prediction = str(text or '').rsplit('</think>', 1)[-1].strip()
    if benchmark in {'amobench', 'imo_answerbench', 'fso'}:
        for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
            prediction = prediction.replace(token, '')
    if benchmark in {'amobench', 'imo_answerbench'}:
        spans = find_boxed_spans(prediction)
        if spans:
            prediction = '\n'.join(span.canonical for span in spans)
    return '' if prediction.strip().casefold() in {'your answer', 'your final answer'} else prediction


def benchmark_judge_messages(benchmark, row, prediction):
    question, reference = row['question'], row['answer']
    if benchmark in {'amobench', 'imo_answerbench'}:
        system = 'You are a strict grader for mathematical competition answers.'
        user = (
            'You are grading an attempted answer to a mathematical competition problem.\n'
            'The reference answer may be a list of acceptable answers. Mark correct only when the attempted '
            'final answer is mathematically equivalent to a reference answer. Ignore harmless formatting '
            'such as \\boxed{...}, surrounding prose, or equivalent LaTeX formatting. '
            'Leading zeros are harmless but a different value is incorrect.\n'
            f'Problem:\n{question}\n\nReference answer:\n{reference}\n\n'
            f'Attempted extracted answer:\n{prediction}\n\n'
            'Return only valid JSON: {"correct": true/false, "reason": "brief explanation"}.'
        )
    elif benchmark == 'fso':
        system = 'You are a strict but fair grader for science olympiad problems.'
        user = (
            'Evaluate the attempted final answer against the reference answer. '
            'Mark correct only if it fully matches or is equivalent to the reference. '
            'Accept equivalent algebraic expressions, compound names or formulas, and unit conversions. '
            'Accept numerical rounding to the precision of the reference '
            '(for example, 6.69 is equivalent to 6.7). '
            'For multiple requested answers, check every part and its correspondence to the question.\n\n'
            f'Problem:\n{question}\n\nReference answer:\n{reference}\n\n'
            f'Attempted answer:\n{prediction}\n\n'
            'Return only valid JSON: {"correct": true/false, "reason": "brief explanation"}.'
        )
    else:
        # CMT uses the same strict boolean grader as HLE, preserving all answer parts.
        system = 'You are a strict grader for problems across all academic disciplines. Given a problem and a candidate solution, decide whether the candidate answer is correct. Be strict: the final answer must match the ground truth (allow equivalent forms).'
        user = ('You will output ONLY valid JSON with this schema:\n{"correct": true/false, "reason": "one-sentence reason"}\n\n'
                f'Problem:\n{question}\n\nCandidate answer (model output):\n{prediction}\n\nGround truth answer:\n{reference}\n')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
