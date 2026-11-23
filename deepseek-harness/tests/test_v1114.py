import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "molt_adapter"))
from verifier import EpisodeVerifier

SNAPSHOT = ROOT / "data/verifiable_1k_v451p9"


@pytest.mark.parametrize('bucket,candidate,objective', [
    ('target400', {'circles': [[0.2, 0.2, 0.1]] * 9}, None),
    ('target400', {'circles': [[0.2 + i * 0.4, 0.2 + j * 0.4, 0.125] for i in range(3) for j in range(3)]}, 1.125),
    ('source300', {'squares': []}, None),
    ('source300', {'squares': [[[(i + j + c) % 6 for j in range(6)] for i in range(6)] for c in range(3)]}, 90),
    ('general300', {'labels': [0] * 13}, 0),
])
def test_new_verifier_dispatch(tmp_path, bucket, candidate, objective):
    rows = [json.loads(line)['dsh_item'] for line in (SNAPSHOT / 'train.jsonl').read_text().splitlines()]
    item = next(row for row in rows if row['task']['bucket'] == bucket)
    verifier = EpisodeVerifier(item, SNAPSHOT / 'verifier', tmp_path)
    result = verifier.submit(json.dumps(candidate))
    assert result['objective'] == objective
    assert result['valid'] == (objective is not None)
    if bucket == 'general300':
        assert verifier.outcome()['reward'] == 0.5
        labels = [i % 2 for i in range(13)]
        improved = verifier.submit(json.dumps({'labels': labels}))
        assert improved['valid'] and improved['objective'] > objective
        assert verifier.outcome()['reward'] > 0.5


def test_public_snapshot_has_all_questions_without_private_qa():
    rows = [json.loads(line) for line in (SNAPSHOT / 'train.jsonl').read_text().splitlines()]
    assert len(rows) == len({row['dsh_item']['task_id'] for row in rows}) == 1000
    assert not list(SNAPSHOT.rglob('trusted_specs.jsonl'))
    for row in rows:
        assert row['dsh_item']['generation_split'] == 'train'
        assert set(row['dsh_item']['task']) == {
            'task_id', 'bucket', 'family', 'problem', 'parameters', 'answer_contract', 'objective', 'verifier_id', 'verifier'}
        assert len(row['prompt']) == 1 and row['prompt'][0]['role'] == 'user'


def test_molt_arrow_loading_and_label_roundtrip(tmp_path):
    from datasets import load_dataset
    from molt.datasets import PromptDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from types import SimpleNamespace

    dataset = load_dataset('json', data_files=str(SNAPSHOT / 'train.molt.jsonl'), cache_dir=str(tmp_path))['train']
    original = [json.loads(line) for line in (SNAPSHOT / 'train.jsonl').read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/model/Qwen3.5-35B-A3B',
        local_files_only=True)
    strategy = SimpleNamespace(args=SimpleNamespace(data=SimpleNamespace(
        input_key='prompt', label_key='dsh_item', apply_chat_template=True)))
    prompts = PromptDataset(dataset, tokenizer, strategy, prerender=False)
    loader = DataLoader(prompts, batch_size=1, collate_fn=prompts.collate_fn)
    for raw, (_, messages, labels, _, _) in zip(original, loader, strict=True):
        assert messages[0] == raw['prompt']
        assert isinstance(labels[0], str)
        assert json.loads(labels[0]) == raw['dsh_item']
    assert len(dataset) == 1000


def test_agent_decodes_molt_string_label():
    import asyncio
    from types import SimpleNamespace
    from agent import DSHAgent

    row = json.loads((SNAPSHOT / 'train.molt.jsonl').read_text().splitlines()[0])
    ctx = SimpleNamespace(label=row['dsh_item'], images=['unexpected-image'], tools=[])
    with pytest.raises(ValueError, match='text tasks'):
        asyncio.run(DSHAgent().run(ctx))
