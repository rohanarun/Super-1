"""Small integration gate for token accounting, data splits and full resume.

Fixture vocabulary is deliberately tiny for fast harness testing. These
synthetic checks are never presented as a FineWeb/model quality experiment.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nanogpt_memory.data import prepare, load_split, next_window, canonical_hash
from experiments.nanogpt_memory.model import Config
from experiments.nanogpt_memory.train import train_fixed, evaluate


def main():
    torch.set_num_threads(2)
    checks = []
    with tempfile.TemporaryDirectory(prefix='nanogpt-data-check-') as temp:
        tmp = Path(temp)
        source = tmp/'fixture.jsonl'
        docs = [dict(id=f'test-{i}', text=f'This is an integration test document numbered {i}. It is not FineWeb.') for i in range(500)]
        # Duplicate must never cross split boundaries or get double counted.
        docs.insert(1, dict(id='duplicate', text=docs[0]['text']))
        source.write_text(''.join(json.dumps(d)+'\n' for d in docs))
        manifest = prepare(source, tmp/'data', 'test-fixture-not-research',
                           limits={'train': 600, 'development': 120, 'confirmation': 120})
        id_sets, hash_sets = [], []
        for split, count in [('train', 600), ('development', 120), ('confirmation', 120)]:
            tokens, _ = load_split(tmp/'data', split)
            assert len(tokens) == count+1
            documents = json.loads((tmp/'data'/manifest['splits'][split]['documents_file']).read_text())
            id_sets.append({d['id'] for d in documents})
            hash_sets.append({d['text_sha256'] for d in documents})
            cursor, scored = 0, 0
            while cursor < count:
                length = min(31, count-cursor)
                x, y, reset, nxt = next_window(tokens, cursor, length, 'cpu')
                assert torch.equal(x[:, 1:], y[:, :-1])
                assert reset[0, 0] and torch.equal(reset[0, 1:], x[0, 1:] == 50256)
                scored += y.numel()
                cursor = nxt
            assert scored == count
        for i in range(3):
            for j in range(i):
                assert not (id_sets[i] & id_sets[j] or hash_sets[i] & hash_sets[j])
        assert manifest['duplicates_skipped'] >= 1
        checks.append(dict(name='document_disjointness_deduplication_exact_token_masks', status='PASS'))

        # Exercise a final partial optimizer step, a partial window, a serialized
        # optimizer, and a real interrupted/resumed training path with all arms.
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8]*16, dtype=np.uint16)
        for arm in ('minkron_8x8', 'mingru_linear', 'transformer'):
            cfg = Config(width=64, layers=1, kron_heads=1, vocab_size=16, arm=arm)
            kwargs = dict(budget=53, dev_count=29, sequence_length=11, global_tokens=23,
                peak_lr=.001, seed=777, device='cpu', frozen_hash='test', data_hash='test',
                source_hash='test', target_nll=1., evaluation_every=1)
            full, summary = train_fixed(cfg, tokens, tokens, outdir=tmp/f'{arm}_full', **kwargs)
            assert summary['trained_tokens'] == 53 and summary['optimizer_steps'] == 3
            assert summary['target_censored'] is True
            # Interrupt just after the first complete checkpoint using the
            # checkpoint hook. Production code is unchanged after this fixture.
            import experiments.nanogpt_memory.train as runner
            original_save = runner.save_checkpoint
            def interrupt_after_save(*a, **kw):
                original_save(*a, **kw)
                raise InterruptedError('intentional resume gate')
            runner.save_checkpoint = interrupt_after_save
            try:
                train_fixed(cfg, tokens, tokens, outdir=tmp/f'{arm}_resume', **kwargs)
            except InterruptedError:
                pass
            finally:
                runner.save_checkpoint = original_save
            resumed, resumed_summary = train_fixed(cfg, tokens, tokens, outdir=tmp/f'{arm}_resume', resume=True, **kwargs)
            error = max(float((a-b).abs().max().detach()) for a, b in zip(full.parameters(), resumed.parameters()))
            assert error == 0.
            assert resumed_summary['trained_tokens'] == 53 and resumed_summary['optimizer_steps'] == 3
            attempts = json.loads((tmp/f'{arm}_resume'/'attempts.json').read_text())
            assert len(attempts) == 2 and attempts[0]['failure_reason'] is not None
            assert attempts[1]['failure_reason'] is None and resumed_summary['failed_attempt_count'] == 1
            assert resumed_summary['train_seconds'] == sum(a['train_seconds'] for a in attempts)
            checks.append(dict(name='fixed_token_runner_resume', arm=arm, status='PASS',
                               scored_tokens=53, optimizer_steps=3, parameter_max_abs=error,
                               failed_attempt_and_spent_compute_preserved=True))
    result = dict(status='PASS', tests=checks, caveat='Harness fixtures only; no research data or quality evidence')
    (ROOT/'research-scratch/nanogpt-memory/data_and_runner_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
