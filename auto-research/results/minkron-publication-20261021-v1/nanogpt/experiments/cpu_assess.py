"""Assess one final CPU checkpoint and emit the ten prespecified samples once."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import tiktoken
from torch.nn import functional as F
from experiments.nanogpt_memory.model import Config, MemoryLM, tensor_bytes
from experiments.nanogpt_memory.data import file_hash, canonical_hash
from experiments.cpu_scale.common import configure_cpu
from experiments.cpu_scale.run import evaluate


@torch.inference_mode()
def reset_diagnostic(model, data, count, length, batch):
    total = 0.
    for cursor in range(0, count, batch*length):
        values = np.stack([data[cursor+i*length:cursor+(i+1)*length+1] for i in range(batch)]).astype(np.int64)
        ids = torch.from_numpy(values)
        logits, _ = model(ids[:, :-1], reset=torch.ones_like(ids[:, :-1], dtype=torch.bool))
        total += float(F.cross_entropy(logits.flatten(0,1), ids[:, 1:].flatten(), reduction='sum'))
    return dict(evaluation_tokens=count, nll=total/count, perplexity=math.exp(total/count))


@torch.inference_mode()
def generate(model, encoder, prompt, seed, options):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = [encoder.eot_token]+encoder.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    reset = x == encoder.eot_token
    started = time.perf_counter()
    logits, state = model(x, reset=reset)
    prefill_seconds = time.perf_counter()-started
    generated = []
    decode_start = time.perf_counter()
    reason = 'max_new_tokens'
    for _ in range(options['max_new_tokens']):
        next_logits = logits[:, -1].float()/options['temperature']
        values, indices = torch.topk(next_logits, options['top_k'], dim=-1)
        selected = torch.multinomial(values.softmax(-1), 1, generator=generator)
        token = int(indices.gather(-1, selected)[0, 0])
        generated.append(token)
        if options['stop_on_eot'] and token == encoder.eot_token:
            reason = 'eot'
            break
        logits, state = model(torch.tensor([[token]]), state=state,
                              reset=torch.tensor([[False]]))
    decode_seconds = time.perf_counter()-decode_start
    return dict(prompt=prompt, seed=seed, input_ids=ids, generated_ids=generated,
                continuation=encoder.decode(generated), stop_reason=reason,
                generated_tokens=len(generated), prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds, total_seconds=time.perf_counter()-started,
                tokens_per_second=len(generated)/decode_seconds,
                state_logical_bytes=tensor_bytes(state)[0], state_storage_bytes=tensor_bytes(state)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='research-scratch/cpu-124m/main')
    parser.add_argument('--confirmation-tokens', type=int, default=8192)
    args = parser.parse_args()
    run = Path(args.run)
    if (run/'assessment.json').exists() or (run/'generations.jsonl').exists():
        raise RuntimeError('Assessment already exists; refuse to regenerate/cherry-pick samples')
    protocol = json.loads((run/'protocol.json').read_text())
    configure_cpu(protocol['threads'])
    checkpoint = run/'checkpoint.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['metrics']['status'] == 'COMPLETED'
    assert payload['protocol_sha256'] == canonical_hash(protocol)
    for path, expected in protocol['code_sha256'].items():
        assert file_hash(path) == expected, path
    config = Config(**payload['config'])
    model = MemoryLM(config)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    data_cursor = payload['data_cursor']
    optimizer_tensors = [v for s in payload['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v)]
    checkpoint_info = dict(sha256=file_hash(checkpoint), bytes=checkpoint.stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer_tensor_bytes=sum(v.numel()*v.element_size() for v in optimizer_tensors),
        optimizer_finite=all(bool(torch.isfinite(v).all()) for v in optimizer_tensors),
        data_cursor=int(data_cursor['cursor']), rng_fields=list(payload['rng']),
        scheduler=payload['scheduler'], state_policy='reset at every independent training window')
    assert checkpoint_info['optimizer_finite']
    del optimizer_tensors, payload
    gc.collect()
    root = Path('data/fineweb_cpu')
    manifest = json.loads((root/'manifest.json').read_text())
    assert canonical_hash(manifest) == protocol['data_manifest_sha256']
    assert file_hash(root/'confirmation.npy') == manifest['splits']['confirmation']['sha256']
    confirmation = np.load(root/'confirmation.npy', mmap_mode='r')
    final = evaluate(model, confirmation, args.confirmation_tokens, protocol['sequence_length'], protocol['batch'], 'fp32')
    print(json.dumps(dict(event='confirmation', **final)), flush=True)
    # A small post-training diagnostic, never used to select any checkpoint.
    normal = evaluate(model, confirmation, 2048, protocol['sequence_length'], protocol['batch'], 'fp32')
    no_history = reset_diagnostic(model, confirmation, 2048, protocol['sequence_length'], protocol['batch'])
    train = np.load(root/'train.npy', mmap_mode='r')
    counts = np.zeros(config.vocab_size, dtype=np.int64)
    length = protocol['sequence_length']
    for index in data_cursor['order'][:data_cursor['cursor']]:
        targets = train[int(index)*length+1:int(index)*length+length+1].astype(np.int64)
        counts += np.bincount(targets, minlength=config.vocab_size)
    log_probs = np.log((counts+1.)/(counts.sum()+config.vocab_size))
    unigram_nll = float(-log_probs[confirmation[1:args.confirmation_tokens+1].astype(np.int64)].mean())
    encoder = tiktoken.get_encoding('gpt2')
    # Check trained-model chunk prefill against token recurrence on one fixed prompt.
    check_ids = torch.tensor([[encoder.eot_token]+encoder.encode_ordinary(protocol['generation']['prompts'][0])])
    with torch.inference_mode():
        chunk_logits, chunk_state = model(check_ids, reset=check_ids == encoder.eot_token)
        state, parts = None, []
        for i in range(check_ids.shape[1]):
            token = check_ids[:, i:i+1]
            logits, state = model(token, state=state, reset=token == encoder.eot_token)
            parts.append(logits)
        recurrent_logits = torch.cat(parts, dim=1)
        max_error = float((chunk_logits-recurrent_logits).abs().max())
        state_error = max(float((a-b).abs().max()) for a,b in zip(chunk_state,state))
        assert torch.allclose(chunk_logits, recurrent_logits, atol=2e-4, rtol=2e-4), max_error
    generation_options = protocol['generation']
    samples = []
    text_lines = ['Ten fixed generations from the final 25-minute CPU checkpoint.',
                  'Prompts/seeds fixed before training; temperature 0.8, top-k 50, at most 64 new GPT-2 tokens; no retries.', '']
    for number, (prompt, seed) in enumerate(zip(generation_options['prompts'], generation_options['seeds']), 1):
        sample = dict(number=number, **generate(model, encoder, prompt, seed, generation_options))
        samples.append(sample)
        with (run/'generations.jsonl').open('a') as handle:
            handle.write(json.dumps(sample, ensure_ascii=False)+'\n')
        text_lines += [f'{number}. PROMPT: {prompt}', 'CONTINUATION:', sample['continuation'], '']
        print(json.dumps(dict(event='generation', **sample), ensure_ascii=False), flush=True)
    (run/'generations.txt').write_text('\n'.join(text_lines), encoding='utf-8')
    assessment = dict(checkpoint=checkpoint_info, confirmation=final,
        recurrence_diagnostic=dict(tokens=2048, normal_nll=normal['nll'],
            reset_every_token_nll=no_history['nll'], context_advantage_nats=no_history['nll']-normal['nll'],
            interpretation='Positive difference means recurrent context helps these tokens; not a long-memory benchmark'),
        unigram_baseline=dict(training_targets=int(counts.sum()), smoothing='add one', nll=unigram_nll,
            perplexity=math.exp(unigram_nll), source='Only targets actually seen by the final training run'),
        trained_chunk_vs_recurrent=dict(logits_max_abs=max_error, state_max_abs=state_error, status='PASS'),
        generation=dict(options=generation_options, count=len(samples),
            generated_tokens=sum(s['generated_tokens'] for s in samples),
            decode_seconds=sum(s['decode_seconds'] for s in samples),
            prefill_seconds=sum(s['prefill_seconds'] for s in samples)),
        assessment_script_sha256=file_hash(__file__))
    (run/'assessment.json').write_text(json.dumps(assessment, indent=2)+'\n')
    print(json.dumps(dict(event='assessment_complete', **assessment)), flush=True)


if __name__ == '__main__':
    main()
