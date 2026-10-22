"""Bounded, auditable document-disjoint slice of one pinned FineWeb shard."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from experiments.nanogpt_memory.data import file_hash, canonical_hash


def main():
    root = Path('data/fineweb_cpu')
    meta = json.loads((root/'source_metadata.json').read_text())
    source = root/'source.parquet'
    expected = next(x for x in meta['files'] if x['path']=='sample/10BT/000_00000.parquet')
    digest = file_hash(source)
    assert digest == expected['lfs']['oid'], 'FineWeb shard checksum mismatch'
    if (root/'manifest.json').exists():
        raise FileExistsError('Keep the original frozen data preparation')
    limits = {'train': 5000000, 'development': 65536, 'confirmation': 65536}
    arrays = {k: np.empty(v+1, dtype=np.uint16) for k, v in limits.items()}
    counts = {k: 0 for k in limits}
    documents = {k: [] for k in limits}
    seen_ids, seen_texts = set(), set()
    duplicate_count = 0
    enc = tiktoken.get_encoding('gpt2')
    row_cursor = 0
    start = time.perf_counter()
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=1000, columns=['text','id','dump','file_path']):
        for row in batch.to_pylist():
            index = row_cursor
            row_cursor += 1
            text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
            if row['id'] in seen_ids or text_hash in seen_texts:
                duplicate_count += 1
                continue
            seen_ids.add(row['id'])
            seen_texts.add(text_hash)
            bucket = int(text_hash[:16], 16) % 100
            split = 'development' if bucket == 0 else 'confirmation' if bucket == 1 else 'train'
            if counts[split] >= limits[split]+1:
                continue
            ids = [enc.eot_token]+enc.encode_ordinary(row['text'])
            take = min(len(ids), limits[split]+1-counts[split])
            lo = counts[split]
            arrays[split][lo:lo+take] = ids[:take]
            counts[split] += take
            documents[split].append(dict(id=row['id'], text_sha256=text_hash, shard_row=index,
                dump=row['dump'], file_path=row['file_path'], start_token=lo,
                retained_tokens=take, full_document_tokens=len(ids)))
            if all(counts[s] == limits[s]+1 for s in limits):
                break
        if all(counts[s] == limits[s]+1 for s in limits):
            break
    assert all(counts[s] == limits[s]+1 for s in limits), counts
    manifest = dict(purpose='separate CPU ~124M 25-minute exploration; not the original GPU comparison contract',
        source=dict(repo='HuggingFaceFW/fineweb', config='sample-10BT', revision=meta['revision'],
                    path=expected['path'], bytes=source.stat().st_size, sha256=digest,
                    rows_examined=row_cursor, total_shard_rows=parquet.metadata.num_rows),
        tokenizer=dict(name='GPT-2', n_vocab=enc.n_vocab, version=tiktoken.__version__,
            ranks_sha256=canonical_hash([(k.hex(), v) for k, v in sorted(enc._mergeable_ranks.items(), key=lambda p:p[1])])),
        split_rule='SHA256 exact text first 16 hex digits modulo 100; 0 development, 1 confirmation, 2-99 train; global ID and exact-text deduplication',
        eot_policy='prepend EOT, reset before EOT input; score preceding document final text-to-EOT target',
        duplicates_skipped=duplicate_count, preparation_seconds=time.perf_counter()-start, splits={})
    for split, values in arrays.items():
        path = root/f'{split}.npy'
        np.save(path, values)
        docfile = root/f'{split}_documents.json'
        docfile.write_text(json.dumps(documents[split], indent=2)+'\n')
        manifest['splits'][split] = dict(file=path.name, sha256=file_hash(path), tokens=len(values),
            scored_tokens=limits[split], documents=len(documents[split]),
            document_manifest=docfile.name, document_manifest_sha256=file_hash(docfile))
    id_sets = [{d['id'] for d in documents[s]} for s in limits]
    text_sets = [{d['text_sha256'] for d in documents[s]} for s in limits]
    for i in range(3):
        for j in range(i):
            assert not (id_sets[i] & id_sets[j] or text_sets[i] & text_sets[j])
    manifest['document_disjointness_check'] = 'PASS'
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
