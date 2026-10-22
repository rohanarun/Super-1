"""Explicit document-disjoint GPT-2 data; no synthetic fallback for research."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

EOT = 50256


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def next_window(tokens, cursor, length, device):
    """Score exactly length next-token targets; independent window state.

    An EOT input begins a new document. The preceding text's last token may
    predict EOT, but EOT cannot attend/read back into the preceding document.
    """
    if cursor+length+1 > len(tokens):
        raise ValueError('Insufficient tokens: refuse wraparound in a fixed-token run')
    values = torch.tensor(np.array(tokens[cursor:cursor+length+1], dtype=np.int64), device=device)
    x, y = values[:-1][None], values[1:][None]
    reset = x == EOT
    reset[:, 0] = True
    return x, y, reset, cursor+length


def load_split(root, split, verify=True):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    meta = manifest['splits'][split]
    path = root/meta['file']
    if verify and file_hash(path) != meta['sha256']:
        raise ValueError(f'{split} data checksum mismatch')
    return np.load(path, mmap_mode='r'), manifest


def prepare(input_jsonl, output, source_revision, limits=None):
    """Tokenize FineWeb JSONL in input order, assigning WHOLE docs by hash.

    Mirrors the pinned pipeline's [EOT]+encode_ordinary(document), while fixing
    its cross-shard split leakage. The manifest makes this adaptation explicit.
    Text duplicates are skipped globally; doc IDs and text hashes are audited.
    """
    import tiktoken
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh data directory')
    output.mkdir(parents=True)
    limits = limits or {'train': 20000000, 'development': 1000000, 'confirmation': 1000000}
    enc = tiktoken.get_encoding('gpt2')
    data, documents = {s: [] for s in limits}, {s: [] for s in limits}
    counts = {s: 0 for s in limits}
    seen_ids, seen_hashes = set(), set()
    duplicates = 0
    for line_number, line in enumerate(Path(input_jsonl).open(), 1):
        doc = json.loads(line)
        if not all(k in doc for k in ('id', 'text')):
            raise ValueError('Each source row requires FineWeb id and text')
        docid, text = str(doc['id']), doc['text']
        digest = hashlib.sha256(text.encode()).hexdigest()
        if docid in seen_ids or digest in seen_hashes:
            duplicates += 1
            continue
        seen_ids.add(docid)
        seen_hashes.add(digest)
        bucket = int(digest[:16], 16) % 22
        split = 'development' if bucket == 0 else 'confirmation' if bucket == 1 else 'train'
        if counts[split] >= limits[split]+1:
            continue
        tokens = [EOT] + enc.encode_ordinary(text)
        take = min(len(tokens), limits[split]+1-counts[split])
        documents[split].append(dict(id=docid, text_sha256=digest, source_line=line_number,
            start_token=counts[split], retained_tokens=take, full_document_tokens=len(tokens),
            dump=doc.get('dump'), file_path=doc.get('file_path')))
        data[split].extend(tokens[:take])
        counts[split] += take
        if all(counts[s] >= limits[s]+1 for s in limits):
            break
    if any(counts[s] < limits[s]+1 for s in limits):
        raise ValueError(f'Insufficient source documents: {counts}; partial output is not qualified')
    manifest = dict(status='prepared; freeze with hardware before main training',
        source=dict(corpus='HuggingFaceFW/fineweb', config='sample-10BT', revision=source_revision,
                    input_file=Path(input_jsonl).name, input_sha256=file_hash(input_jsonl)),
        tokenizer=dict(name='gpt2', library='tiktoken', version=tiktoken.__version__, n_vocab=enc.n_vocab,
            mergeable_ranks_sha256=canonical_hash([(k.hex(), v) for k, v in sorted(enc._mergeable_ranks.items(), key=lambda p:p[1])])),
        split_rule='SHA256 UTF-8 exact text; first 16 hex digits modulo 22: 0 dev, 1 confirmation, rest train; globally deduplicated',
        state_policy='reset each independent window and before each EOT input; score text-to-EOT target',
        duplicates_skipped=duplicates, splits={})
    for split in limits:
        path = output/f'{split}.npy'
        np.save(path, np.asarray(data[split], dtype=np.uint16))
        docpath = output/f'{split}_documents.json'
        docpath.write_text(json.dumps(documents[split], indent=2)+'\n')
        manifest['splits'][split] = dict(file=path.name, sha256=file_hash(path),
            tokens=counts[split], scored_tokens=limits[split], documents=len(documents[split]),
            documents_file=docpath.name, documents_sha256=file_hash(docpath))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-revision', required=True, help='Exact Hugging Face dataset commit')
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_jsonl, args.output, args.source_revision), indent=2))
