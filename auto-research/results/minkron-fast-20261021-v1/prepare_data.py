"""Losslessly retokenize the recovered FineWeb arrays with a 4096-entry GPT-2 prefix BPE."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parent
EOT = 4095

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tokenizer():
    spec = json.loads((ROOT/'data/tokenizer.json').read_text())
    return tiktoken.Encoding(name='gpt2-prefix4096', pat_str=spec['pat_str'],
        mergeable_ranks={bytes.fromhex(k): v for k, v in spec['ranks']},
        special_tokens={'<|endoftext|>': EOT})

def main(source):
    source = Path(source)
    dest = ROOT/'data'
    dest.mkdir(exist_ok=True)
    original = json.loads((source/'manifest.json').read_text())
    full = tiktoken.get_encoding('gpt2')
    ranks = [(k.hex(), v) for k, v in sorted(full._mergeable_ranks.items(), key=lambda p:p[1])]
    canonical = hashlib.sha256(json.dumps(ranks, separators=(',', ':')).encode()).hexdigest()
    assert canonical == original['tokenizer']['ranks_sha256']
    (dest/'tokenizer.json').write_text(json.dumps(dict(pat_str=full._pat_str,
        ranks=ranks[:EOT], eot=EOT, n_vocab=4096, full_gpt2_ranks_sha256=canonical), indent=2)+'\n')
    enc = tokenizer()
    result = dict(source=original, tokenizer_sha256=sha(dest/'tokenizer.json'),
        method='First 4095 GPT-2 merge ranks plus EOT; lossless byte decoding/reencoding, no fitted vocabulary.',
        cross_run_caveat='Development and confirmation documents were used in the earlier 124M pilot. This is internal ablation evidence, not a globally unseen transfer test.',
        splits={})
    for split, meta in original['splits'].items():
        assert sha(source/meta['file']) == meta['sha256']
        old = np.load(source/meta['file'])
        docs = json.loads((source/meta['document_manifest']).read_text())
        ids, new_docs = [], []
        for doc in docs:
            start = doc['start_token']
            count = doc['retained_tokens']
            tokens = old[start:start+count].tolist()
            assert tokens[0] == 50256
            raw = full.decode_bytes(tokens[1:])
            small = enc._encode_bytes(raw)
            assert enc.decode_bytes(small) == raw
            new_docs.append(dict(source_id=doc['id'], text_sha256=doc['text_sha256'],
                start_token=len(ids), retained_tokens=1+len(small), retained_bytes=len(raw)))
            ids.extend([EOT]+small)
        np.save(dest/f'{split}.npy', np.asarray(ids, dtype=np.uint16))
        (dest/f'{split}_documents.json').write_text(json.dumps(new_docs, indent=2)+'\n')
        result['splits'][split] = dict(file=f'{split}.npy', sha256=sha(dest/f'{split}.npy'),
            tokens=len(ids), source_gpt2_tokens=len(old), documents=len(docs),
            document_manifest=f'{split}_documents.json', document_manifest_sha256=sha(dest/f'{split}_documents.json'))
    (dest/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v['tokens'] for k,v in result['splits'].items()}), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source', required=True)
    main(p.parse_args().source)
