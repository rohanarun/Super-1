"""Freeze a shared FP32 GPU/data contract only after local verification."""
import argparse
import json
from pathlib import Path
import torch
from .data import canonical_hash, file_hash, load_split
from .train import source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not torch.cuda.is_available():
        raise RuntimeError('Cannot qualify a GPU run without a CUDA device')
    if Path(args.output).exists():
        raise FileExistsError('Do not silently replace a frozen comparison contract')
    verification = json.loads(Path(args.verification).read_text())
    if verification['status'] != 'PASS' or not verification['device'].startswith('cuda'):
        raise ValueError('Require successful checks on this GPU, not CPU evidence')
    if verification['torch'] != torch.__version__:
        raise ValueError('Verification runtime changed')
    model_sha = file_hash(root/'experiments/nanogpt_memory/model.py')
    if verification['model_sha256'] != model_sha:
        raise ValueError('Model changed after verification')
    if verification['device_name'] != torch.cuda.get_device_name(0):
        raise ValueError('Verification used a different device')
    data = Path(args.data)
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest['source']['corpus'] != 'HuggingFaceFW/fineweb':
        raise ValueError('Research corpus must be FineWeb')
    if len(manifest['source']['revision']) != 40:
        raise ValueError('Pin the source dataset to a full commit SHA')
    id_sets, hash_sets = [], []
    for split, count in [('train', 20000000), ('development', 1000000), ('confirmation', 1000000)]:
        tokens, _ = load_split(data, split)
        meta = manifest['splits'][split]
        if len(tokens) != count+1 or meta['scored_tokens'] != count:
            raise ValueError(f'Incorrect token budget for {split}')
        docfile = data/meta['documents_file']
        if file_hash(docfile) != meta['documents_sha256']:
            raise ValueError('Document manifest checksum mismatch')
        documents = json.loads(docfile.read_text())
        id_sets.append({d['id'] for d in documents})
        hash_sets.append({d['text_sha256'] for d in documents})
    for i in range(3):
        for j in range(i):
            if id_sets[i] & id_sets[j] or hash_sets[i] & hash_sets[j]:
                raise ValueError('Document/content leakage between splits')
    original = json.loads((root/'comparison_contract.json').read_text())
    result = dict(status='frozen_after_gpu_and_data_qualification',
        original_contract_sha256=canonical_hash(original), data_manifest_sha256=canonical_hash(manifest),
        source_sha256=source_hash(root), model_sha256=model_sha,
        device_name=torch.cuda.get_device_name(0), device_count_used=1,
        device_total_memory=torch.cuda.get_device_properties(0).total_memory,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        verification_sha256=file_hash(args.verification),
        projection_precision='FP32 qualification baseline; BF16 requires separate tested revision',
        shared_clarifications=dict(rms_epsilon=1e-5, rope_base=10000, chunk_length=32,
            common_weight_init_std=.02, residual_mlp_scale='1/sqrt(2*layers)',
            microbatch_sequences=1, target_token_count_includes_text_to_eot=True,
            exact_final_partial_optimizer_step=True, peak_lr_grid=[.0003, .001, .003],
            target_rule='best Transformer pilot final development NLL, fixed before confirmation'),
        selection=None)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
