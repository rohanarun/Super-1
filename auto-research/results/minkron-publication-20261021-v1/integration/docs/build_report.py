"""Build compact, self-contained research status report from measured results."""
import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'research-scratch/nanogpt-memory'
OUTPUT = ROOT.parent


def main():
    summary = json.loads((RESULTS/'smoke/summary.json').read_text())
    curves = json.loads((RESULTS/'smoke/curves.json').read_text())
    verification = json.loads((RESULTS/'verification_cpu.json').read_text())
    harness = json.loads((RESULTS/'data_and_runner_checks.json').read_text())
    accounting = json.loads((RESULTS/'model_accounting.json').read_text())
    assert summary['status'] == verification['status'] == harness['status'] == 'PASS'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), layout='constrained')
    steps = [r['step'] for r in curves]
    axes[0].semilogy(steps, [r['nll'] for r in curves], marker='o', color='#087f8c', lw=2)
    axes[0].set(xlabel='Optimizer updates', ylabel='Training-batch NLL (log scale)', title='Fixed-batch memorization diagnostic')
    axes[0].grid(alpha=.2)
    axes[1].plot(steps, [r['accuracy']*100 for r in curves], marker='o', color='#7254b3', lw=2)
    axes[1].set(xlabel='Optimizer updates', ylabel='Next-token accuracy (%)', ylim=(-2, 104), title='128 repeated scored positions')
    axes[1].grid(alpha=.2)
    chart = RESULTS/'smoke_learning.png'
    fig.savefig(chart, dpi=155, facecolor='white')
    plt.close(fig)
    figure = base64.b64encode(chart.read_bytes()).decode()
    rows = ''.join(f'<tr><td>{r["arm"]}</td><td>{r["parameters"]:,}</td><td>{r["parameter_bytes"]:,}</td><td>{r["state_backing_bytes"]:,}</td></tr>' for r in accounting)
    core32 = [r for r in verification['tests'] if r['name']=='core_vs_authors' and r['dtype']=='torch.float32']
    max_output = max(r['output_max_abs'] for r in core32)
    max_state = max(r['state_max_abs'] for r in core32)
    max_grad = max(r['all_inputs_and_initial_state_gradient_max_abs'] for r in core32)
    result = dict(status='CPU_INTEGRATION_PASS_GPU_FINEWEB_PENDING', date='2026-10-21', source_commit=commit,
        original_contract_sha256=summary['original_comparison_contract_sha256'],
        source_model_sha256=verification['model_sha256'],
        data_manifest_status='FineWeb unresolved; fixture token IDs preserved separately',
        completed=['source recovery and checksum verification', 'common decoder implementations',
                   'FP32/FP64 reference and gradient checks', 'real-text GPT-2 overfit diagnostic',
                   'source optimizer equivalence', 'checkpoint and interrupted-run restoration',
                   'document-disjoint preparation and exact token-accounting checks'],
        not_run=['FineWeb LR grid', 'paired 20M-token confirmation seeds', 'custom GPU kernel',
                 'BF16 GPU validation', 'GPU prefill/decode benchmarks', 'quality/cost comparison'],
        smoke=summary, model_accounting=accounting,
        verification=dict(records=len(verification['tests']), all_pass=True,
            fp32_core_output_max_abs=max_output, fp32_core_state_max_abs=max_state, fp32_core_gradient_max_abs=max_grad),
        harness_checks=harness['tests'],
        shared_contract_clarifications_proposed=dict(RMSNorm_epsilon=1e-5, RoPE_base=10000,
            bounded_chunk_length=32, initial_projection_dtype='FP32', microbatch_sequences=1,
            data_split='whole-document exact-content-hash assignment with deduplication; not official speedrun split'),
        warnings=['No generalization result from memorization', 'No GPU throughput inference from CPU time',
                  '4 KB recurrent state excludes weights, optimizer, gradients, activations and temporary workspace'])
    status_file = OUTPUT/'minKron_NanoGPT_Status_2026-10-21.json'
    status_file.write_text(json.dumps(result, indent=2)+'\n')
    (RESULTS/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>minKron NanoGPT — integration results, 21 October 2026</title>
<style>body{{margin:0;background:#f3f5f7;color:#1b2938;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1040px;margin:40px auto;padding:40px;background:white;border-radius:14px}}h1{{font-size:34px;line-height:1.15;margin:12px 0}}h2{{font-size:23px;margin-top:36px}}p{{max-width:88ch}}.eyebrow{{font-size:13px;font-weight:700;letter-spacing:.08em;color:#087f8c}}.status{{padding:18px 22px;background:#eef7f7;border-left:5px solid #087f8c}}.pending{{padding:18px 22px;background:#fff7e7;border-left:5px solid #cd8b18}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:25px 0}}.metric{{background:#f2f5f8;padding:18px;border-radius:8px}}.value{{display:block;font-size:26px;font-weight:700}}.label{{font-size:13px;color:#526273}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:right;border-bottom:1px solid #dce2e8;padding:10px}}td:first-child,th:first-child{{text-align:left}}th{{background:#f2f5f8}}img{{width:100%;height:auto}}code{{font-family:monospace;font-size:.88em;overflow-wrap:anywhere}}.small{{font-size:13px;color:#526273}}a{{color:#076e91}}@media(max-width:650px){{main{{margin:0;padding:22px;border-radius:0}}.metrics{{grid-template-columns:1fr}}h1{{font-size:29px}}table{{font-size:11px}}td,th{{padding:6px}}}}</style></head><body><main>
<div class="eyebrow">GEOMETRIC SELF ATTENTION · EXPERIMENT CONTINUATION · 21 OCTOBER 2026</div>
<h1>minKron is integrated into the common language-model decoder</h1>
<div class="status"><b>CPU integration and learning diagnostic: passed.</b> The exact 4-layer, 256-wide decoder learns through the full GPT-2 vocabulary head while retaining 4 KB of recurrent state. Reference behavior, gradients, document resets, and checkpoint restoration are verified.</div>
<p class="pending"><b>The requested FineWeb/GPU research run is pending.</b> This environment has PyTorch 2.9.1+cpu and no CUDA device. No FineWeb LR search, paired-seed quality experiment, custom GPU kernel, BF16 GPU validation, or GPU throughput comparison was run.</p>
<div class="metrics"><div class="metric"><span class="value">15.15M</span><span class="label">minKron parameters · 60.59 MB in FP32</span></div><div class="metric"><span class="value">4,096 B</span><span class="label">Owned recurrent state per sequence · four layers</span></div><div class="metric"><span class="value">100 updates</span><span class="label">12,800 presentations of 128 fixed positions</span></div></div>
<h2>The diagnostic that actually ran</h2>
<p>A two-sequence batch, each 64 tokens long, was drawn from the supplied handoff's real prose and tokenized using GPT-2 (50,257 vocabulary items). The complete proposed minKron model was trained in FP32 for 100 AdamW updates at LR 0.003, seed 22901. This LR is a diagnostic choice; it is not a selection for the FineWeb pilot.</p>
<p>Training-batch NLL fell from <b>{summary['training_batch_initial_nll']:.5f}</b> to <b>{summary['training_batch_final_nll']:.8f}</b>; final next-token accuracy was <b>100%</b>. These are memorization measurements, with no held-out set. They establish an operational learning path, not generalization or parity with a Transformer.</p>
<img alt="Training-batch NLL decreases from 10.87 to 0.00036 and accuracy reaches 100 percent over 100 updates" src="data:image/png;base64,{figure}">
<p class="small">The recorded {summary['train_seconds']:.2f} seconds sums optimizer-step wall times on two CPU threads. Evaluation and the checkpoint test are excluded. This diagnostic timing is not comparable with the future research harness's full-loop timing and is not a GPU estimate.</p>
<h2>What the reference checks establish</h2>
<ul><li>{len(verification['tests'])} verification records passed, including FP32/FP64 comparisons with the authors' recurrence at lengths 1, 17, 33, 65 and 513, asymmetric per-example reset boundaries, and nonzero initial states.</li>
<li>Largest measured FP32 core error: output {max_output:.3g}, final state {max_state:.3g}, all input/initial-state gradients {max_grad:.3g}. Finite-difference gradient checking also passed.</li>
<li>Permanent packing preserves source initialization exactly; source-versus-packed parameter error after three AdamW steps was 4.04e-15 in FP64.</li>
<li>All three decoder arms passed future-token causality, document reset, sequence splitting, and shared-backbone initialization checks. A full 512-token, 50,257-vocabulary minKron forward/backward completed with finite gradients.</li>
<li>At diagnostic step 50, the full checkpoint was serialized and restored. The next optimizer step matched exactly: parameter and loss differences both zero. The actual training harness also passed interrupted/resumed run tests for every arm with an exact partial final token budget.</li></ul>
<p class="small">One initial test demanded bitwise-zero prefix gradients after a reset. Cumulative-decay subtraction produced 4.26e-9 FP32 residue versus a 0.149 suffix gradient; FP64 residue was 2.98e-18. The test now bounds numerical residue. The initial failure record and the measured diagnostic are retained; no model equation was changed to hide it.</p>
<h2>Separate the memory quantities</h2>
<table><thead><tr><th>Arm</th><th>Parameters</th><th>FP32 weight bytes</th><th>State/cache bytes at 512</th></tr></thead><tbody>{rows}</tbody></table>
<p class="small">Batch 1. State column counts actual backing storage. Transformer cache includes K/V and document/position metadata. The models are not parameter-matched. minKron and minGRU state is constant with context; all have additional training activations, gradients, optimizer state and workspace. The minKron diagnostic AdamW state alone occupied {summary['optimizer_bytes']:,} bytes.</p>
<p>The successful small-head geometry is preserved: four 8×8 matrices per layer, with a 32-channel readout projected to width 256. The separate write and erase gates, vector forgetting, normalization, SiLU, output gating and tiny-vector safeguard remain. Pairwise solve tensors are bounded to chunks of 32, but this is an unfused native PyTorch backend.</p>
<h2>Data correction before the main run</h2>
<p>The pinned speedrun preprocessor can cut a document between its first validation shard and the following training shard. That does not satisfy the handoff's document-disjoint rule. The new loader assigns whole documents by a deterministic content hash, deduplicates exact text, retains explicit IDs and document offsets, and hashes tokens and manifests. Every arm must use the same resulting split. This adapted split is not the official speedrun validation set.</p>
<p>Only small loader fixtures were exercised here. No actual FineWeb shard, dataset revision or main-data checksum is claimed resolved. The original comparison JSON is preserved unchanged; the pending qualification manifest spells out common implementation choices before any paired training.</p>
<h2>Next execution</h2>
<ol><li>Provide a CUDA runner and pin/export real FineWeb documents, including source shard provenance.</li><li>Run GPU reference/gradient checks, qualify precision and freeze the shared data/hardware contract.</li><li>Run all three arms at the same 3 learning rates and 2M-token budget. Freeze development-only LR selections and the common development target.</li><li>Run the three paired 20M-token confirmation seeds, retaining failures, then measure GPU prefill/decode and peak working memory.</li></ol>
<p><code>docs/GPU_LAUNCH.md</code> contains the commands and remaining kernel/serving gates. The common runnable training harness includes exact token accounting, causal document resets and complete resume state. No paid job was provisioned and no GitHub push was made.</p>
<h2>Provenance and reproducibility</h2>
<p>Recovered source archive SHA256: <code>16f71c993b2fa310610604038e3105256f6805868756c7b8d02b0e9324c8d5bc</code>.<br>Original shared contract SHA256: <code>{summary['original_comparison_contract_sha256']}</code>.<br>Local source snapshot commit: <code>{commit}</code>.</p>
<p>Sources: <a href="https://github.com/NX-AI/KronLSTM">authors' KronLSTM implementation</a>; <a href="https://github.com/JRowe47/Super-1/tree/568cfb7b4017619b93c1f4107cc688ca33d30c5c">pinned Super-1 hierarchy</a>; <a href="https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/data/fineweb.py">pinned FineWeb preprocessing source</a>. Vendored references and MIT notices are included.</p>
<p class="small">The compact archive preserves source, source patch, exact fixture tokens, diagnostic protocol/curves, all verification evidence and launch instructions. It omits runtimes, full datasets and reproducible temporary diagnostic checkpoints. GSA curve-generated weights, the Super-1 feedback hierarchy and Zipfy are not part of this decoder run.</p>
</main></body></html>'''
    path = OUTPUT/'minKron_NanoGPT_Integration_Report_2026-10-21.html'
    path.write_text(document)
    print(json.dumps(dict(report=str(path), status=str(status_file), source_commit=commit)))


if __name__ == '__main__':
    main()
