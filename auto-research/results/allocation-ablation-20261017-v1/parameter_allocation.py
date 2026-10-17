#!/usr/bin/env python3
"""
Parameter-allocation analysis for the ~100M hierarchy-versus-nanoGPT question.

Reads nothing, trains nothing, and touches no device. It is pure arithmetic on
two published formulas so that the numbers in PARAMETER_ALLOCATION.json can be
regenerated and checked by anyone:

  * hierarchy native body:  P(V, d) = V*d + 32*d^2 + 23*d
    (the recovered native HierarchicalDistributionStudent formula recorded in
    auto-research/SCALING_1B.md; it is NOT the formula for function-weight arms)

  * nanoGPT block:  V*d + block_size*d + n_layer*(12*d^2 + 11*d) + 2*d
    with a tied lm_head, i.e. block = ln_1 + c_attn + c_proj + mlp(c_fc, c_proj)
    = 2d + (3d^2+3d) + (d^2+d) + (4d^2+4d) + (4d^2+d)

Both vocabularies are reported because the binding constraint at ~100M is the
vocabulary, not the recurrence. Usage:

    python3 parameter_allocation.py > PARAMETER_ALLOCATION.json
"""

import json
import math

QWEN_V = 151936   # Qwen3-0.6B tokenizer, per recovered INPUT_MANIFEST.json
TOKENS = 10_000_000  # historical short-budget pilot exposure


def hier_total(v, d):
    return v * d + 32 * d * d + 23 * d


def hier_body(d):
    return 32 * d * d + 23 * d


def nano_block(d):
    return 12 * d * d + 11 * d


def nano_total(v, d, layers, block_size):
    return v * d + block_size * d + layers * nano_block(d) + 2 * d


def row(v, d, total, embed, body):
    return {
        "vocab": v,
        "width": d,
        "total": total,
        "embedding": embed,
        "body": body,
        "body_fraction_pct": round(100.0 * body / total, 3),
    }


def main():
    out = {
        "purpose": (
            "Isolate parameter allocation from architecture before spending compute "
            "on a ~100M hierarchy-versus-nanoGPT comparison."
        ),
        "status": "ANALYSIS_ONLY_NO_TRAINING_NO_QUALITY_CLAIM",
        "inputs": {
            "hierarchy_formula": "P(V,d) = V*d + 32*d^2 + 23*d",
            "nanogpt_block_formula": "12*d^2 + 11*d (tied lm_head)",
            "vocabularies_reported": [50257, 32768, QWEN_V],
            "token_budget_assumed": TOKENS,
            "token_budget_source": "historical 120M/10M-token report",
        },
        "hierarchy_by_width_at_qwen_vocab": [],
        "nanogpt_by_shape": [],
        "hierarchy_by_vocab_at_width_688": [],
        "embedding_data_per_row": [],
        "allocation_lever_split_embed_width": [],
        "findings": [],
    }

    for d in (512, 560, 608, 640, 688, 768):
        t = hier_total(QWEN_V, d)
        out["hierarchy_by_width_at_qwen_vocab"].append(
            row(QWEN_V, d, t, QWEN_V * d, hier_body(d))
        )

    for v, d, layers in ((QWEN_V, 640, 6), (QWEN_V, 768, 8),
                         (QWEN_V, 768, 10), (QWEN_V, 768, 12)):
        block_size = 2048
        t = nano_total(v, d, layers, block_size)
        embed = v * d + block_size * d
        body = layers * nano_block(d) + 2 * d
        r = row(v, d, t, embed, body)
        r["layers"] = layers
        r["block_size"] = block_size
        out["nanogpt_by_shape"].append(r)

    for v in (50257, 32768, QWEN_V):
        d = 688
        t = hier_total(v, d)
        r = row(v, d, t, v * d, hier_body(d))
        r["tokens_per_embedding_row"] = round(TOKENS / v, 1)
        out["hierarchy_by_vocab_at_width_688"].append(r)

    for v in (50257, 32768, QWEN_V):
        out["embedding_data_per_row"].append(
            {"vocab": v, "tokens_per_embedding_row": round(TOKENS / v, 1)}
        )

    # Same total-parameter envelope; less spent on the tied embedding, more on the
    # body. embed_d below body_d compresses 151936 tokens into fewer dimensions,
    # which is a real information cost and is exactly why this stays an ablation.
    for embed_d, body_d in ((512, 688), (448, 768), (384, 832),
                            (320, 896), (256, 960)):
        e = QWEN_V * embed_d
        b = hier_body(body_d)
        r = row(QWEN_V, embed_d, e + b, e, b)
        r["body_width"] = body_d
        r["embed_width"] = embed_d
        out["allocation_lever_split_embed_width"].append(r)

    h = next(x for x in out["hierarchy_by_width_at_qwen_vocab"] if x["width"] == 688)
    n = next(x for x in out["nanogpt_by_shape"] if x["layers"] == 6 and x["width"] == 640)
    out["findings"] = [
        {
            "id": "F1-embedding-dominates",
            "statement": (
                "At V=151936 the recovered hierarchy at d=688 spends "
                f"{h['body_fraction_pct']}% of its parameters on the tied embedding "
                f"({h['embedding']} of {h['total']}), leaving {h['body']} for the body."
            ),
            "consequence": (
                "The ~100M envelope is dominated by a lookup table, so a "
                "parameter-matched comparison is not a capacity-matched comparison."
            ),
        },
        {
            "id": "F2-baseline-gets-more-body",
            "statement": (
                "At a comparable total, nanoGPT d=640/6L holds "
                f"{n['body']} body parameters against the hierarchy's {h['body']} "
                f"({n['body'] / h['body']:.2f}x more)."
            ),
            "consequence": (
                "Part of the observed hierarchy deficit is explainable by allocation "
                "rather than by recurrence, and must be excluded before any "
                "architecture claim."
            ),
        },
        {
            "id": "F3-embedding-untrained-at-budget",
            "statement": (
                "At the 10M-token budget the V=151936 embedding gives each row "
                f"{TOKENS / QWEN_V:.0f} tokens."
            ),
            "consequence": (
                "That table is essentially untrained, so a V=151936 short-budget "
                "comparison mostly measures embedding efficiency."
            ),
        },
        {
            "id": "F4-vocab-shrink-frees-budget",
            "statement": (
                "Holding the exact recovered body width d=688, V=32768 costs "
                f"{hier_total(32768, 688)} parameters versus {hier_total(QWEN_V, 688)} "
                f"at V=151936, freeing {hier_total(QWEN_V, 688) - hier_total(32768, 688)}."
            ),
            "consequence": (
                "Vocabulary is the cheapest declared lever available and leaves the "
                "gates, clocks and recurrence untouched."
            ),
        },
        {
            "id": "F5-not-a-quality-result",
            "statement": "No model was trained and no loss, quality or frontier number was produced.",
            "consequence": (
                "These are analytic parameter counts. They do not license any quality, "
                "SOTA, novelty or transfer claim."
            ),
        },
    ]
    out["decision_rule"] = (
        "If a matched-context comparison at the reallocated budget still loses to "
        "nanoGPT, the allocation explanation is falsified and the recurrence carries "
        "the remaining deficit. Only that outcome may be reported as an architecture "
        "result."
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()