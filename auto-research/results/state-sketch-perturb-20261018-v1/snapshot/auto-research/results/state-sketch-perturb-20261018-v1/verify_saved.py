"""Post-run artifact audit; reload one guided checkpoint per architecture."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
import torch
from experiment import Corpus, evaluate, state_hash, source_manifest
from models import build

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    terminal = json.loads((OUT / "TERMINAL.json").read_text())
    assert terminal["status"] == "COMPLETE"
    assert source_manifest() == protocol["source_hashes"]
    assert hashlib.sha256((OUT / "PROTOCOL.json").read_bytes()).hexdigest() == terminal["protocol_sha256"]
    confirmation = Corpus("confirmation")
    checks = []
    for kind, width in protocol["widths"].items():
        folder = OUT / f"{kind}-73"
        parent = json.loads((folder / "PARENT.json").read_text())
        expected = json.loads((folder / "sketch.json").read_text())
        model = build(kind, parent["config"], 73)
        model.load_state_dict(torch.load(folder / "sketch.pt", map_location="cpu", weights_only=True))
        actual = evaluate(model, confirmation)
        assert actual == expected["followup"]
        assert state_hash(model) == expected["final_state_sha256"]
        checks.append({"architecture": kind, "seed": 73, "checkpoint_hash_exact": True,
                       "fresh_process_eval_exact": True, "ce": actual["ce"]})
    results = []
    for kind in protocol["widths"]:
        for seed in protocol["seeds"]:
            rows = {m: json.loads((OUT / f"{kind}-{seed}" / f"{m}.json").read_text()) for m in protocol["methods"]}
            for method, row in rows.items():
                assert len(row["training_losses"]) == 128
                assert row["nominal_target_exposures"] == 262144
                assert row["duplicate_exact"]
                if method in ("iid", "random_pair", "sketch", "scrambled"):
                    assert len(row["events"]) == 3
                    for e in row["events"]:
                        assert len(e["candidates"]) == 4
                        assert e["accepted_loss"] <= e["base_loss"]
                if method in ("sketch", "scrambled"):
                    assert [e["anchor_refreshed"] for e in row["events"]] == [True, False, True]
                if method == "replay_backprop":
                    for e in row["events"]:
                        assert 0 <= e["overshoot_seconds"] <= e["last_step_seconds"] + .001
                results.append(row)
    assert len(results) == 54
    audit = {"status": "PASS", "branches_audited": len(results), "protocol_and_source_seals": True,
             "fresh_checkpoint_checks": checks, "scope": "Stored local checkpoints are audited before compact packaging; checkpoint tensors are omitted from the compact archive."}
    (OUT / "ARTIFACT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
