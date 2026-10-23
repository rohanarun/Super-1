# Perturbation and annealing publication

Both completed pilots are prepared for a pull request to rohanarun/Super-1,
from a new branch on JRowe47/Super-1, based on `492a87a5e263e806b0a2581c57259e81f1a000e6`.

- [State-sketch perturbations](../reports/state-sketch-perturb-20261018-v1.md):
  54 outcomes; cheap descriptors, no useful optimization gain; no advance.
- [Standalone annealing](../reports/phase-annealing-20261019-v1.md):
  99 outcomes; primary sandwich gate failed on all three architectures.
  The GRU post-SA nominal pass retained unchanged weights, not learned improvement.
- [Publication receipt](../results/perturbation-annealing-publication-20261019-v1/IMPORT.json)
- [Scoped verification](../results/perturbation-annealing-publication-20261019-v1/VERIFICATION.json)

All three architectures and all seeds/controls are retained. The two protocols
use different schedules and confirmation sets: do not compare their absolute CE
as a pooled leaderboard. Original snapshot bytes and source/protocol hashes are
preserved. Publication does not start another run or promote an optimizer.
