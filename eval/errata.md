# Errata to the pre-registration

`eval/prereg.json` is kept exactly as frozen. Statements in it found to be wrong after scoring are corrected here, with the date and the evidence; no protocol parameter, rule or prediction is changed.

## 2026-10-08: determinism and pairing

**Frozen text** (`pairing.determinism`): "Seeded GR00T inference is not bitwise reproducible on A10G (dev seeds 1001, 1002 diverged), so pairing uses the pre-registered fallback: same seed and same phase trigger."

**Correction.** The check behind this sentence was invalid: its two runs used `--rep 0` and `--rep 1`, and `scripts/run_episode.py` sets `policy_seed = seed + 1000000 * rep`, so they drew different policy noise streams by design and had to diverge. In the scored data, runs with the same seed and rep do reproduce: in all 29 non-void (seed, phase) groups, the four stop modes reached the same simulator state hash at t_I (`results/results.json` → `summary.pairing_exact`). The pairing is therefore exact up to the interrupt, which is the pre-registered primary design; the fallback rule (same seed, same phase trigger) gives the same t_I when trajectories are identical, so no episode or number changes.

## 2026-10-08: size of the old ReturnBase contacts

**Frozen text** (`analysis.after_window_contact`): "dev episodes on two layouts showed 300-500 N contacts during ReturnBase before the ReturnBase fix."

**Correction.** The largest 3-step mean contact in those dev episodes was 365 N (`results/dev/returnbase_before_fix.txt`); "300-365 N" is the accurate range.
