# PIDG progress audit

Checked: 2026-10-15T15:53:18.832086+01:00

Long runs: {'training': 1, 'queued': 74}; total 75.
Active: Phase 1 / mechanics_cloth, 7690/50,000 (15.38%).
Observed advancement: 100 steps in 10.01 seconds.
Controller alive: True; trainer alive: True.
Checkpoint verified: 7000; SHA256, size, optimizer and RNG flags checked.
GPU: NVIDIA GeForce RTX 5070 Ti, 100 %, 10718 MiB, 16303 MiB; free disk 405.47 GiB.
Pending stop requests: [].

## Phase status

| Phase | Code status | Final-revision smoke | Long training | Source hashes match |
|---|---|---|---|---|
| 1 | implemented | completed | running | True |
| 2 | implemented | pending | pending | True |
| 3 | implemented | pending | pending | True |
| 4 | implemented | pending | pending | True |
| 5 | implemented | pending | pending | True |
| 6 | implemented | pending | pending | True |
| 7 | implemented | pending | pending | True |
| 8 | implemented | pending | pending | True |
| 9 | implemented | pending | pending | True |
| 10 | implemented | pending | pending | True |
| 11 | implemented | pending | pending | True |
| 12 | implemented | pending | pending | True |
| 13 | implemented | pending | pending | True |
| 14 | implemented | pending | pending | True |
| 15 | implemented | pending | pending | True |

All 15 algorithm stages have earlier GPU smoke receipts; final source revisions are revalidated sequentially before their long runs. No new long-run performance comparison is available until training, rendering and official metrics finish.

See [results.md](results.md) for all 75 runs and [progress_check.json](progress_check.json) for evidence. This is a timestamped snapshot; the live queue continues.
