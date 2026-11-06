# Metadata refresh correction; unchanged experiment

First formal RGB control stopped after 100 completed updates: the original
transition copied `measurement`, then refresh passed another keyword with that
name. No candidate arm ran. Preserve its initial state, failure JSON, progress,
stdout and pipeline/active states under `attempt1_failed_refresh`; there is no
complete 100-step checkpoint and no attempted partial continuation.

Independent `repair_run.py` removes only the old measurement metadata before
calling the frozen original transition. Its status, counters, thresholds and
current measurement are unchanged. Frozen training sources, source_freeze,
protocol, pairs, parent, two Adams, rates, losses and all gates remain exact.
Both arms restart from the original full J6, not from partial work. Existing
qualification artifacts are preserved; repaired entrypoint repeats 20-step
smokes and independent exact-state/fixed-gradient Adam recovery. A separate
regression covers the refresh merge beyond 600 calls and hysteresis states.
The new repair files/tests are separately frozen before the restart.
