# Run-time deviations of the sentence-encoder run (recorded during execution)

These are departures that arose while executing SENCODER_PREREGISTRATION.md (amendment 6, sha256 cef3e7677710e561496b8427a261d2e34737c7af7908e5ef5809ffb5a43a8493). They are not plan amendments. Each was recorded before any analysis was run and before any test-set result was looked at.

## D13. One fit exceeded the 10-minute stop rule under CPU contention (2026-10-09T09:03:17+02:00)

Slurm job 2762 (cn001, 32 CPUs) ran the embedding pass, the smoke gate and the fits. The fits ran as 15 parallel workers at 2 torch threads each. The stop rule of Section 10 fired as designed. SE R4 seed 1 took 630.1 s, against the 600 s limit, and the job stopped starting new fits. By then 157 of 570 units were done, namely all 130 SE single-fit units and 27 of 50 SE within-AgentDojo UT units. No SE C1, SE-W0 or GRU unit had started.

The same R4 configuration took 156.9 s for seed 0 in the smoke test, run alone with the same 2 threads. That fit stopped early after 21 epochs, which is about 7.5 s per epoch. Seed 1 ran the full 40 epochs. Run alone it would take roughly 300 s at that rate, so the comparison of 630 s with 157 s overstates the slowdown. The slowdown from contention is about a factor of two (630 s against roughly 300 s), not four. Per-setting mean fit times in the contended run were 346 to 437 s, and the next-largest fit took 570.2 s. The overrun is therefore a property of running 15 fits at once on 32 CPUs, not of the model or the data.

*Addendum, written while the resumed run was in progress and before any analysis.* With 6 workers, the SE-W0 R4 fits that ran the full 40 epochs still took 536 to 565 s. The node carried other users' jobs at the same time (CPULoad about 27, 104 CPUs allocated). These fits came within 10% of the limit but did not exceed it.

With the author's agreement, the remaining 413 units are resumed with 6 parallel workers instead of 15. The thread count per fit stays at 2, and every hyperparameter, seed, split and threshold is unchanged. The 157 finished units are kept. Fits are deterministic on CPU at a fixed thread count (smoke test, maximum absolute score difference 0.0 between two runs of the same seed), so the change of worker count does not change any score. The stop rule stays active for the resumed run. The trigger record is kept as revision/sencoder_run/STOP_RULE_TRIGGERED_2762.json on the cluster.

The rule is read here as limiting the compute time of a single fit, not its wall-clock time under external load. A strict reading would have stopped the study at this point.
