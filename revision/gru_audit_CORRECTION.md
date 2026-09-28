# Correction to `gru_audit.json` (v0.4 revision, 27 September 2026)

`gru_audit.json` in the project root (and its v0.3 copy, `archive_v0.3/gru_audit.json`, which is byte-identical, SHA-256 `30e43fbeda56842ff96bae4ef28ef066217806b3407da7c1c4e838a23347de8b`) contains two records that do not describe the GRU fits that produced the reported results. The JSON file is left unchanged, because it is the output of the audit script as it ran. This note records the error instead.

## What the file says

- `"R1_val_source": "data/val.jsonl (1880)"`
- `"R1"` → `"n_val": 1880`

## What the code did

- Every GRU fit, including every R1 fit, stopped early on a stratified 10% hold-out of that rung's own training data. The hold-out is drawn by `gru_ladder.stratified_val` with `numpy.random.default_rng(1000 + training seed)`, separately for each class (`gru_ladder.py` lines 17 to 23), and the model is trained on the remaining 90% (`gru_ladder.py` lines 50 and 108). For R1 the hold-out is therefore 10% of the 8,775 R1 training trajectories, not the 1,880 trajectories of `data/val.jsonl`.
- `data/val.jsonl` is loaded only by the audit function (`gru_ladder.py` line 59), which counts it and writes the string above into the audit file (`gru_ladder.py` line 73). No GRU fit reads it.
- The audit record `"other_val_source": "stratified 10% of that rung's train"` is correct, and it applies to R1 as well.

## Consequences

- No reported number changes. The published R1 GRU AUROC (0.936) was produced by the code described above, and the v0.4 prediction cache reproduces it with the same validation split (`revision/DEVIATIONS.md`, entry G1).
- The error concerns the description of the procedure. It is a deviation from `GRU_PREREGISTRATION.md` Section 4, which names `data/val.jsonl` for R1.

## Related records

- `GRU_RESULTS.md` repeats the error in its split-audit table (R1 `n_val` 1,880). Its appended "Correction (v0.4 revision)" section corrects it.
- `revision/DEVIATIONS.md`, Part B item V15 and Part E entry G1, lists the deviation with file and line references.
- Other deviations of the GRU runs from their plan (within-AgentDojo seeds 0 to 4 with one training seed per repetition, gradient clipping at norm 1.0, R1-wd on split seed 0 only) are listed in `revision/DEVIATIONS.md`, items V16 to V24, and in Table S3 of the v0.4 supplementary material.
