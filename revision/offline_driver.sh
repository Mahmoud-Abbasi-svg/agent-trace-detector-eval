#!/usr/bin/env bash
# Offline driver: finishes the linear and GRU prediction caches without any network access.
# Resumable: rerunning skips finished units. Set AGENTDRIFT and AGENTDOJO to local clones first
# (see revision/REPRODUCIBILITY.md); PY defaults to "python".
: "${AGENTDRIFT:?set AGENTDRIFT to a clone of AgentDrift at commit 014a514f}"
: "${AGENTDOJO:?set AGENTDOJO to a clone of AgentDojo at commit 089ed468}"
export AGENTDRIFT AGENTDOJO
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p logs
L=logs/offline_driver.log
echo "$(date -Iseconds) start" >> $L

# 1) linear cache: full rebuild of every setting (an interrupted run may have left a half-written file)
$PY build_linear_cache.py --all --force --budget 100000 >> logs/offline_linear.log 2>&1
echo "$(date -Iseconds) linear exit $?" >> $L

# 2) GRU cache, primary units first
run() { $PY build_gru_cache.py run "$1" "$2" --budget 100000 >> logs/offline_gru.log 2>&1; echo "$(date -Iseconds) gru $2 $1 exit $?" >> $L; }
for s in R1 R3_medical R3_banking R3_web R3_email R3_coding R4; do run $s GRU; done
for s in R1 R3_medical R3_banking R3_web R3_email R3_coding R4; do run $s GRU-drop0.2; done
for s in R2 WUT R1wd_s0 R1wd_s1 R1wd_s2 R1wd_s3 R1wd_s4; do run $s GRU; done
for s in R1 R3_medical R3_banking R3_web R3_email R3_coding R4; do run $s GRU-drop0.1; done
$PY build_gru_cache.py status >> $L 2>&1
$PY build_gru_cache.py summarize >> logs/offline_gru.log 2>&1
echo "$(date -Iseconds) ALL DONE" >> $L
