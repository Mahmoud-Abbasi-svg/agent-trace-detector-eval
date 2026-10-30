#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# vLLM 0.29.0 env per amendment 3 (D11).
# MON-L main run, in the registered order (plan Section 10): AgentDojo first (headline), then AgentDrift,
# then MON-L-th. MON-L-mask runs afterwards in mon_l_mask_job.sh (amendment 2, used only if the pilot projects
# MON-L-mask above 48 GPU hours).
# Resumable: rerun the same job to continue after a stop.
module load Python/3.13.5-GCCcore-14.3.0
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
tar xzf monitor_bundle.tgz
cd revision
~/venvs/agent029/bin/python monitor_score_vllm.py --groups B1_dojo_p0 B3_dojo_failed B2_drift_test L_drift_rest TH_drift_test TH_dojo_all
cp monitor_run/mon_l/main_*.jsonl monitor_run/mon_l/env.json ../ 2>/dev/null || true
ls -lh ../main_*.jsonl || true
