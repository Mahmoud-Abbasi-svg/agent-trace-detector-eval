#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# vLLM 0.29.0 env per amendment 3 (D11).
# MON-L main run, in the registered order (plan Section 10): AgentDojo first (headline), then AgentDrift,
# then MON-L-mask (all 16 masked sets, amendment 2), then MON-L-th. Use mon_l_main_nomask_job.sh plus
# mon_l_mask_job.sh instead if the pilot projects MON-L-mask above 48 GPU hours (amendment 2).
# Resumable across submissions: outputs live in ~/monitor/mon_l (persistent), linked into the workdir,
# so resubmitting the same script continues where a stopped job ended.
module load Python/3.13.5-GCCcore-14.3.0
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
tar xzf monitor_bundle.tgz
cd revision
mkdir -p ~/monitor/mon_l && rm -rf monitor_run/mon_l && ln -s ~/monitor/mon_l monitor_run/mon_l
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
~/venvs/agent029/bin/python monitor_score_vllm.py --groups B1_dojo_p0 B3_dojo_failed B2_drift_test L_drift_rest MASK_0.75_d0 MASK_0.75_d1 MASK_0.75_d2 MASK_0.75_d3 MASK_0.75_d4 MASK_0.5_d0 MASK_0.5_d1 MASK_0.5_d2 MASK_0.5_d3 MASK_0.5_d4 MASK_0.25_d0 MASK_0.25_d1 MASK_0.25_d2 MASK_0.25_d3 MASK_0.25_d4 MASK_0.0_d0 TH_drift_test TH_dojo_all
cp monitor_run/mon_l/main_*.jsonl monitor_run/mon_l/env.json ../ 2>/dev/null || true
ls -lh ../main_*.jsonl || true
