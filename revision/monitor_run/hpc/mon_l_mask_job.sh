#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# vLLM 0.29.0 env per amendment 3 (D11).
# MON-L-mask as a separate job (amendment 2): all 16 masked sets on the AgentDrift R1 test set.
# Resumable: rerun the same job to continue after a stop.
module load Python/3.13.5-GCCcore-14.3.0
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
tar xzf monitor_bundle.tgz
cd revision
~/venvs/agent029/bin/python monitor_score_vllm.py --groups MASK_0.75_d0 MASK_0.75_d1 MASK_0.75_d2 MASK_0.75_d3 MASK_0.75_d4 MASK_0.5_d0 MASK_0.5_d1 MASK_0.5_d2 MASK_0.5_d3 MASK_0.5_d4 MASK_0.25_d0 MASK_0.25_d1 MASK_0.25_d2 MASK_0.25_d3 MASK_0.25_d4 MASK_0.0_d0
cp monitor_run/mon_l/main_*.jsonl monitor_run/mon_l/env.json ../ 2>/dev/null || true
ls -lh ../main_*.jsonl || true
