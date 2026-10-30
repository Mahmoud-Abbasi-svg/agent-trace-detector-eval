#SBATCH --gres=gpu:1
#SBATCH --time=180
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# vLLM 0.29.0 env per amendment 3 (D11).
# MON-L smoke test (20 calls) then the pre-registered pilot (300 AgentDrift validation traces, 1,738 calls).
# The pilot stores no conclusion and no score (plan Section 10). Inputs: monitor_bundle.tgz staged into the workdir.
module load Python/3.13.5-GCCcore-14.3.0
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
tar xzf monitor_bundle.tgz
cd revision
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
SECONDS=0
~/venvs/agent029/bin/python monitor_score_vllm.py --smoke 20 --groups pilot_val
echo "smoke done after ${SECONDS}s"
~/venvs/agent029/bin/python monitor_score_vllm.py --pilot --groups pilot_val
echo "pilot done after ${SECONDS}s"
cp monitor_run/mon_l/*.jsonl monitor_run/mon_l/env.json ../ 2>/dev/null || true
ls -lh ../*.jsonl ../env.json || true
