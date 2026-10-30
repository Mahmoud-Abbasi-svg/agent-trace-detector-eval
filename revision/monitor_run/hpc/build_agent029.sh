#!/bin/bash
# Build ~/venvs/agent029 (vLLM 0.29.0) for MON-L. vLLM 0.30.0 rejects the registered checkpoint (actorder=group). Amendment 3, D11.
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
date
V=$HOME/venvs/agent029/bin
python -m venv $HOME/venvs/agent029 || { echo BUILD_FAIL venv; exit 1; }
$V/pip install -q --progress-bar off --upgrade pip > $HOME/monitor/pip029.log 2>&1
$V/pip install --progress-bar off "vllm==0.29.0" "transformers==5.18.0" >> $HOME/monitor/pip029.log 2>&1 || { tail -n 5 $HOME/monitor/pip029.log; echo BUILD_FAIL pip; exit 1; }
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cat > $HOME/monitor/check029.py <<'EOF'
import json, glob, os, inspect
import vllm, torch, transformers, compressed_tensors
from vllm import LLM, SamplingParams
import vllm.model_executor.models.llama
import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors as ctq
from compressed_tensors.quantization import QuantizationArgs
print("versions", vllm.__version__, torch.__version__, transformers.__version__, compressed_tensors.__version__)
print("rejects_actorder_group", "no longer supported" in inspect.getsource(ctq))
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--RedHatAI--Llama-3.3-70B-Instruct-quantized.w4a16/snapshots/da23851b*"))[0]
qc = json.load(open(os.path.join(snap, "config.json")))["quantization_config"]
for k, g in qc["config_groups"].items():
    a = QuantizationArgs.model_validate(g["weights"]); print("group", k, a.strategy, a.num_bits, a.group_size, a.actorder)
try:
    from vllm.sampling_params import StructuredOutputsParams; print("StructuredOutputsParams ok")
except Exception as e:
    print("StructuredOutputsParams missing", e)
from vllm.engine.arg_utils import EngineArgs
print("logprobs_mode field", "logprobs_mode" in EngineArgs.__dataclass_fields__)
EOF
if $V/python $HOME/monitor/check029.py; then echo BUILD_OK
else
  echo "retry with transformers below 5.17"
  $V/pip install --progress-bar off "transformers>=5.10.4,<5.17" >> $HOME/monitor/pip029.log 2>&1
  $V/python $HOME/monitor/check029.py && echo BUILD_OK_TF_FALLBACK || echo BUILD_FAIL check
fi
date
