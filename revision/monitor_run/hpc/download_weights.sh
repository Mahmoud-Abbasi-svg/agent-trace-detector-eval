#!/bin/bash
# Run DETACHED on login01 (compute nodes have no internet):
#   nohup setsid bash ~/monitor/download_weights.sh > ~/monitor/download.log 2>&1 < /dev/null & disown; exit 0
# Then poll: tail -3 ~/monitor/download.log   (last line DOWNLOAD_OK or DOWNLOAD_FAILED)
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0 || exit 1
export HF_HOME=$HOME/.cache/huggingface
REPO=RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16
REV=da23851b92f5ea8023d6c70f790feb8e05269812
HF=~/venvs/agent/bin/hf; [ -x "$HF" ] || HF=~/venvs/crywolf/bin/hf
date; echo "using $HF"
"$HF" download "$REPO" --revision "$REV" || { echo DOWNLOAD_FAILED; exit 1; }
D=$HF_HOME/hub/models--RedHatAI--Llama-3.3-70B-Instruct-quantized.w4a16
mkdir -p "$D/refs" && printf '%s' "$REV" > "$D/refs/main"     # offline load by name needs refs/main
S="$D/snapshots/$REV"
ok=1
check() { a=$(stat -L -c %s "$S/$1" 2>/dev/null || echo 0); [ "$a" = "$2" ] && echo "size ok $1" || { echo "SIZE MISMATCH $1 $a != $2"; ok=0; }; }
check model-00001-of-00008.safetensors 4949695672
check model-00002-of-00008.safetensors 4978039408
check model-00003-of-00008.safetensors 4978154432
check model-00004-of-00008.safetensors 4934885512
check model-00005-of-00008.safetensors 4978039528
check model-00006-of-00008.safetensors 4978039520
check model-00007-of-00008.safetensors 4978154432
check model-00008-of-00008.safetensors 4750578576
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/venvs/agent/bin/python -c "
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained('$REPO', revision='$REV'); t = AutoTokenizer.from_pretrained('$REPO', revision='$REV')
print('offline load ok', c.model_type, c.num_hidden_layers, len(t))" || ok=0
du -shL --apparent-size "$S"; date
[ $ok = 1 ] && echo DOWNLOAD_OK || echo DOWNLOAD_FAILED
