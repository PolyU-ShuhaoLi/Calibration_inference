cd Calibration_inference/IntraAfterCoT/deepconf_modify
conda activate deepconf
export MODEL=/data/llm/opd_file/opd/outputs/qwen25_7b_gold_math_rl_48k_vllm_zero2/merged_full_model
bash run_opd_model_inference_pt2.sh
bash run_opd_model_inference_pt3.sh
bash run_opd_model_inference_pt4.sh
bash run_opd_model_inference_pt5.sh

针对第一个报错：主要是RL ckpt路径所导致，需要在global_step_4000后面新增子路径: /actor/huggingface/，您可以依照README重新指定一下

针对第二个报错：我认为是vllm和transformers库版本不匹配导致的。建议您先看一下现在环境的版本：
```bash
python -c "import transformers, vllm; print(transformers.__version__); print(vllm.__version__)"

输出是否为：

4.57.6
0.10.2


若不是，您可以这样重新安装transformers（vllm版本大概是对的，transformers大概是不对的）：

  python -m pip uninstall -y transformers
  python -m pip install "transformers<5" scikit-learn accelerate matplotlib numpy scipy

```

SFT_MODEL: SFT_training/LlamaFactory/saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplelr_think/checkpoint-500 (这是一个相对路径，找到并改为绝对路径) 

RL_MODEL: RL_training/checkpoints/verl-grpo_Qwen2.5-7B-Instruct_rl_data_epochs3_max_response16384_batch16_rollout8_klcoef0.0001_entcoef0.001/global_step_4000/actor/huggingface/ (这是一个相对路径，找到并改为绝对路径)

SFT_MODEL=/path/to/your_sft_model

RL_MODEL=/path/to/your_rl_model

PARALLEL_INFERENCE=1

TENSOR_PARALLEL_SIZE=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash run_one_click_8gpu.sh

这里 RUN_ROOT 是：

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/full_${RUN_TAG}}"

也就是说 RUN_ROOT 为刚刚完成推理的这个 runs/full_... 路径

运行完后，文件在：

- `REPORT_DIR` (default: `<RUN_ROOT>/reports_rerun_<timestamp>`)
- `FIG_DIR` (default: `<RUN_ROOT>/figures_rerun_<timestamp>`)

您看是否方便将 REPORT_DIR 和 FIG_DIR 路径下的文件导出，都是KB级别的文件，加起来的量应该也就几MB.

---

4.3 分析脚本 Update

```bash

cd 到 Calibration_inference/IntraAfterCoT/deepconf_modify 路径下

运行 bash postprocess.sh就好了

RUN_ROOT=/path/to/your/runs/full_YYYYMMDD_HHMMSS \
SFT_MODEL=/path/to/your_sft_model \
RL_MODEL=/path/to/your_rl_model \
ENV_NAME=deepconf \
bash postprocess.sh

这里指定 RUN_ROOT 为您群里说的这个路径：runs/full_YYYYMMDD_HHMMSS/sft_offline 和 runs/full_YYYYMMDD_HHMMSS/rl_offline 里有一些文件和log。

如果命令行先前指定过 SFT_MODEL ， RL_MODEL 和 ENV_NAME，其实也不需要额外指定了

运行完查看最新时间的 reports_rerun_<timestamp> 和 figures_rerun_<timestamp> 是否有文件产出，有的话导出

```

4.12 Update:

export PKL_DIR=/data/llm/Calibration_inference/IntraAfterCoT/deepconf_modify/sft_thinking_aime2024
export MODEL=/data/llm/RL_model/simpleRL-reason-ckpts/verl-grpo_Qwen2.5-7B-Instruct_simplelr_qwen_level3to5_epochs1_max_response4096_batch2_rollout8_klcoef0.0001_entcoef0.001/global_step_1/actor/huggingface

export MODEL=/data/llm/LlamaFactory/saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplelr_think/checkpoint-5/global_step5

bash run_sft_model_inference.sh
