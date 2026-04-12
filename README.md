- 文档中的路径示例大多是相对路径，实际执行时请替换为绝对路径
- "..." 表示命令执行时自动生成的时间戳（time tag）

- `SFT模型`
  Calibration/SFT_training/LlamaFactory/saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplelr_think/checkpoint-500 
  (这是一个相对路径，您需要在用到时指定这个路径的绝对路径) 

- `RL模型`
  Calibration/RL_training/checkpoints/verl-grpo_Qwen2.5-7B-Instruct_rl_data_epochs3_max_response16384_batch16_rollout8_klcoef0.0001_entcoef0.001/global_step_4000/actor/huggingface/ 
  (这是一个相对路径，您需要在用到时指定这个路径的绝对路径) 

您需要运行：

1. RL的结果呈现

   ```bash
   conda activate deepconf
   cd Calibration_inference/IntraAfterCoT/deepconf_modify
   export PKL_DIR=Calibration_inference/IntraAfterCoT/deepconf_modify/runs/full_.../rl_offline  （您上次说的有30个pkl文件的文件夹）
   export MODEL=RL模型的绝对路径
   bash postprocess_single_model.sh
   ```

然后，约5分钟运行完毕，输出目录为：
`Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_...`
请您拍这三个文件：
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../reports/passk.json`
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../figures/figure1_points.csv`
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../reports/metrics_summary.txt`

---

2. 运行SFT的推理和结果呈现：

```bash
export MODEL=SFT模型的绝对路径
export TOKENIZER=Calibration/SFT_training/LlamaFactory/.ms_cache/models/Qwen2.5-7B-Instruct （您需要找到这个路径，然后指定为绝对路径）
bash run_sft_model_inference.sh

这个命令运行时间久一些，估计需要半天
跑完之后，会有一个新的时间tag，然后里面应该是有30个pkl文件
```

```bash
这时候指定这个跑完之后的路径

export PKL_DIR=Calibration_inference/IntraAfterCoT/deepconf_modify/runs/full_.../sft_offline 

bash postprocess_single_model.sh
```

然后，约5分钟运行完毕，输出目录为：
`Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_...`
请您拍这三个文件：
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../reports/passk.json`
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../figures/figure1_points.csv`
- `Calibration_inference/IntraAfterCoT/deepconf_modify/runs/postprocess_single_.../reports/metrics_summary.txt`
