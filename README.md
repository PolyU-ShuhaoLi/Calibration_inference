cd Calibration_inference/IntraAfterCoT/deepconf_modify

SFT_MODEL=/path/to/your_sft_model
RL_MODEL=/path/to/your_rl_model
PARALLEL_INFERENCE=1
TENSOR_PARALLEL_SIZE=4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_one_click_8gpu.sh

Here, SFT_MODEL should be "SFT_training/LlamaFactory/saves/Qwen2.5-7B-Instruct/full/train_deepscaler_simplelr_think/checkpoint-500" (这是一个相对路径，找到并改为绝对路径) 


Here, RL_MODEL should be "RL_training/checkpoints/verl-grpo_Qwen2.5-7B-Instruct_rl_data_epochs3_max_response16384_batch16_rollout8_klcoef0.0001_entcoef0.001/global_step_4000" (这是一个相对路径，找到并改为绝对路径)