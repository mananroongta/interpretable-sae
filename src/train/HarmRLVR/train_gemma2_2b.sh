#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN env 

unset CUDA_VISIBLE_DEVICES

LOG_DIR="${ROOT_DIR}/logs/gemma2b_air64"
mkdir -p "${LOG_DIR}"/{rollout,val}

python -m HarmRLVR.prepare_harmrlvr_data

ray stop || true

python -m HarmRLVR.train \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=64 \
    data.max_prompt_length=500 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=google/gemma-2-2b-it \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=gemma2_2b_it_adversarial \
    trainer.experiment_name=gemma2_2b_it_adversarial \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.rollout_data_dir="${LOG_DIR}/rollout" \
    trainer.validation_data_dir="${LOG_DIR}/val" \
    "$@"
