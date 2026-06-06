# RL Environments

## MMLU RL

`rl_envs/mmlu_rl.py` trains a chat checkpoint with the same flat rollout pattern used by `tinygroot.training.chat_rl`:

- all `examples_per_rank * num_samples` rollouts are decoded as one KV-cached batch;
- rewards are exact multiple-choice letter matches;
- rollouts are flattened into one GRPO-style policy-gradient update and microbatched only for memory.

Example:

```bash
torchrun --standalone --nproc_per_node=8 rl_envs/mmlu_rl.py \
  --hf-checkpoint-repo-id harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000 \
  --out-dir /runs/hrm-loop-mmlu-rl \
  --examples-per-step 64 \
  --num-samples 16 \
  --device-batch-size 64 \
  --max-new-tokens 8 \
  --temperature 1.0 \
  --top-k 50 \
  --max-steps 500 \
  --eval-every 50 \
  --save-every 50 \
  --optimizer muon \
  --amp-dtype bfloat16 \
  --wandb \
  --wandb-project tinyGroot-mmlu-rl
```

For a single-process smoke test, lower the batch sizes:

```bash
python rl_envs/mmlu_rl.py \
  --hf-checkpoint-repo-id harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000 \
  --out-dir runs/mmlu-rl-smoke \
  --examples-per-step 2 \
  --num-samples 2 \
  --device-batch-size 2 \
  --max-new-tokens 4 \
  --max-steps 1 \
  --eval-every 0 \
  --save-every 0 \
  --optimizer adamw
```

## Modal

Run the same MMLU RL trainer on Modal:

```bash
modal run rl_envs/modal_run.py \
  --gpu-type H100 \
  --gpu-count 8 \
  --hf-checkpoint-repo-id harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000 \
  --out-dir /runs/hrm-loop-mmlu-rl \
  --max-steps 500 \
  --examples-per-step 64 \
  --num-samples 16 \
  --device-batch-size 64 \
  --max-new-tokens 8 \
  --wandb
```

Optional upload after training:

```bash
modal run rl_envs/modal_run.py \
  --gpu-type H100 \
  --gpu-count 8 \
  --out-dir /runs/hrm-loop-mmlu-rl \
  --push-to-hf \
  --hf-repo-id harshbhatt7585/hrm-loop-mmlu-rl
```

The launcher mounts the existing Modal volumes at `/data` and `/runs`.
