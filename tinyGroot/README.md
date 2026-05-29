<p align="center">
  <img src="assets/tinygroot-logo.png" alt="tinyGroot logo" width="360">
</p>

# tinyGroot

Small-scale language model experiments around nanochat-style training, MTP,
Token Superposition Training, and chat SFT.

## Layout

```text
tinygroot/
  training/   train, SFT, and pretokenization entrypoints
  infer/      checkpoint sampling, chat inference, and speculative decoding
  modal/      Modal launchers for remote training and inference
```

Common commands:

```bash
python -m tinygroot.training.train --help
python -m tinygroot.training.chat_rl --help
python -m tinygroot.infer.chat_infer --help
modal run tinygroot/modal/modal_chat_sft.py::infer --help
```

Run nanochat-style GSM8K RL after SFT:

```bash
python -m tinygroot.training.chat_rl \
  --checkpoint runs/sft-run/checkpoint.pt \
  --out-dir runs/rl-run \
  --num-epochs 1 \
  --examples-per-step 16 \
  --num-samples 16
```

On Modal:

```bash
modal run tinygroot/modal/modal_train.py::rl \
  --checkpoint /runs/sft-run/checkpoint.pt \
  --out-dir /runs/rl-run
```

## Hugging Face Uploads

Upload an existing checkpoint directory:

```bash
HF_TOKEN=... python -m tinygroot.hf_upload \
  --checkpoint-dir runs/my-run \
  --repo-id username/my-run
```

Push automatically after local training:

```bash
HF_TOKEN=... python -m tinygroot.training.train ... \
  --push-to-hf \
  --hf-repo-id username/my-run
```

Push automatically after Modal training:

```bash
modal secret create huggingface HF_TOKEN=...
modal run tinygroot/modal/modal_train.py::main ... \
  --push-to-hf \
  --hf-repo-id username/my-run
```

See [BLOG.md](BLOG.md) for experiment notes, leaderboards, and current results.
