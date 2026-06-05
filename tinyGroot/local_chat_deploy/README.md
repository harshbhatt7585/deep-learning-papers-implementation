# Local tinyGroot Chat

Run a browser chat UI against:

```text
harshbhatt7585/hrm-loop-pdr12-sft-gsm8k8-mmlu5-1000
```

Start it from the repository root:

```bash
uv run python local_chat_deploy/server.py
```

Then open:

```text
http://127.0.0.1:7860
```

The server downloads the checkpoint into `runs/hf_checkpoints`, loads it once, and runs generation on MPS by default. The UI keeps the visible chat transcript, but defaults model context to the latest turn because this small checkpoint degrades quickly when its earlier completions are fed back in. Use the Context control to try recent/full history.

If you need to use an already downloaded checkpoint directory:

```bash
uv run python local_chat_deploy/server.py --checkpoint runs/hf_checkpoints/<snapshot-dir>
```

CPU fallback is intentionally off. To bypass that:

```bash
uv run python local_chat_deploy/server.py --allow-cpu-fallback
```
