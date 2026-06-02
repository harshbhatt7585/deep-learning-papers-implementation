# Examples

Minimal, copy-paste steps to go from a clean clone to a sampled completion on a
laptop — no GPU, no Modal, no wandb. Run them from the `tinyGroot/` directory.

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `make doctor` | Sanity-check Python / torch / device |
| 1 | [`01_pretrain_tiny.sh`](01_pretrain_tiny.sh) | Tokenize a small shard + pretrain a tiny model on MPS/CPU |
| 2 | [`02_sample.sh`](02_sample.sh) | Sample text from the checkpoint you just trained |

```bash
uv sync                 # one-time setup
bash examples/01_pretrain_tiny.sh
bash examples/02_sample.sh runs/<the-run-dir-printed-above>
```

For the full pretrain → SFT → RL → eval pipeline on real GPUs, use the Modal
launcher described in the top-level [README](../README.md#-the-pipeline):

```bash
SLUG=run1 bash speed_run.sh pipeline 8gpu
```
