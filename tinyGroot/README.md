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
python -m tinygroot.infer.chat_infer --help
modal run tinygroot/modal/modal_chat_sft.py::infer --help
```

See [BLOG.md](BLOG.md) for experiment notes, leaderboards, and current results.
