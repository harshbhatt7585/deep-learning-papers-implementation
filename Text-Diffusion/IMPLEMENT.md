# Small LLaDA-Style Text Diffusion Implementation

This guide implements a small text diffusion language model that is closer to
LLaDA2.1-mini's inference algorithm while staying readable.

The real LLaDA2.1-mini model is much larger: it is a 16B-parameter
Mixture-of-Experts diffusion language model with RoPE positions, a large
tokenizer, and pretrained weights. This repo intentionally does not implement
MoE or load those weights. It does implement the important LLaDA-style decoding
pieces: masked token denoising, block-causal diffusion attention, confidence
thresholds, and token editing.

## Core Idea

Autoregressive language models generate left to right:

```text
prompt -> token 1 -> token 2 -> token 3
```

Text diffusion models generate by starting with unknown tokens and repeatedly
denoising them:

```text
prompt + [MASK] [MASK] [MASK] [MASK]
prompt + "t"    [MASK] "e"    [MASK]
prompt + "t"    "h"    "e"    [MASK]
prompt + "t"    "h"    "e"    "n"
```

The model is trained to recover original tokens from randomly masked text.
At inference time, the model fills multiple masked positions in parallel.

## Step 1: Define Special Tokens

You need at least these token IDs:

- `pad_token_id`: used only for batching variable-length sequences.
- `mask_token_id`: the diffusion placeholder the model must denoise.
- `eos_token_id`: optional end-of-sequence marker.

In `tokenizer.py`, `SimpleCharTokenizer` builds a tiny character-level vocabulary:

```python
tokenizer = SimpleCharTokenizer.from_texts(["hello world"])
ids = tokenizer.encode("hello", add_eos=True)
```

This keeps the example independent of Hugging Face tokenizers.

## Step 2: Build A Denoiser Transformer

The denoiser receives a full sequence containing real tokens and mask tokens.
It predicts the clean token at every position.

The model is still a normal small Transformer:

```text
token ids -> token embeddings + position embeddings
          -> Transformer blocks
          -> vocabulary logits at every position
```

Unlike a standard causal language model, the denoising pass can use a custom
attention mask. During training, the toy implementation uses a simple padding
mask. During generation, it uses a LLaDA-style block diffusion mask.

Implemented in `TextDiffusionModel`.

## Step 3: Add Forward Diffusion

Forward diffusion corrupts clean text by replacing random tokens with
`[MASK]`.

Example:

```text
clean:   the cat sat
noised:  the [MASK] s[MASK]t
target:      cat    a
```

In code:

```python
noised, labels = make_masked_inputs(
    input_ids,
    mask_token_id=config.mask_token_id,
    pad_token_id=config.pad_token_id,
    mask_prob=0.30,
)
```

Only masked positions contribute to the loss. Non-masked labels are set to
`-100`, which PyTorch cross entropy ignores.

## Step 4: Train With Denoising Loss

Training is standard cross entropy over masked positions:

```python
loss = diffusion_loss(model, input_ids, mask_prob=0.30)
loss.backward()
optimizer.step()
```

The model learns:

```text
given corrupted text -> predict original clean tokens
```

## Step 5: Generate With A Mask Template

Generation starts by appending masks after the prompt and rounding the total
length up to a multiple of `block_length`:

```text
prompt:   "hello "
state:    "hello " + [MASK] [MASK] [MASK] [MASK] ...
```

The generator processes output in aligned blocks, like LLaDA:

```python
generated = generate(
    model,
    prompt_ids,
    gen_length=32,
    block_length=8,
    steps=8,
    threshold=0.70,
    editing_threshold=0.90,
    max_post_steps=16,
)
```

For each block:

1. Build a block-causal attention mask.
2. Sample or greedily choose candidate tokens for every masked position.
3. Accept high-confidence candidates.
4. If too few candidates pass the threshold, accept the most confident one.
5. Optionally edit already-filled non-prompt tokens.
6. Repeat until all masks in the block are filled and edit passes are done.
7. Move to the next block.

This is the key difference from autoregressive decoding: several positions can
be filled in one model call.

## Step 6: Use Block Diffusion Attention

LLaDA does not use a plain left-to-right causal mask during diffusion
generation. It uses blocks:

```text
block 0 can attend to block 0
block 1 can attend to block 0 and block 1
block 2 can attend to block 0, block 1, and block 2
```

Inside the active block, tokens can see each other. This is what lets the model
denoise multiple positions in parallel.

Implemented in:

```python
build_block_diffusion_attention_mask(...)
```

and passed into:

```python
model(x[:, :block_end], attention_mask=attention_mask)
```

## Step 7: Add Editing

LLaDA2.1-mini highlights editable generation. After a token is filled, later
denoising passes may replace it if the model becomes highly confident in a
different token.

In `model.py`, this is controlled by:

```python
editing_threshold=0.90
```

Set it to `None` to disable editing.

Prompt tokens are never edited.

## Step 8: Run The Example

From this directory:

```bash
python model.py
```

The script builds a toy tokenizer, creates a tiny model, runs a few training
steps on a tiny character dataset, and calls the diffusion generator.

Do not expect useful language quality from this toy run. It is intentionally
small so you can understand and modify it.

## Step 9: What Is Now Closer To LLaDA2.1-mini

The implementation now matches these high-level LLaDA2.1 ideas:

- Full masked output template instead of token-by-token generation.
- Block-aligned generation with `block_length`.
- Lower-triangular block attention, with full attention inside each block.
- Confidence threshold for accepting denoised mask tokens.
- Fallback to the most confident tokens when the threshold accepts too few.
- Optional token editing with `editing_threshold`.
- Post-fill edit passes controlled by `max_post_steps`.
- Greedy, top-k, and top-p sampling support.

## Step 10: What Is Still Simplified

After you understand `model.py`, scale one part at a time:

- Replace character tokenization with a real subword tokenizer.
- Replace learned absolute positions with RoPE.
- Replace the tiny MLP blocks with larger Transformer blocks.
- Add Mixture-of-Experts feed-forward layers.
- Train on a real corpus instead of the toy strings.
- Match the exact Hugging Face remote-code API if you want drop-in parity.

## Files

- `model.py`: model, denoising loss, and LLaDA-style generation loop.
- `tokenizer.py`: tiny character tokenizer and padding helper.
- `IMPLEMENT.md`: this learning guide.

## Reference Points

- LLaDA2.1-mini Hugging Face model card:
  `https://huggingface.co/inclusionAI/LLaDA2.1-mini`
- LLaDA2.1-mini generation code:
  `https://huggingface.co/inclusionAI/LLaDA2.1-mini/blob/main/modeling_llada2_moe.py`
