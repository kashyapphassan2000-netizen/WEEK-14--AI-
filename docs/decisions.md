# How this repo was built, step by step, in plain English

This file is for anyone who opens the repo and asks: *what is this, how was it
built, and why does it look the way it does?* No GPU or math background needed.
Each step says **what** was done, **why**, and, where this repo departs from
the build guide it follows, **what changed and why**.

---

## Part 0: The problem in one picture

Modern AI models (ChatGPT-style transformers) use an operation called **attention**.
Every word in a sentence "looks at" every other word to decide what matters.
For a text of N words, that means building a table of N × N scores.

- 1,000 words → 1 million scores
- 8,000 words → 64 million scores, *per attention head, per layer*

A GPU has two kinds of memory. Think of a workshop:

| GPU part | Workshop analogy | Size | Speed |
|---|---|---|---|
| **HBM** (main GPU memory) | The warehouse down the hall | Huge (40–80 GB) | Slow to walk to |
| **SRAM** (on-chip memory) | Your workbench | Tiny (~192 KB per core) | Instant |
| **Compute cores** | Your hands | n/a | Very fast |

**Normal attention** builds the whole N × N score table, carries it to the warehouse,
walks back to fetch it, does one step (softmax), carries it back, fetches it again,
and finally multiplies. The worker's hands are fast, but most of the time goes to
walking to the warehouse and back.

**FlashAttention** (Tri Dao et al., 2022) never builds the full table. It works on
small tiles that fit on the workbench, keeps a running tally, and only walks to the
warehouse to drop off the final answer. Same exact result, far fewer trips.

This repo implements that idea, FlashAttention v1, in **Triton**. Triton is a
Python-like language for writing GPU programs ("kernels") without writing raw CUDA.

---

## Part 1: The order of work, and why that order

The build guide splits the work into 7 days. Each step builds on the one before it:

```
Day 1  Understand the math          → docs/notes.md
Day 2  Warm-up kernel (matmul)      → kernels/matmul.py
Day 3  Slow-but-simple attention    → kernels/attn_naive.py      (the baseline)
Day 4  Fast FlashAttention forward  → kernels/attn_fwd.py        (the main event)
Day 5  Backward pass (for training) → kernels/attn_bwd.py
Day 6  Benchmarks                   → bench/run.py, bench/plot.py
Day 7  Write-up                     → blog/post.md, README.md
```

**Why this order?** Each step gives you something to check the next step against.
You can't trust a fast kernel unless a slow, obviously-correct one exists to compare
it with. You can't claim a speedup without a baseline to be faster than.

---

## Part 2: Step by step

### Step 1: Write down the math first (`docs/notes.md`)

**What:** Wrote out, by hand, how the "running tally" trick works (called *online softmax*).

**The idea, in plain words:** Softmax turns scores into percentages that add up to 100%.
To do that you need the total across *all* scores, which seems to require seeing the
whole row first. The trick: keep a running total, and whenever a new, bigger score
shows up, shrink the old total to match. It's like adding up a bill in mixed
currencies. When you switch to a new currency, you convert the old subtotal once and
keep going. At the end, the answer is exactly right.

Three numbers are tracked per row:
- `m`: the biggest score seen so far (kept so the exponentials don't overflow)
- `l`: the running total (the softmax denominator)
- `acc`: the running weighted sum of values (the answer before dividing by `l`)

**Why first:** If you can't explain the recurrence on paper, you won't be able to
debug it in a kernel.

---

### Step 2: The warm-up kernel, matrix multiply (`kernels/matmul.py`)

**What:** A Triton program that multiplies two matrices by splitting them into tiles.
Each GPU "program" computes one tile of the output.

**Why:** It teaches the three building blocks every later kernel uses:
1. `tl.load` / `tl.store`: move a tile between warehouse and workbench
2. `tl.dot`: multiply two tiles on the fast matrix hardware (tensor cores)
3. The *grid*: how thousands of programs split up the work

**Decisions:**
- **Accumulate in fp32, store in fp16.** Inputs are 16-bit floats (fast, small).
  Adding up thousands of 16-bit numbers loses precision, so the running sum is kept
  in 32-bit and converted back only at the end.
- **Masks on every load/store.** If a matrix is 100 wide and tiles are 32 wide,
  the last tile hangs off the edge. Masks say "ignore the part past the edge".
  Without them you read garbage or crash. Tests deliberately use odd sizes like
  37 × 129 to exercise this.
- **Autotuning.** `@triton.autotune` tries several tile sizes on the actual GPU and
  keeps the fastest. Different GPUs (T4, 4090, A100) prefer different sizes.

---

### Step 3: The slow, simple baseline (`kernels/attn_naive.py`)

**What:** Attention done the textbook way, in three separate GPU programs:
1. Compute the full N × N score table, **write it to HBM** (the warehouse)
2. Read it back, apply softmax row by row, write it again
3. Read it again and multiply by V

**Why:** Two reasons. It's easy to verify, and it's the thing we beat. The
"speedup vs naive" number needs a naive version to measure.

**⚠️ Change from the guide:** The guide's naive kernel tried to load *all* of K and V
for one head onto the workbench at once. At N = 4096 that is 512 KB per tensor, but
the workbench holds about 192 KB. On a real GPU that version fails to compile at the
sizes we benchmark. So this repo writes the score table to HBM, which is what
"standard attention" means in the FlashAttention paper and what plain PyTorch does.
It's slow for the honest reason (warehouse trips), not because of a broken kernel.

---

### Step 4: The fast forward kernel (`kernels/attn_fwd.py`)

**What:** The real FlashAttention. Each GPU program:
1. Loads one tile of queries (say 128 rows) onto the workbench
2. Loops over K and V a tile at a time (say 64 rows each)
3. For each tile: computes scores, updates the running `m`, `l`, `acc`
4. At the end: divides `acc` by `l`, writes the answer to HBM once

The N × N table never exists anywhere.

It also saves one extra number per row, the **log-sum-exp (LSE)**. That's a compact
summary of each row's softmax, and it's all the backward pass needs later.

**Decisions:**
- **`exp2` instead of `exp`.** GPUs have a fast built-in "2 to the power of x"
  instruction. Since `e^x = 2^(x · 1.4427)`, the kernel multiplies by 1.4427 once
  and uses the fast instruction throughout. Same answer, cheaper.
- **Running total starts at 0, max starts at −∞.** The guide says start `l` at 1.0
  "to avoid divide-by-zero". In this kernel it makes no difference: on the first tile
  the old values get multiplied by `2^(−∞) = 0`, which wipes them either way. Starting
  at 0 is the more honest statement of the math.
- **64-bit address math.** Positions in memory are computed as `batch × stride`.
  For big inputs that number exceeds what a 32-bit integer can hold (about 2 billion),
  wraps around, and you silently read the wrong data. The batch/head offsets are cast
  to 64-bit (`.to(tl.int64)`).
- **Plain pointers + masks, not `tl.make_block_ptr`.** The guide uses block pointers,
  an older Triton API (newer Triton favors "tensor descriptors"). Plain pointer
  arithmetic with explicit masks behaves the same on every Triton 3.x version, runs
  in the CPU interpreter, and keeps every "is this row past the end?" decision
  visible in the code. It handles any N (tests use N = 77, 100, 1000).
- **Causal mode done correctly.** "Causal" means word 10 may only look at words 1–10,
  never the future. That's how GPT-style models work. The guide's version skipped
  future tiles but forgot to mask *inside* the tile on the diagonal, so some rows
  could still see a few future words. Here every tile applies the
  `query position >= key position` mask, and fully-future tiles are skipped for speed.
- **Two public functions.** `flash_attn_fwd` returns the output plus the LSE in
  ordinary natural-log units, so you can check it against `torch.logsumexp`.
  Internally the kernel keeps the base-2 version for the backward pass.

---

### Step 5: The backward pass (`kernels/attn_bwd.py`)

**What:** To *train* a model you also need gradients: how much each input should
change to reduce the error. That's the backward pass.

**The trick: recompute instead of remember.** The obvious approach stores the N × N
probability table from the forward pass. That's exactly the huge thing we avoided.
Instead, the backward pass rebuilds each small tile of it on the fly from Q, K and
the saved LSE. Recomputing is cheap; storing N × N is not. That trade (more math,
less memory) is the core of FlashAttention.

**Three small kernels:**
1. `_attn_bwd_preprocess`: one number per row, `D = sum(O × dO)`. Simple and fast.
2. `_attn_bwd_dkdv`: each program owns one tile of K/V and loops over all queries,
   accumulating dK and dV.
3. `_attn_bwd_dq`: each program owns one tile of Q and loops over all keys,
   accumulating dQ.

**Why split dK/dV from dQ?** If one program computed all three, many programs would
need to add into the same dQ rows at the same time. That needs "atomic" adds, which
are slow and give slightly different answers each run. With the split, every output
tile has exactly one owner. The cost is computing the scores twice. That's the same
choice Tri Dao's reference and the official Triton tutorial make.

**⚠️ Changes from the guide:** The guide's backward is labelled "abridged". Its V
pointer reused K's pointer (`v_ptrs = k_ptrs # placeholder`), the dQ kernel was left as
"similarly...", and the backward returned an uncomputed dq. This repo has all three
kernels written out in full, with masking for odd N and causal support.

**Plugging into PyTorch:** `FlashAttnFunc` is a `torch.autograd.Function`. To PyTorch
it looks like a normal layer: call `flash_attention(q, k, v)` and `.backward()`
just works. Forward saves Q, K, V, O and the LSE, never the N × N table.

---

### Step 6: Testing, and how it was done without a GPU

**The test idea:** Run the same random inputs through our kernel and through PyTorch's
built-in attention (`F.scaled_dot_product_attention`), and check the answers match
within a small tolerance (1e-2 for outputs, 2e-2 for gradients, as the guide specifies).

**Decision: compare against a 32-bit reference.** PyTorch's reference is run in fp32,
then compared. Comparing two fp16 results against each other can hide errors, because
both sides carry rounding noise.

**Decision: test the awkward cases, not just the easy ones:**
- Sequence lengths that aren't multiples of the tile size (77, 100, 1000)
- Causal and non-causal
- Head sizes 16, 32, 64, 128
- Non-contiguous inputs (a transposed view, common in real models)
- The LSE checked against `torch.logsumexp` (debug checkpoint 2 in the guide)
- The guide's exact pass gate: N = 2048, D = 64, fp16 (runs on GPU only)

**How it was tested without a GPU:** This repo was built in a cloud container with
no GPU. Triton has an **interpreter mode** (`TRITON_INTERPRET=1`) that runs the exact
same kernel code on the CPU, using numpy to emulate the GPU. It's around 1000× slower,
so tests shrink the sizes on CPU (N ≈ 50–128 instead of 2048). The math, masking, and
loop logic are identical, which makes this a real correctness check.

`tests/conftest.py` decides automatically:
- GPU found → real kernels, full-size tests
- No GPU → interpreter, small tests

**Proving the tests can fail:** A test that never fails proves nothing. Two classic
bugs from the guide's "common errors" appendix were planted on purpose:
- Forgot to multiply dQ by the scale → **7 backward tests failed** ✔
- Forgot to rescale `acc` by `alpha` → **9 forward tests failed** ✔

The bugs were then removed. So the tests catch the mistakes they're meant to catch.

**One honest limitation found:** In interpreter mode, `tl.dot` on **bf16** inputs
returns garbage. A 16 × 16 bf16 matrix multiply came out wrong by about 10¹⁰, while
converting bf16 to fp32 was exact. That's a gap in Triton's CPU emulator (numpy has
no bf16 type), not in these kernels. So the bf16 tests are marked "GPU only" and skip
on CPU. They run on a real GPU.

**Automatic testing on every push:** `.github/workflows/tests.yml` runs the
interpreter tests on GitHub's free CPU machines on every push.

---

### Step 7: Benchmarks (`bench/run.py`, `bench/plot.py`)

**What:** Time three things at sequence lengths 1k, 2k, 4k, 8k:
naive Triton, our flash kernel, and PyTorch's built-in SDPA. SDPA picks its own
fused CUDA kernel: FlashAttention-2 on A100/4090, memory-efficient attention on a
T4. Either way it's the hard target. Matching it is a good result; beating it isn't
expected.

**Decisions:**
- **`triton.testing.do_bench`, not hand-rolled timers.** It warms up the GPU, clears
  caches between runs, and reports the median. Hand-rolled loops tend to flatter
  the kernel.
- **Out-of-memory becomes "nan", not a crash.** At 8k the naive version needs 2 GB
  just for the score table. If the GPU runs out, that cell shows `nan` and the run
  continues.
- **Also reports TFLOP/s and ours ÷ SDPA**, so results compare across GPUs.
- **A forward+backward mode** (`--mode fwd_bwd`) for training speed. Naive is skipped
  there because it has no backward.
- **Results saved as JSON**, and the plot reads the JSON, so you don't have to rerun
  the benchmark to tweak a chart.

**⚠️ Status: benchmark numbers are NOT filled in yet.** They need a real GPU, and this
repo was built without one. The README and blog have empty slots for them on purpose.
Making up numbers would defeat the purpose of a benchmark. To fill them in:

```bash
# On Kaggle (free T4), Colab, or a rented A100:
git clone <this repo> && cd <repo>
pip install -r requirements.txt
bash scripts/run_all_gpu.sh
```

That runs every test at full size, both benchmarks, and the plot.

---

### Step 8: Write-up (`README.md`, `blog/post.md`)

The README has install, quickstart, the results table (to be filled), and the
citation. The blog post follows the guide's 5-section outline and explains the math
from first principles.

---

## Part 3: Summary of every departure from the guide

| # | Guide said | This repo does | Why |
|---|---|---|---|
| 1 | Naive kernel loads all of K, V into SRAM | Writes the N × N matrix to HBM (3 kernels) | Guide version can't fit in SRAM at N ≥ 4096 on real hardware; HBM version is the true "standard attention" baseline |
| 2 | `tl.make_block_ptr` in forward | Plain pointers + masks | Same behavior across Triton 3.x and the CPU interpreter; masking for any N is explicit |
| 3 | Causal: skip future tiles only | Skip future tiles **and** mask the diagonal tile | Guide version leaks a few future tokens per row |
| 4 | `l_i` starts at 1.0 | `l_i` starts at 0.0 | Mathematically identical here (first alpha = 0); 0 is the honest statement |
| 5 | Backward "abridged": V reuses K pointer, no dQ kernel | All three kernels fully written | The abridged version computes wrong gradients |
| 6 | Test with `o.sum().backward()` | That test kept, **plus** random dO and causal | A sum loss gives dO = 1 everywhere, which hides some bugs |
| 7 | Reference = fp16 SDPA | Reference = fp32 SDPA | Stricter, catches more |
| 8 | Manual CUDA-event timing | `triton.testing.do_bench` | More accurate; the guide's own appendix recommends it |
| 9 | Plot script reruns the benchmark | Plot reads saved JSON | Rerun the chart without rerunning the GPU |
| 10 | (not in guide) | CPU interpreter tests + GitHub CI | Lets anyone verify correctness without a GPU |

---

## Part 4: Repo map

```
kernels/
  _utils.py        detects interpreter mode, picks autotune configs
  matmul.py        Day 2 warm-up: tiled matrix multiply
  attn_naive.py    Day 3 baseline: materializes the N x N matrix
  attn_fwd.py      Day 4 FlashAttention forward (online softmax)
  attn_bwd.py      Day 5 backward by recomputation + autograd wrapper
tests/             one test file per kernel; conftest.py picks GPU vs CPU
bench/run.py       timing vs naive and torch SDPA → JSON + markdown table
bench/plot.py      JSON → latency.png
scripts/run_all_gpu.sh   one command on a GPU box: test + bench + plot
docs/notes.md      the math, compact
docs/decisions.md  this file
blog/post.md       the write-up
```

## Part 5: Glossary

- **Kernel**: a function that runs on the GPU, thousands of copies in parallel.
- **Triton**: a Python-like language for writing GPU kernels (by OpenAI).
- **Tile / block**: a small rectangular chunk of a matrix that fits on-chip.
- **HBM**: the GPU's main memory. Big, but slow to reach.
- **SRAM**: the GPU's on-chip memory. Tiny, but fast.
- **Softmax**: turns a row of scores into probabilities that sum to 1.
- **Online softmax**: computing softmax in pieces with a running max and running total.
- **LSE (log-sum-exp)**: a one-number-per-row summary of a softmax, used to rebuild it.
- **Forward / backward pass**: compute the output / compute the gradients for training.
- **fp16 / bf16 / fp32**: 16-bit and 32-bit number formats. 16-bit is faster but less precise.
- **Causal**: each token may only attend to itself and earlier tokens.
- **Autotune**: try several kernel settings on the real GPU and keep the fastest.
- **SDPA**: `torch.nn.functional.scaled_dot_product_attention`, PyTorch's built-in attention.
