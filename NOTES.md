# NOTES: the whole project, step by step, for a complete beginner

This file assumes you have **never** done any of this before. Every technical word is
explained the first time it appears (look for **bold** terms). If a word is unclear later,
check the [Glossary](#glossary) at the bottom.

This is written to be **brutally honest**. Where something is not done, not verified,
or harder than it looks, it says so plainly.

---

## Table of contents

0. [The honest summary: read this first](#0-the-honest-summary-read-this-first)
1. [What this project actually is](#1-what-this-project-actually-is)
2. [What you need before you start](#2-what-you-need-before-you-start)
3. [Step 1: Open a terminal](#step-1-open-a-terminal)
4. [Step 2: Check that Git and Python are installed](#step-2-check-that-git-and-python-are-installed)
5. [Step 3: Download the code (git clone)](#step-3-download-the-code-git-clone)
6. [Step 4: Create a virtual environment](#step-4-create-a-virtual-environment)
7. [Step 5: Install the libraries](#step-5-install-the-libraries)
8. [Step 6: Check the install worked](#step-6-check-the-install-worked)
9. [Step 7: Run the tests on your CPU](#step-7-run-the-tests-on-your-cpu)
10. [Step 8: Read and understand the code, in this order](#step-8-read-and-understand-the-code-in-this-order)
11. [Step 9: Get a free GPU (Kaggle)](#step-9-get-a-free-gpu-kaggle)
12. [Step 10: Run tests and benchmarks on the GPU](#step-10-run-tests-and-benchmarks-on-the-gpu)
13. [Step 11: Put your real numbers in, commit, push](#step-11-put-your-real-numbers-in-commit-push)
14. [Step 12: The PASS/FAIL gate (be strict)](#step-12-the-passfail-gate-be-strict)
15. [Step 13: Tag the release v0.1.0](#step-13-tag-the-release-v010)
16. [Step 14: Publish the write-up](#step-14-publish-the-write-up)
17. [How this repo was actually built: the real log, mistakes included](#how-this-repo-was-actually-built-the-real-log-mistakes-included)
18. [Troubleshooting](#troubleshooting)
19. [Brutal honesty: what this project is worth for jobs](#brutal-honesty-what-this-project-is-worth-for-jobs)
20. [Glossary](#glossary)

---

## 0. The honest summary: read this first

| Item | Status | Honest comment |
|---|---|---|
| All kernel code written (matmul, naive attention, flash forward, flash backward) | ✅ Done | |
| Correctness tests | ✅ 28 pass, 5 skipped, **on CPU only** | Checked with Triton's CPU emulator, not a real GPU |
| Tests on a real GPU | ❌ **Never run** | Built in a cloud machine with no GPU. Very likely to pass (same code, same math), but "very likely" is not "verified" |
| Speed benchmarks | ❌ **No numbers exist** | Need a GPU. README/blog tables are empty on purpose. Nothing was invented |
| bf16 support | ⚠️ Written, **never tested** | Can only run on an A100 / RTX 30xx / RTX 40xx class GPU |
| Release tag v0.1.0 | ❌ Not done | You must not tag until the GPU tests and the speed check pass (Step 12) |
| Blog post | ⚠️ Written, results section empty | Fill it from your GPU run |

**The single most important thing:** the code was written by an AI assistant (Claude).
If you put this on your CV and cannot explain every line in an interview, you will be
caught within minutes. The interview is the real test, not the repo. Steps 8 and 19
tell you how to actually own it.

---

## 1. What this project actually is

### The 30-second version

AI models like ChatGPT are built from a part called a **transformer** (a type of neural
network, a program that learns patterns from data). Inside every transformer is an
operation called **attention**: every word in the input "looks at" every other word to
decide which ones matter.

If the input has N words, attention builds an N × N table of scores. For 8,000 words
that's 64 million numbers, per layer, per "head" (a transformer runs many attention
copies in parallel, called **heads**).

**FlashAttention** (a 2022 research paper by Tri Dao and others) is a smarter way to
compute *exactly the same answer* without ever building that giant table. It's faster
and uses much less memory. It's used in almost every modern large AI model.

**This project re-implements FlashAttention (version 1) from scratch**, in a language
called **Triton**, then tests it and measures its speed.

### Why the giant table is a problem (the workshop analogy)

A **GPU** (Graphics Processing Unit) is the chip AI runs on. It's thousands of small
calculators working at once. It has two kinds of memory:

| GPU part | Analogy | Size | Speed |
|---|---|---|---|
| **HBM** (High Bandwidth Memory, the GPU's main memory) | A warehouse down the hall | Big: 16–80 GB | Slow to reach |
| **SRAM** (on-chip memory) | Your workbench | Tiny: ~100–200 KB per core | Instant |
| Compute cores | Your hands | n/a | Extremely fast |

Normal attention builds the whole table, carries it to the warehouse, walks back to
get it, does one step, carries it back, fetches it again… Your hands are fast, but
you spend all your time walking.

FlashAttention works on small pieces (**tiles**) that fit on the workbench, keeps a
running total, and only walks to the warehouse once at the end with the final answer.

### The words in the project name

- **Triton**: a programming language (by OpenAI) for writing GPU programs. It looks
  like Python. The alternative, **CUDA** (NVIDIA's own GPU language, similar to C++),
  is much harder.
- **Kernel**: a function that runs on the GPU. Thousands of copies run at once, each
  on a different piece of data. "Writing a kernel" means writing one of these.
- **PyTorch**: the most popular Python library for AI. We use it to create data, check
  our answers, and plug our kernel into training.

---

## 2. What you need before you start

### Your computer

| Your computer | Can you run this project locally? |
|---|---|
| **Linux** (Ubuntu etc.) | ✅ Yes |
| **Windows** | ⚠️ Only inside **WSL2** (Windows Subsystem for Linux, a real Linux running inside Windows). Plain Windows: Triton has no official Windows version |
| **Mac** (Intel or Apple Silicon M1/M2/M3) | ❌ **No.** Triton isn't published for macOS at all, not even the CPU emulator. Mac users: skip to Step 9 and do everything on Kaggle |

Be honest with yourself about this. If you're on a Mac, don't spend two hours fighting
installs. Go straight to the free GPU in Step 9.

**Installing WSL2 on Windows** (one time): open PowerShell as Administrator, run
`wsl --install`, restart, open "Ubuntu" from the Start menu. Everything below then
happens inside that Ubuntu window.

### Disk space and internet

- **About 2 GB** free disk space. The virtual environment measured **2.0 GB** with the
  CPU-only PyTorch. The GPU version of PyTorch is several GB more.
- A working internet connection to download ~1 GB of libraries.

### Accounts (free)

- **GitHub** account: github.com. That's where the code lives.
- **Kaggle** account: kaggle.com. That's where you get a free GPU. You must verify a
  phone number to unlock GPUs and internet in notebooks.

---

## Step 1: Open a terminal

A **terminal** (also called command line or shell) is a window where you type commands
instead of clicking. Every grey code block in this file is something you type into it,
then press Enter.

- Linux: press `Ctrl + Alt + T`
- Windows: open the "Ubuntu" app (WSL2)
- Mac: press `Cmd + Space`, type "Terminal" (but read the Mac warning above)

A line starting with `#` inside a code block is a **comment**, a note for you. The
terminal ignores it.

---

## Step 2: Check that Git and Python are installed

**Git** is a tool that saves the history of your code. Each saved snapshot is a
**commit**. **GitHub** is a website that stores Git projects online.

**Python** is the programming language this project uses.

```bash
git --version
python3 --version
```

You should see something like `git version 2.43.0` and `Python 3.12.3`.

- You need **Python 3.10 or newer**. This project was built and tested with **Python 3.13.16**.
- If either command says "command not found", install them (Ubuntu / WSL2):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

`sudo` means "run as administrator". It will ask for your password. Nothing shows while
you type the password. That's normal.

**Tell Git who you are** (one time). Every commit is stamped with this:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 3: Download the code (git clone)

A **repository** (or **repo**) is a project folder tracked by Git. **Cloning** means
downloading a full copy of it, history included.

```bash
cd ~                       # "cd" = change directory. "~" = your home folder
git clone https://github.com/kashyapphassan2000-netizen/WEEK-14--AI-.git
cd WEEK-14--AI-            # move into the project folder
ls                         # "ls" = list the files here
```

You should see: `README.md  NOTES.md  bench  blog  docs  kernels  scripts  tests ...`

**Branches:** a **branch** is a parallel line of history. The work was pushed to a branch
called `claude/busy-curie-eya5dj`. If `ls` shows an almost empty folder, switch to it:

```bash
git branch -a                           # list all branches
git checkout claude/busy-curie-eya5dj   # switch to the branch with the code
```

(Step 11 shows how to make this your `main` branch.)

---

## Step 4: Create a virtual environment

### What it is, and why

A **library** (or **package**) is code someone else wrote that you can reuse, e.g.
PyTorch. **pip** is Python's tool for downloading and installing libraries.

If you install libraries "globally" for your whole computer, different projects
eventually need different versions and break each other. A **virtual environment**
(**venv**) is a private, isolated folder of libraries for *this project only*. If it
breaks, you delete the folder and make a new one. Nothing else is affected.

### Create it

Make sure you're inside the project folder (`cd ~/WEEK-14--AI-`), then:

```bash
python3 -m venv .venv
```

- `python3 -m venv` means "run Python's built-in venv tool".
- `.venv` is the folder name. The leading dot hides it from normal `ls` (use `ls -a` to see it).
- Measured time: **about 2 seconds**.

### Turn it on ("activate")

```bash
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. **That's how you know it's active.**
Every `pip install` now goes into this folder only.

- You must **activate again every time you open a new terminal**. Forgetting this is the
  #1 beginner mistake: "it worked yesterday, today it says module not found".
- To turn it off: `deactivate`
- To check which Python you're using: `which python` should print a path ending in
  `WEEK-14--AI-/.venv/bin/python`.

Upgrade pip inside the venv (old pip versions cause weird install errors):

```bash
python -m pip install --upgrade pip
```

---

## Step 5: Install the libraries

### The libraries and what each one does

| Library | What it's for in this project |
|---|---|
| `torch` (PyTorch) | Creates the test data (tensors), provides the "correct answer" to compare against, and the autograd system our backward pass plugs into |
| `triton` | The GPU language our kernels are written in. Also contains the CPU emulator ("interpreter") |
| `numpy` | Number-crunching library. Triton's CPU emulator uses it under the hood |
| `pytest` | Runs the tests and reports pass/fail |
| `matplotlib` | Draws the speed chart (a PNG image) |

A **tensor** is PyTorch's word for a grid of numbers: a list (1-D), a table (2-D), or
more dimensions. Our attention inputs are 4-D: `[batch, heads, sequence length, head dim]`.

### Option A: computer WITHOUT an NVIDIA GPU (most laptops)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton numpy matplotlib pytest
```

Why the strange first line? Plain `pip install torch` on Linux downloads the **GPU
version** of PyTorch, which is several GB of NVIDIA files you can't use without an
NVIDIA GPU. The `--index-url .../cpu` part gets the small CPU-only version.

Measured on a fast cloud connection: torch **26 s**, the rest **22 s**. On home Wi-Fi
expect 2–15 minutes. Final venv size: **2.0 GB**.

### Option B: Linux computer WITH an NVIDIA GPU

```bash
pip install -r requirements.txt
```

`requirements.txt` is a text file listing the libraries and their **minimum** versions
(`torch>=2.4.0` means "2.4.0 or newer"). `-r` tells pip to read the list from the file.

---

## Step 6: Check the install worked

```bash
python -c "import torch, triton, numpy; print('torch', torch.__version__, '| triton', triton.__version__, '| numpy', numpy.__version__, '| cuda?', torch.cuda.is_available())"
```

`python -c "..."` runs one line of Python directly. Real output from this project's
build machine:

```
torch 2.14.1+cpu | triton 3.8.0 | numpy 2.5.4 | cuda? False
```

- `+cpu` = the CPU-only PyTorch (expected for Option A).
- `cuda? False` = no NVIDIA GPU visible. Expected on a laptop. **CUDA** here just means
  "NVIDIA GPU support".
- Your version numbers may be newer. That's fine.

If you see `ModuleNotFoundError`, your venv isn't active (Step 4) or the install failed (Step 5).

---

## Step 7: Run the tests on your CPU

### What a test is

A **test** is a small program that runs our code on known inputs and checks the answer.
Here, every test does:

1. Create random input numbers.
2. Run **our** kernel.
3. Run **PyTorch's official** attention (`scaled_dot_product_attention`, short **SDPA**)
   on the same numbers, in high precision.
4. Check the two answers match within a small **tolerance** (allowed error, e.g. 0.01).

### How it runs without a GPU

Triton has an **interpreter mode**: set `TRITON_INTERPRET=1` and it runs the same kernel
code on your CPU, faking the GPU with numpy. It's roughly 1000× slower, so the tests
automatically use small inputs on CPU. `tests/conftest.py` decides this for you:

- NVIDIA GPU found → real GPU, full-size tests
- No GPU → interpreter, small tests

### Run them

```bash
python -m pytest tests/ -v
```

`-v` = verbose (one line per test). Real output from a fresh clone:

```
======================== 28 passed, 5 skipped in 8.19s =========================
```

To see **why** things were skipped:

```bash
python -m pytest tests/ -rs
```

```
SKIPPED [2] tests/test_attn_bwd.py:52: needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16
SKIPPED [2] tests/test_attn_fwd.py:47: needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16
SKIPPED [1] tests/test_attn_fwd.py:55: PASS gate size; needs a GPU
```

**What "skipped" means, honestly:** those 5 tests did **not** run. Skipped is not passed.
- 4 are **bf16** tests. **bf16** ("brain float 16") is a 16-bit number format. Triton's
  CPU emulator computes bf16 matrix multiplies wrongly (verified: an error of ~10¹⁰ on
  a 16 × 16 multiply). That's a limitation of the emulator, not of our code, but it means
  bf16 is unverified until you run on an A100 / RTX 30xx / 40xx.
- 1 is the official PASS-gate test at full size (N = 2048). Too big for the emulator.

**What CPU tests do prove:** the math, the loops, the edge handling and the gradients are
correct. **What they don't prove:** speed, real-GPU compilation, memory limits. Only a GPU
run proves those.

Optional quick check of the warm-up kernel:

```bash
TRITON_INTERPRET=1 python -m kernels.matmul
# max abs diff vs torch: 3.1174e-02
```

0.03 looks big, but each output is a sum of 512 products in 16-bit precision, with
values around ±20. That's normal fp16 rounding, well inside tolerance. (Run it as
`python -m kernels.matmul`, not `python kernels/matmul.py`. The second form can't
find the `kernels` package.)

---

## Step 8: Read and understand the code, in this order

**Don't skip this step.** This is the part that makes the project yours.

| Order | File | What to understand before moving on |
|---|---|---|
| 1 | `docs/decisions.md` | The plain-English story and every design decision |
| 2 | `docs/notes.md` | The math: online softmax, why it's exact, the backward formulas |
| 3 | `kernels/matmul.py` | `tl.program_id`, `tl.load`/`tl.store`, masks, `tl.dot`, fp32 accumulator |
| 4 | `kernels/attn_naive.py` | Why writing the N × N matrix to HBM is slow |
| 5 | `kernels/attn_fwd.py` | The running `m`, `l`, `acc` and the rescale by `alpha` |
| 6 | `kernels/attn_bwd.py` | Rebuilding P from the saved LSE; why there are 3 kernels |
| 7 | `tests/*.py` | What exactly is being checked, and at what tolerance |
| 8 | `bench/run.py` | How timing is done |

**A strict self-test.** You understand it when you can do these **without looking**:

1. On paper, derive why `acc = alpha * acc + P @ V` gives the exact softmax result.
2. Explain why `exp2` is used and why the code multiplies by 1.44269504.
3. Explain what breaks if you delete the line `acc = acc * alpha[:, None]`.
   (Try it: the forward tests fail. That experiment was run while building this repo; 9 tests failed.)
4. Explain why the backward pass doesn't store the N × N matrix.
5. Explain why dK/dV and dQ are computed in separate kernels.
6. Explain what a mask does when the sequence length is 100 and the tile size is 64.

If you can't do all six, you're not ready to talk about this project in an interview yet.

---

## Step 9: Get a free GPU (Kaggle)

You need a real NVIDIA GPU for Steps 10–12. **Kaggle** gives about 30 free GPU-hours
per week. A **notebook** is a web page where you run code in boxes called **cells**.

1. Go to **kaggle.com** and sign in. Verify your phone number (Settings → Phone verification).
   Without it, GPU and internet options stay greyed out.
2. Click **Create → New Notebook**.
3. In the right-hand panel, **Session options**:
   - **Accelerator** → `GPU T4 x2` (a **T4** is an older, modest NVIDIA GPU)
   - **Internet** → `On`
4. In the first cell, type and run (Shift + Enter). In a notebook, a leading `!` means
   "run this as a terminal command", and `%cd` changes folder:

```bash
!nvidia-smi
```

`nvidia-smi` shows the GPU. You should see "Tesla T4".

```bash
!git clone https://github.com/kashyapphassan2000-netizen/WEEK-14--AI-.git
%cd WEEK-14--AI-
!git checkout claude/busy-curie-eya5dj
!pip install -q pytest matplotlib
!python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

- Kaggle already has PyTorch and Triton installed. **Don't reinstall torch** there.
  You can break the GPU setup.
- The last line must end with `True`. If it says `False`, the GPU isn't switched on (go back to step 3).
- If `git clone` asks for a password, your repo is **private**. Either make it public
  (GitHub → repo → Settings → General → Danger Zone → Change visibility; the guide
  wants it public anyway) or use a GitHub personal access token.

**Free alternatives:** Google Colab (Runtime → Change runtime type → T4 GPU). Same commands.
**Paid, for the headline numbers:** RunPod or Vast.ai, an A100 for about $1/hour.
Always stop the machine when done: it bills per minute.

**Brutal truth about the T4:** it's old (2018) and slow, with no bf16. Your T4 numbers
will look modest, and the bf16 tests will skip. That's fine for proving correctness.
For impressive numbers you need an A100-class GPU (Ampere generation or newer).

---

## Step 10: Run tests and benchmarks on the GPU

One command does everything:

```bash
!bash scripts/run_all_gpu.sh
```

It runs, in order:

1. **All tests at full size**, including the PASS-gate test at N = 2048.
2. **Forward benchmark**: times naive vs ours vs PyTorch SDPA at sequence lengths 1024, 2048, 4096, 8192.
3. **Forward + backward benchmark**: the speed of a training step.
4. **The plot**: `bench/results/latency.png`.

A **benchmark** is a speed measurement. **Latency** is how long one call takes, in
**ms** (milliseconds, thousandths of a second). Lower is better.

**Expect it to be slow the first time.** Triton **compiles** each kernel (translates it
into GPU machine code) on first use, and **autotune** tries several tile-size settings
and keeps the fastest. The first run can take several minutes. That's normal. This hasn't
been timed on a GPU, so no exact number can be promised.

### How to read the result table

```
| seqlen | naive Triton (ms) | this kernel (ms) | torch SDPA (ms) | speedup vs naive | ours / SDPA | TFLOP/s |
```

- **seqlen**: N, the number of tokens (≈ words).
- **naive Triton**: the slow baseline (builds the N × N table).
- **this kernel**: our FlashAttention.
- **torch SDPA**: PyTorch's professional, hand-tuned version. On A100 it uses
  FlashAttention-**2**; on a T4 it uses a different fused kernel.
- **speedup vs naive** = naive ÷ ours. **Bigger than 1 means we're faster.** The PASS rule is ≥ 1.5 at seqlen 2048.
- **ours / SDPA** = ours ÷ SDPA. **Below 1 means we beat PyTorch** (don't expect that).
  1.3 means we're 30% slower than PyTorch.
- **TFLOP/s**: trillions of math operations per second. A measure of how well we use the GPU.
- `nan` = "not a number". The naive version ran out of GPU memory at that size. Expected at 8192 on small GPUs.

**Honest expectations:**
- Ours should beat naive, and the gap should grow with seqlen. If it doesn't, something is wrong.
- Ours will very probably be **slower than PyTorch SDPA**. PyTorch uses FlashAttention-2,
  written in hand-tuned CUDA by experts. Being within 1.2–2× of it with a from-scratch
  Triton kernel is a reasonable result. Claiming you "beat PyTorch" without evidence would be a lie.

---

## Step 11: Put your real numbers in, commit, push

1. Copy the printed forward table into `README.md` (section **Results**) and
   `blog/post.md` (section 5). Replace the empty rows. Write the GPU name where it says `_TBD_`.
2. Remove the `<!-- -->` around the image line in `README.md` so the plot shows.
3. Download `bench/results/latency.png` and the `results_*.json` files from Kaggle
   (right panel → Output) into your local `bench/results/` folder, or commit from Kaggle.

Then save and upload with Git:

```bash
git status                       # shows which files changed
git add README.md blog/post.md bench/results/
git commit -m "bench: real numbers on <GPU name>"
git push
```

- `git add` = choose which changes go into the next snapshot (called **staging**).
- `git commit -m "..."` = take the snapshot, with a message describing it.
- `git push` = upload your commits to GitHub.

The first `git push` from a new machine asks you to log in. GitHub no longer accepts
your normal password here. Use a **personal access token** (GitHub → Settings →
Developer settings → Personal access tokens) or set up an SSH key.

**Making this branch your `main` branch** (the repo had no `main` before this work):

```bash
git checkout claude/busy-curie-eya5dj
git checkout -b main             # create "main" from the current branch
git push -u origin main
```

Then on GitHub: repo → Settings → General → Default branch → `main`.
(Or open a **Pull Request**, a request to merge one branch into another, on the GitHub website.)

---

## Step 12: The PASS/FAIL gate (be strict)

From the guide's Appendix C. **All** must be true before you call it done:

| # | Criterion | How to check | Current status |
|---|---|---|---|
| 1 | Forward matches PyTorch at N=2048, D=64, fp16, tolerance 1e-2 | `test_pass_gate_forward_n2048` passes on GPU | ❌ Not run (needs GPU) |
| 2 | dQ, dK, dV match PyTorch within 2e-2 | `test_bwd_*` pass on GPU | ⚠️ Pass on CPU emulator; GPU not run |
| 3 | ≥ 1.5× faster than naive at seqlen 2048 | `speedup vs naive` column, row 2048 | ❌ Not measured |
| 4 | All four test files pass | `pytest tests/` on GPU, **0 failed** | ⚠️ CPU only |
| 5 | README has a filled table and a working quickstart | Look at it | ❌ Table empty |
| 6 | Blog published (≥ 1 public link), linked from README | A real URL | ❌ Not done |

**FAIL rules. Do NOT tag if:**
- Any test fails. A "small" failure is still a failure.
- The speedup at 2048 is below 1.5×.
- You haven't published the write-up anywhere public.

**Don't lower the bar to make it pass.** Changing a tolerance from 1e-2 to 1e-1 so a test
goes green is cheating yourself. Find the bug instead (see Troubleshooting).

---

## Step 13: Tag the release v0.1.0

Only after **every** box in Step 12 is ✅:

```bash
python -m pytest tests/ -v                     # on the GPU: must show 0 failed
git tag -a v0.1.0 -m "FlashAttention v1 in Triton - forward+backward, benchmarked vs torch SDPA"
git push origin v0.1.0
```

A **tag** is a permanent name on one specific commit, used to mark a release version.
Then on GitHub: **Releases → Draft a new release → choose tag v0.1.0** and write the notes.

**Write the release notes with YOUR measured numbers.** The guide's suggested message
says "2.5× speedup" and "within 8% of torch SDPA". Those are the guide author's
*expectations*, not your measurements. Copying them without measuring is publishing false claims.

---

## Step 14: Publish the write-up

- `blog/post.md` is the draft. Fill section 5 with your numbers and plot.
- Free places to publish: GitHub itself (the README), dev.to, Medium, Hashnode, or a
  LinkedIn article.
- Link the published post from the README.

---

## How this repo was actually built: the real log, mistakes included

This is exactly what happened, in order, including the mistakes. Nothing was cleaned up.

1. **Read the guide** (`flash-att-triton-build-guide`, 25 pages, a 7-day plan).
2. **Checked the machine:** cloud container, **no GPU** (`nvidia-smi: command not found`),
   Python 3.13.16, empty Git repo.
3. **Created a venv and installed libraries:**
   ```bash
   python3 -m venv /tmp/claude-0/venv
   /tmp/claude-0/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
   /tmp/claude-0/venv/bin/pip install triton numpy pytest matplotlib
   # -> torch 2.14.1+cpu, triton 3.8.0
   ```
4. **Wrote the scaffold:** `.gitignore`, `requirements.txt`, `pyproject.toml`, `LICENSE`, `kernels/_utils.py`.
5. **Wrote all four kernels**, fixing bugs found in the guide's code:
   - The naive kernel couldn't fit in on-chip memory at large N, so it was rewritten to write S to HBM.
   - The causal mask leaked future tokens inside the diagonal tile. Fixed.
   - The backward was "abridged": V used K's pointer, and the dQ kernel was missing. Written in full.
6. **First test run: 17 passed, then 1 failed** (bf16 + strided input test, error ~8×10⁸).
7. **Isolated the failure:** fp16 strided = fine; bf16 = broken. A tiny 16 × 16 bf16
   `tl.dot` in the emulator was wrong by ~2.7×10¹⁰ while a bf16 → fp32 conversion was exact.
   **Conclusion:** an emulator limitation. bf16 tests were moved to GPU-only.
8. **Second run: 1 failed.** This time it was **my own test bug** (wrong matrix shape in
   `test_matmul_non_contiguous`), not a kernel bug. Fixed.
9. **Third run: 28 passed, 5 skipped.**
10. **Checked the tests can fail** (a test that can never fail is worthless). Planted
    two known bugs on purpose:
    - Removed the `sm_scale` multiply on dQ → **7 backward tests failed** ✅
    - Removed the `acc` rescale by `alpha` → **9 forward tests failed** ✅
    Then restored the correct code.
11. **Wrote the benchmark and plot scripts.** Checked that `bench/run.py` exits cleanly
    with a clear message when no GPU is present, and that `bench/plot.py` draws a chart
    (tested with dummy numbers that were **not** committed).
12. **Found that `python kernels/matmul.py` fails** with `ModuleNotFoundError`. The fix
    is to run it as `python -m kernels.matmul`. Documented.
13. **Wrote the docs:** `docs/notes.md`, `docs/decisions.md`, `README.md`, `blog/post.md`.
    During a self-review, three of my own claims were too strong and got corrected
    (what PyTorch uses on a T4, why block pointers were avoided, and a wrong statement
    about sequence parallelism).
14. **Committed in 9 logical commits** (scaffold → math notes → matmul → naive → forward
    → backward → bench → CI → docs) and pushed to branch `claude/busy-curie-eya5dj`.
15. **Reproduced from scratch:** fresh clone + brand-new venv. Measured: venv 2.3 s, torch
    install 26 s, other libraries 22 s, venv size 2.0 GB, tests **28 passed, 5 skipped in 8.19 s**.
16. **Found a GPU gap while writing this file:** a free T4 has no bf16 support, so the
    bf16 tests would likely have failed there. They now skip unless the GPU is Ampere
    or newer (compute capability ≥ 8). Added this NOTES.md.

**Never done:** any run on a real GPU. Steps 9–14 are yours.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | venv not active | `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'kernels'` | Ran a file directly, or from the wrong folder | `cd` to the repo root; use `python -m pytest tests/` or `python -m kernels.matmul` |
| `pip install triton` fails: "no matching distribution" | You're on macOS or plain Windows | Use WSL2 or Kaggle (see section 2) |
| pip downloads gigabytes of `nvidia-*` packages | Installed GPU PyTorch on a CPU machine | Delete `.venv`, redo Steps 4–5 with the `--index-url .../cpu` line |
| `torch.cuda.is_available()` is `False` on Kaggle | GPU accelerator not switched on | Session options → Accelerator → GPU, then restart |
| `bench/run.py needs a CUDA GPU` | You ran the benchmark on CPU | Expected. Benchmarks only work on a GPU |
| `OutOfResources: shared memory` | A tile setting too big for your GPU | Autotune skips those automatically; if a fixed kernel hits it, lower BLOCK_M/BLOCK_N |
| Tests very slow on CPU | Emulator is ~1000× slower | Normal. Should finish in ~10 seconds; if it takes minutes, check that you didn't set a big size |
| Forward wrong, NaN in output | Online softmax bug | Read `docs/notes.md` §4; check `m_i` starts at -inf and `acc` is rescaled by `alpha` |
| Gradients off by ~10% | Missing `sm_scale` on dQ/dK, or wrong `D` | See `kernels/attn_bwd.py`; compare with `docs/notes.md` §7 |
| Correct at small N, wrong at N ≥ 4096 | 32-bit address overflow | Offsets must use `.to(tl.int64)`; already done here |
| `git push` rejects your password | GitHub needs a token now | Use a personal access token or SSH key |
| Messed up the venv | Anything | `deactivate; rm -rf .venv` and redo Steps 4–5. That's the whole point of a venv |

---

## Brutal honesty: what this project is worth for jobs

- **It's a learning project, not a unique invention.** The official Triton tutorial
  ("06-fused-attention") already implements FlashAttention forward and backward, and many
  people have similar repos. This repo follows the same standard structure (and credits it).
- **The guide oversells it.** Lines like "a credential most ML engineers cannot reproduce"
  are marketing. A repo alone gets you nothing. What gets you hired is being able to
  explain the IO math, the online softmax, and the backward pass at a whiteboard, and
  answer "why is yours slower than PyTorch?" without bluffing.
- **The AI wrote this code.** That's fine *as a learning tool* only if you then work
  through Step 8 until you can rebuild the kernels yourself. Interviewers who hire for
  kernel work will ask you to modify the kernel live (add causal masking, change the tile
  shape, explain a race condition). If you can't, the repo hurts you instead of helping.
- **What actually makes it strong:**
  1. Real GPU numbers, honestly reported, including where you lose to PyTorch.
  2. A write-up that shows *you* understand it.
  3. One extension you did yourself, e.g. FlashAttention-2 loop reordering, a dropout
     mask, or variable sequence lengths. That's the part that is genuinely yours.
- **Realistic time to truly own this** as a beginner: the guide says 25–35 hours. If
  Python, PyTorch and matrix math are all new to you, plan for more. Don't skip the
  fundamentals to get to the "impressive" part faster.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Attention** | The transformer operation where every token weighs every other token |
| **Autograd** | PyTorch's system that computes gradients automatically |
| **Autotune** | Trying several kernel settings on the real GPU and keeping the fastest |
| **Backward pass** | Computing gradients: how each input should change to reduce error. Needed for training |
| **Benchmark** | A speed measurement |
| **bf16 / fp16 / fp32** | Number formats: 16-bit "brain float", 16-bit "half", 32-bit "single". Fewer bits = faster, less precise |
| **Branch** | A parallel line of history in Git |
| **Causal** | Each token may only look at itself and earlier tokens (how GPT-style models work) |
| **Clone** | Download a full copy of a Git repo |
| **Commit** | A saved snapshot of your code in Git |
| **Compile** | Translate code into machine instructions the GPU can run |
| **Compute capability** | NVIDIA's GPU generation number. T4 = 7.5, A100 = 8.0, RTX 4090 = 8.9 |
| **CUDA** | NVIDIA's GPU programming platform. "CUDA available" = an NVIDIA GPU is usable |
| **Forward pass** | Computing the output from the inputs |
| **GPU** | Graphics Processing Unit: a chip with thousands of small cores, used for AI |
| **Gradient** | A number saying how much the error changes if an input changes. Used to train models |
| **HBM** | The GPU's main memory. Big but relatively slow |
| **Head / head dim** | One of several parallel attention copies / the size of each token's vector in one head (64 here) |
| **Interpreter mode** | `TRITON_INTERPRET=1`: runs Triton kernels on the CPU by emulation. Slow, but checks correctness |
| **Kernel** | A function that runs on the GPU, in thousands of copies at once |
| **Latency** | Time for one call. Lower is better |
| **Library / package** | Reusable code written by others, installed with pip |
| **LSE (log-sum-exp)** | A one-number-per-row summary of a softmax, saved so the backward pass can rebuild it |
| **Mask** | A true/false grid saying which positions are valid. Used at edges and for causal |
| **NaN** | "Not a number": the result of an invalid calculation, or a benchmark that ran out of memory |
| **Naive attention** | The textbook version that builds the full N × N table |
| **Notebook / cell** | A web page (Kaggle/Colab) where you run code in boxes |
| **Online softmax** | Computing softmax in pieces with a running max and running total. The core trick |
| **pip** | Python's package installer |
| **Pull Request (PR)** | A request on GitHub to merge one branch into another |
| **Push** | Upload your commits to GitHub |
| **PyTorch** | The main Python library for AI / deep learning |
| **Repository (repo)** | A project folder tracked by Git |
| **SDPA** | `scaled_dot_product_attention`, PyTorch's built-in, optimized attention |
| **Sequence length (seqlen, N)** | How many tokens are in the input |
| **Softmax** | Turns a row of scores into probabilities that sum to 1 |
| **SRAM** | Tiny, very fast on-chip GPU memory |
| **Tag** | A permanent name on a commit, marking a release (e.g. v0.1.0) |
| **Tensor** | A grid of numbers with any number of dimensions |
| **Terminal** | The text window where you type commands |
| **Test** | Code that checks other code gives the right answer |
| **TFLOP/s** | Trillions of floating-point operations per second: how hard the GPU is working |
| **Tile / block** | A small rectangular piece of a matrix that fits in fast memory |
| **Token** | A piece of text (roughly a word or part of one) that the model processes |
| **Tolerance** | How much difference between two answers is still counted as "equal" |
| **Transformer** | The neural-network design behind ChatGPT-style models |
| **Triton** | A Python-like language (by OpenAI) for writing GPU kernels |
| **Virtual environment (venv)** | A private, disposable folder of Python libraries for one project |
| **WSL2** | Windows Subsystem for Linux: real Linux running inside Windows |

---

*Reference: Dao, Fu, Ermon, Rudra, Ré. "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness." NeurIPS 2022. arXiv:2205.14135.*
