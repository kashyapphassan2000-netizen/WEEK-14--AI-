# NOTES_IMPLEMENTATION: build the whole project yourself, step by step

This is the **type-along** version of the project. You start with an empty folder and
finish with the complete, tested FlashAttention-in-Triton repo, writing every file
yourself, one piece at a time, with an explanation of every piece.

- **[NOTES.md](NOTES.md)** = the overview, honest status, Kaggle GPU guide, glossary.
- **This file** = the exact keystrokes and the code, from zero to the end.
- **[docs/decisions.md](docs/decisions.md)** = *why* each design choice was made.

---

## Read this first (honest, no sugarcoating)

1. **This document was tested by following it literally.** A script took every command
   and every code block in this file, ran them in order in an empty folder on a CPU-only
   Linux machine, and checked two things: the files you end up with are identical to the
   real repo (blank lines aside), and all the tests pass. So if you follow it exactly and
   something breaks, the difference is on your side. Re-read the step.
2. **What was NOT tested:** the GPU parts (Phase 9 onwards). No GPU was available.
   Those steps are written carefully, but they have never been executed.
3. **It's a lot.** About 1,200 lines of code. Type or paste one part at a time, run the
   check after each file, and commit at the end of each phase. Don't paste everything at
   once and hope. When it breaks you won't know where.
4. **Pasting is not learning.** You can finish this in an afternoon by pasting. You will
   understand it only if you read every explanation and can answer the "Check yourself"
   questions without looking. If you'll ever talk about this in an interview, do that.
5. **You need Linux or WSL2 (Windows).** Triton does not install on macOS, even for the
   CPU emulator. Mac users: do the whole thing in a Kaggle notebook instead (see NOTES.md, Step 9).

### How to read the symbols

| You see | Meaning |
|---|---|
| A grey block under **▶ Run** | Type (or paste) these commands into the terminal, press Enter |
| 📄 **`path/to/file.py`** | Put this code into that file, using the editor |
| **✅ Expected** | What you should see. Numbers like times and commit IDs will differ |
| `# text` inside a command block | A comment for you. The terminal ignores it |

### What each phase builds

| Phase | Guide day | What you build | Files |
|---|---|---|---|
| 0 | n/a | Tools: terminal, git, Python, an editor | n/a |
| 1 | Day 0 | Project folder, git, virtual environment, libraries, config files | `.gitignore`, `requirements.txt`, `pyproject.toml`, `LICENSE` |
| 2 | Day 1 | The math notes | `docs/notes.md` |
| 3 | Day 2 | Your first Triton kernels: vector add, then matrix multiply | `kernels/_utils.py`, `kernels/matmul.py`, test setup |
| 4 | Day 3 | Naive attention (the slow baseline) | `kernels/attn_naive.py` |
| 5 | Day 4 | FlashAttention forward (the main kernel) | `kernels/attn_fwd.py` |
| 6 | Day 5 | FlashAttention backward (training) | `kernels/attn_bwd.py` |
| 7 | Day 6 | Benchmark + plot scripts | `bench/run.py`, `bench/plot.py`, `scripts/run_all_gpu.sh` |
| 8 | n/a | Automatic tests on GitHub (CI) | `.github/workflows/tests.yml` |
| 9 | Day 6 | Run on a real GPU, get numbers | (no new code) |
| 10 | Day 7 | README, blog, pass/fail gate, release tag | `README.md`, `blog/post.md` |

---

# Phase 0: Tools

You need four things: a **terminal** (the text window where you type commands), **git**
(saves the history of your code), **Python** (the language), and an **editor** (an app
for editing text files).

### 0.1 Open a terminal

- **Linux:** `Ctrl + Alt + T`
- **Windows:** install WSL2 once. Open PowerShell **as Administrator**, run `wsl --install`,
  restart the PC, then open the **Ubuntu** app from the Start menu. Do *everything* below
  inside that Ubuntu window.

### 0.2 Install git and Python

▶ Run (Ubuntu / WSL2):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

`sudo` = run as administrator. It asks for your password. The password doesn't show while
you type. That's normal, just type it and press Enter.

▶ Run, to check:

```bash
git --version
python3 --version
```

✅ Expected (your numbers may be higher, which is fine):

```text
git version 2.43.0
Python 3.12.3
```

You need **Python 3.10 or newer**. This document was verified with Python 3.13.16.

### 0.3 Tell git your name (once per computer)

Every commit you make gets stamped with this.

▶ Run (use your own name and email):

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

### 0.4 Install an editor: VS Code

Download **Visual Studio Code** from code.visualstudio.com. It's free.
- On **WSL2**: install VS Code on Windows, then inside Ubuntu, in your project folder, type
  `code .` and it opens connected to Linux. (Install the "WSL" extension if it asks.)
- Useful extensions: **Python** (by Microsoft). That's enough.

**How to create a file in VS Code:** in the left sidebar (Explorer), right-click the folder →
**New File** → type the name, e.g. `kernels/matmul.py` → paste → **Ctrl + S** to save.

**Python indentation is part of the code.** Python uses spaces at the start of a line to
know which lines belong together. Paste code exactly as shown and don't "tidy" the spaces.
Use 4 spaces, never tabs (VS Code does this by default for `.py` files).

---

# Phase 1: Create the project (guide Day 0)

### 1.1 Make the folder and start git

A **repository** (repo) is a folder whose history git tracks.

<!-- run -->
▶ Run:

```bash
cd ~
mkdir flash-att-triton
cd flash-att-triton
git init
git branch -M main
```

- `cd ~` = go to your home folder.
- `mkdir` = make directory (folder). `cd` = change directory (go into it).
- `git init` = turn this folder into a git repository. It creates a hidden `.git` folder.
- `git branch -M main` = name the main line of history `main`.

✅ Expected: `Initialized empty Git repository in /home/<you>/flash-att-triton/.git/`

> **From now on, every command runs inside `flash-att-triton`.** If you open a new
> terminal, first type `cd ~/flash-att-triton`.

### 1.2 Create the sub-folders

<!-- run -->
▶ Run:

```bash
mkdir -p docs kernels tests bench/results blog scripts .github/workflows playground
touch kernels/__init__.py bench/__init__.py bench/results/.gitkeep
echo "playground/" >> .git/info/exclude
```

- `mkdir -p` makes several folders at once (`-p` = don't complain if they already exist).
- `touch` creates empty files.
  - `kernels/__init__.py`: an empty file named `__init__.py` tells Python "this folder is a
    **package**", so you can write `from kernels.matmul import matmul`. It stays empty until Phase 6.
  - `bench/results/.gitkeep`: git doesn't save empty folders. This empty file keeps the folder.
- `playground/` is your scratch area for experiments. The `echo ... >> .git/info/exclude`
  line tells git to ignore it on *your* machine only, so experiments never get committed.

| Folder | What goes there |
|---|---|
| `kernels/` | The GPU code (the actual project) |
| `tests/` | Code that checks the kernels give correct answers |
| `bench/` | Speed measurement scripts; `bench/results/` holds their output |
| `docs/` | Notes and explanations |
| `blog/` | The write-up |
| `scripts/` | Helper shell scripts |
| `.github/workflows/` | Instructions for GitHub to run your tests automatically |
| `playground/` | Your private experiments (ignored by git) |

### 1.3 Create the virtual environment

A **library** is code other people wrote that you reuse (PyTorch, Triton…). **pip** installs
libraries. A **virtual environment (venv)** is a private folder of libraries for this
project only, so this project can't break other projects and vice versa. If it ever gets
messed up, you delete the folder and recreate it.

<!-- run -->
▶ Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

- `python3 -m venv .venv` creates the venv in a folder named `.venv` (takes ~2 seconds).
- `source .venv/bin/activate` **turns it on**. Your prompt now starts with `(.venv)`.
- The last line upgrades pip itself (old pip causes odd install errors).

> ⚠️ **#1 beginner mistake:** every new terminal needs `source .venv/bin/activate` again.
> If you see `ModuleNotFoundError: No module named 'torch'`, you forgot this.

### 1.4 Install the libraries

| Library | Why we need it |
|---|---|
| `torch` (PyTorch) | Makes the test data, gives the "correct answer" to compare against, and runs training (autograd) |
| `triton` | The GPU language our kernels are written in. Also contains the CPU emulator |
| `numpy` | Math library. Triton's CPU emulator uses it internally |
| `matplotlib` | Draws the speed chart |
| `pytest` | Runs our tests and reports pass/fail |

**If your computer has NO NVIDIA GPU** (most laptops):

<!-- run -->
▶ Run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton numpy matplotlib pytest
```

The `--index-url .../cpu` part matters. Plain `pip install torch` on Linux downloads the
GPU version, several GB of NVIDIA files you can't use. Measured: about 50 seconds on a
fast connection; on home Wi-Fi expect 2–15 minutes. The venv ends up about **2 GB**.

**If your Linux computer HAS an NVIDIA GPU:** run `pip install torch triton numpy matplotlib pytest` instead.

<!-- run -->
▶ Run, to check:

```bash
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| GPU?', torch.cuda.is_available())"
```

✅ Expected (versions may be newer):

```text
torch 2.14.1+cpu | triton 3.8.0 | GPU? False
```

`GPU? False` is correct on a laptop. **CUDA** is NVIDIA's GPU system;
`torch.cuda.is_available()` asks "is there a usable NVIDIA GPU?".

### 1.5 The config files

Create these four files in the editor.

**`.gitignore`** lists files git should never save: caches, the venv, images (except the benchmark plot).

<!-- impl-file: .gitignore part: 1/1 -->
📄 **`.gitignore`** · create this file and paste **exactly** this into it:

```text
__pycache__/
*.pyc
.venv/
venv/
.env
*.log
.cache/
.pytest_cache/
.DS_Store
.vscode/settings.json
*.png
!bench/results/*.png
```

- `__pycache__/`, `*.pyc`: files Python generates automatically. Never commit them.
- `.venv/`: your venv is 2 GB and machine-specific. Never commit it.
- `*.png` then `!bench/results/*.png`: ignore all images *except* (`!`) the benchmark plot.

**`requirements.txt`** lists the libraries with **minimum** versions (`>=`), so anyone can
install them with `pip install -r requirements.txt`.

<!-- impl-file: requirements.txt part: 1/1 -->
📄 **`requirements.txt`** · create this file and paste **exactly** this into it:

```text
torch>=2.4.0
triton>=3.0.0
numpy>=1.26
matplotlib>=3.8
pytest>=8.0
```

**`pyproject.toml`** describes the project (name, version) and configures pytest.

<!-- impl-file: pyproject.toml part: 1/1 -->
📄 **`pyproject.toml`** · create this file and paste **exactly** this into it:

```toml
[project]
name = "flash-att-triton"
version = "0.1.0"
description = "FlashAttention v1 (forward + backward) implemented in Triton"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
dependencies = ["torch>=2.4.0", "triton>=3.0.0", "numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "matplotlib>=3.8"]

[tool.setuptools]
packages = ["kernels"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

The important part is the last section:
- `testpaths = ["tests"]`: pytest looks for tests in the `tests/` folder.
- `pythonpath = ["."]`: pytest can import `kernels` from the project root. Without it,
  tests fail with `No module named 'kernels'`.

**`LICENSE`**: the MIT license says anyone may use your code. **Put your own name** on the
"Copyright" line (the verified copy below has the original author's name).

<!-- impl-file: LICENSE part: 1/1 -->
📄 **`LICENSE`** · create this file and paste **exactly** this into it:

```text
MIT License

Copyright (c) 2026 kashyapphassan2000-netizen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 1.6 First commit

A **commit** is a saved snapshot. `git add` picks which files go into it ("staging");
`git commit` takes the snapshot with a message.

<!-- run -->
▶ Run:

```bash
git add .gitignore requirements.txt pyproject.toml LICENSE kernels/__init__.py bench/__init__.py bench/results/.gitkeep
git commit -m "chore: project scaffold, requirements, MIT license"
git status
```

✅ Expected: a line like `[main (root-commit) 1a2b3c4] chore: project scaffold, ...`, then
`git status` says `nothing to commit, working tree clean` (or lists only empty folders).

> Always `git add` **named files**, not `git add .`. That way you never commit something by accident.

### 1.7 Put it on GitHub (once)

1. On github.com: **+** (top right) → **New repository** → name `flash-att-triton` →
   **Public** → do **not** tick "Add a README" → **Create repository**.
2. ▶ Run (replace `<you>` with your GitHub username):

```bash
git remote add origin https://github.com/<you>/flash-att-triton.git
git push -u origin main
```

- `remote add origin` = remember the GitHub address under the nickname `origin`.
- `push` = upload your commits. `-u` = remember this, so next time plain `git push` works.
- When it asks for a password, GitHub needs a **personal access token**, not your account
  password: github.com → Settings → Developer settings → Personal access tokens → Tokens
  (classic) → Generate, tick `repo`. Paste the token as the password.

---

# Phase 2: The math notes (guide Day 1)

Before any kernel code, write the math in **your own words**. The guide is strict about this:
if you can't derive it on paper, you can't debug it in a kernel.

<!-- impl-starter: docs/notes.md -->
📄 **`docs/notes.md`** · create it with this skeleton, then fill every section yourself:

```markdown
# Notes: the IO math behind FlashAttention v1

## 1. Standard attention
(S = QK^T * scale, P = softmax(S), O = PV. Why the N x N matrix is a problem in HBM.)

## 2. Why memory, not math, is the bottleneck
(HBM is big and slow, SRAM is tiny and fast. Numbers for your GPU.)

## 3. Softmax with a running max
(m = max, l = sum of exp(s - m), p = exp(s - m) / l)

## 4. The online-softmax recurrence
(Two blocks -> merge with exp(m1 - m) and exp(m2 - m). Write the loop.)

## 5. IO complexity
(Standard: O(N^2 + Nd). Flash: O(N^2 d^2 / M).)

## 6. The exp2 trick

## 7. Backward by recomputation
(D = rowsum(dO * O); dV, dP, dS, dQ, dK formulas.)
```

The complete reference version is `docs/notes.md` in the original repo. **Write yours first,
then compare.** That comparison is where you learn.

The key idea in one picture. For each query row, walking over key blocks:

```text
start:     m = -inf, l = 0, acc = 0
per block: m_new = max(m, max(scores))
           p     = exp(scores - m_new)
           alpha = exp(m - m_new)          <- shrinks the old totals to the new max
           l     = alpha * l + sum(p)
           acc   = alpha * acc + p @ V
           m     = m_new
end:       output = acc / l                <- exactly softmax(scores) @ V
```

<!-- run -->
▶ Run:

```bash
git add docs/notes.md
git commit -m "docs: hand-derived IO math and online softmax recurrence"
```

---

# Phase 3: Your first Triton kernels (guide Day 2)

## 3.1 Five ideas you need before writing any kernel

**1. A kernel is run by many "programs" at once.** You write the code for *one* program.
Triton launches many copies. Each copy asks `tl.program_id(0)` ("which number am I?") and
uses the answer to pick its own piece of the data. The list of how many programs to launch
is the **grid**.

**2. Tensors live in memory as one long line.** A **tensor** is a grid of numbers
(PyTorch's name for it). A 4-D tensor of shape `[2, 3, 4, 5]` is really 120 numbers in a
row. The **stride** of each dimension says how many positions to jump to move one step
along that dimension.

<!-- run -->
▶ Run (to see strides for yourself):

```bash
python -c "
import torch
t = torch.zeros(2, 3, 4, 5)
print(t.shape, t.stride())
u = torch.zeros(2, 4, 3, 5).transpose(1, 2)
print(u.shape, u.stride(), 'contiguous:', u.is_contiguous())
"
```

✅ Expected:

```text
torch.Size([2, 3, 4, 5]) (60, 20, 5, 1)
torch.Size([2, 3, 4, 5]) (60, 5, 15, 1) contiguous: False
```

So element `[b, h, n, d]` of `t` sits at position `b*60 + h*20 + n*5 + d*1`.
`u` has the same shape but different strides: it's a **view** (same memory, read in a
different order). This is why every kernel takes stride arguments instead of assuming
a layout. Then it works for both.

**3. Pointer = address.** In a kernel, `x_ptr` is the memory address of the first number.
`x_ptr + 7` is the address of the 8th. `tl.load(address)` reads, `tl.store(address, value)` writes.

**4. Masks protect the edges.** If you have 10 numbers and each program handles 4, the
third program covers positions 8, 9, 10, 11, but 10 and 11 don't exist. A **mask** is a
list of True/False saying which positions are real. `tl.load(..., mask=mask, other=0.0)`
reads only the real ones and uses `0.0` for the rest.

**5. Tile sizes are fixed at compile time.** `BLOCK: tl.constexpr` means the value is
baked in when Triton **compiles** the kernel (translates it into GPU instructions).
`tl.arange(0, BLOCK)` needs a fixed size, and it must be a power of 2 (16, 32, 64…).

## 3.2 Warm-up: vector add (in the playground, not committed)

<!-- impl-starter: playground/hello_add.py -->
📄 **`playground/hello_add.py`** · create it with:

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                     # which program am I? 0, 1, 2, ...
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # the element positions I own
    mask = offs < n                            # True only for positions that exist
    x = tl.load(x_ptr + offs, mask=mask)       # read my slice of x
    y = tl.load(y_ptr + offs, mask=mask)       # read my slice of y
    tl.store(out_ptr + offs, x + y, mask=mask) # write my slice of the answer


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4
    grid = (triton.cdiv(n, BLOCK),)            # how many programs to launch
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


x = torch.arange(10, dtype=torch.float32)
y = torch.ones(10)
print("x      :", x.tolist())
print("y      :", y.tolist())
print("triton :", add(x, y).tolist())
print("torch  :", (x + y).tolist())
print("programs launched:", triton.cdiv(10, 4))
```

Line by line:
- `@triton.jit`: this line above a function is a **decorator**. It tells Triton "compile
  this function for the GPU". Inside it you can only use Triton operations (`tl.*`),
  not normal PyTorch.
- `pid = tl.program_id(0)`: program 0 handles positions 0–3, program 1 handles 4–7,
  program 2 handles 8–11.
- `offs = pid * BLOCK + tl.arange(0, BLOCK)`: for program 2 that's `[8, 9, 10, 11]`.
- `mask = offs < n`: for program 2 that's `[True, True, False, False]`.
- `triton.cdiv(10, 4)` = "divide and round **up**" = 3 programs.
- `add_kernel[grid](...)`: the square brackets are the launch. "Run `add_kernel` with this grid."

<!-- run -->
▶ Run (`TRITON_INTERPRET=1` = run on the CPU emulator, since you have no GPU):

```bash
TRITON_INTERPRET=1 python playground/hello_add.py
```

✅ Expected:

```text
x      : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y      : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
triton : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
torch  : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
programs launched: 3
```

**Experiment (do it!):** delete `, mask=mask` from the `tl.store` line and run again. On a
real GPU this writes past the end of the array (a memory bug). Then put it back. Try
`BLOCK = 8` and see the program count change.

## 3.3 A shared helper: `kernels/_utils.py`

On a GPU, `@triton.autotune` tries several tile sizes and keeps the fastest. The CPU
emulator can't time anything, so this helper gives it just one setting. (With a single
setting, autotune doesn't try to time anything.)

<!-- impl-file: kernels/_utils.py part: 1/1 -->
📄 **`kernels/_utils.py`** · create this file and paste **exactly** this into it:

```python
"""Small helpers shared by every kernel file."""
import os

import triton


def is_interpreter() -> bool:
    """True when Triton runs kernels in its CPU interpreter (TRITON_INTERPRET=1)."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def autotune_configs(gpu_configs, interpreter_config):
    """Pick the autotune search space.

    On a GPU we let `triton.autotune` try every config. The interpreter has no
    GPU timers, so we hand it exactly one config: with a single config the
    autotuner skips benchmarking entirely.
    """
    if is_interpreter():
        return [interpreter_config]
    return gpu_configs


def cfg(num_warps=4, num_stages=3, **meta) -> triton.Config:
    return triton.Config(meta, num_warps=num_warps, num_stages=num_stages)
```

- `os.environ.get("TRITON_INTERPRET", "0") == "1"` reads the environment variable you set
  with `TRITON_INTERPRET=1 ...`.
- `cfg(...)` is shorthand: `cfg(BLOCK_M=64, num_warps=4)` makes a `triton.Config`.
  **Warps** are groups of 32 GPU threads. More warps = more threads per program.
  **Stages** = how many loads the GPU prefetches ahead (pipelining).

## 3.4 Matrix multiply: `kernels/matmul.py`

C = A × B. Each program computes one rectangular tile of C, `BLOCK_M` rows by `BLOCK_N` columns.
To do that it walks along the shared dimension K, `BLOCK_K` at a time, multiplying a tile of A
by a tile of B and adding into an accumulator.

```text
        K                     N                      N
   ┌─────────┐          ┌─────────┐           ┌─────────┐
 M │ A  ▓▓▓▓ │  ×     K │ B  ▓    │  =      M │ C  ▓    │   one program = one ▓ tile of C
   │         │          │    ▓    │           │         │   it walks ▓▓▓▓ along A's row
   └─────────┘          │    ▓    │           └─────────┘   and down B's column
                        └─────────┘
```

<!-- impl-file: kernels/matmul.py part: 1/7 -->
📄 **`kernels/matmul.py`** · part 1 of 7. Create this file and paste this in:

```python
"""Day 2: tiled fp16 matmul in Triton.

The "hello world" kernel. It exercises everything the attention kernels need:
the program-id grid, pointer arithmetic with strides, masked `tl.load` /
`tl.store`, and `tl.dot` with an fp32 accumulator.
"""
import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg
```

The top: a **docstring** (the text in triple quotes, documentation for humans), then the
**imports** (load the libraries).
`import triton.language as tl` gives Triton's kernel operations the short name `tl`.

<!-- impl-file: kernels/matmul.py part: 2/7 -->
📄 **`kernels/matmul.py`** · part 2 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.autotune(
    configs=autotune_configs(
        [
            cfg(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=3),
            cfg(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
            cfg(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
            cfg(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=4),
        ],
        cfg(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=4),
    ),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
```

- `@triton.autotune(configs=..., key=[...])`: on a GPU, try these 4 tile-size combos the
  first time it runs, keep the fastest, and redo it whenever `M`, `N` or `K` (the `key`) change.
  The 5th argument to `autotune_configs` is the single CPU-emulator setting.
- The arguments: `a_ptr, b_ptr, c_ptr` are memory addresses; `M, N, K` are sizes;
  `stride_am, stride_ak` are A's strides along rows and columns (same for B and C).
- `BLOCK_M: tl.constexpr`: compile-time tile sizes, filled in by autotune.

<!-- impl-file: kernels/matmul.py part: 3/7 -->
📄 **`kernels/matmul.py`** · part 3 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    """Each program computes one [BLOCK_M, BLOCK_N] tile of C = A @ B."""
    # Group-ordered program ids: neighbouring programs share rows of A,
    # which improves L2 cache reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

This block decides *which* tile of C this program owns. The simple way is: program 0 =
top-left tile, program 1 = next tile to the right, and so on. The **grouped** order used here
instead walks `GROUP_M` tile-rows in a zig-zag, so programs running at the same time reuse
the same rows of A from the GPU's cache. **It's only a speed trick.** If it confuses you,
the result is just: `pid_m` = which tile-row, `pid_n` = which tile-column.

<!-- impl-file: kernels/matmul.py part: 4/7 -->
📄 **`kernels/matmul.py`** · part 4 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    # Wrap-around (% M, % N) keeps loads in bounds; the store mask discards them.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
```

The core loop. The key line is the **broadcasting** trick for building a 2-D grid of addresses:

```text
offs_am[:, None]   turns [0, 1] into a column    [[0],
                                                  [1]]
offs_k[None, :]    turns [0, 1, 2] into a row     [[0, 1, 2]]
adding them gives every (row, col) combination:   [[0+0, 0+1, 0+2],
                                                   [1+0, 1+1, 1+2]]
```

Multiply each by its stride and you have the address of every element of the tile.

- `% M` and `% N` (**modulo**) wrap out-of-range rows back to valid ones, so the loads never
  go out of bounds. Those extra rows are thrown away at the store.
- `acc = tl.zeros(..., dtype=tl.float32)`: the running sum is **fp32** (32-bit, precise),
  even though the inputs are **fp16** (16-bit, fast). Adding thousands of fp16 numbers
  loses precision.
- `mask=offs_k[None, :] < k_remaining`: the last K-tile may hang off the edge.
- `acc = tl.dot(a, b, acc)`: multiply the two tiles on the GPU's matrix hardware (**tensor
  cores**) and add into `acc`. `tl.dot` needs every tile side to be at least 16.
- `a_ptrs += BLOCK_K * stride_ak`: slide A's tile to the right, B's tile down.

<!-- impl-file: kernels/matmul.py part: 5/7 -->
📄 **`kernels/matmul.py`** · part 5 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

Convert the fp32 sum back to the output type (fp16) and store. Here the mask is
essential: it stops us writing the wrapped-around garbage rows.

<!-- impl-file: kernels/matmul.py part: 6/7 -->
📄 **`kernels/matmul.py`** · part 6 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B for 2-D fp16/bf16 tensors. Accumulates in fp32."""
    assert a.ndim == 2 and b.ndim == 2, "matmul expects 2-D tensors"
    assert a.shape[1] == b.shape[0], "K mismatch"
    assert a.dtype == b.dtype, "dtype mismatch"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

This is ordinary Python that runs on the CPU and **launches** the kernel:
- `assert` checks the inputs. If one fails, the program stops with that message.
- `torch.empty(...)` allocates the output without filling it (the kernel fills it).
- `grid = lambda META: (...)`: the grid depends on the tile size autotune picks, so it's a
  small function. `META` holds the chosen `BLOCK_M`, `BLOCK_N`. A **lambda** is a one-line function.
- `a.stride(0), a.stride(1)`: pass the real strides, so transposed inputs work too.

<!-- impl-file: kernels/matmul.py part: 7/7 -->
📄 **`kernels/matmul.py`** · part 7 of 7. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
if __name__ == "__main__":  # python -m kernels.matmul
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(512, 512, device=dev, dtype=torch.float16)
    b = torch.randn(512, 512, device=dev, dtype=torch.float16)
    diff = (matmul(a, b).float() - a.float() @ b.float()).abs().max().item()
    print(f"max abs diff vs torch: {diff:.4e}")
```

`if __name__ == "__main__":` means "only run this when the file is run directly, not when
it's imported".

<!-- run -->
▶ Run (note: `-m kernels.matmul`, not `kernels/matmul.py`):

```bash
TRITON_INTERPRET=1 python -m kernels.matmul
```

✅ Expected:

```text
max abs diff vs torch: 3.1174e-02
```

0.03 is fine. Each output adds 512 products in fp16, with values around ±20, so small
rounding differences are expected. (After Phase 6, this command also prints a
`RuntimeWarning: 'kernels.matmul' found in sys.modules ...` line. That's harmless: it
appears because `kernels/__init__.py` then imports `matmul` too.) (Running it as `python kernels/matmul.py` fails with
`No module named 'kernels'`, because Python then can't see the project root.)

## 3.5 The test setup

A **test** runs your code on known inputs and checks the answer. **pytest** finds every
function whose name starts with `test_` and runs it.

**`tests/conftest.py`** is a special file pytest loads first, automatically. It decides
GPU vs CPU-emulator **before** any kernel is imported:

<!-- impl-file: tests/conftest.py part: 1/1 -->
📄 **`tests/conftest.py`** · create this file and paste **exactly** this into it:

```python
"""Pick where the kernels run before any kernel module is imported.

- CUDA GPU present  -> real kernels, the guide's full test sizes.
- No GPU            -> Triton's CPU interpreter (TRITON_INTERPRET=1), small sizes.
  Same kernel code, same math, ~1000x slower, so shapes are shrunk.
"""
import os

import torch

ON_GPU = torch.cuda.is_available()
if not ON_GPU:
    os.environ.setdefault("TRITON_INTERPRET", "1")

DEVICE = "cuda" if ON_GPU else "cpu"

# bf16 tensor cores exist from Ampere (compute capability 8.x: A100, RTX 30xx/40xx) on.
# A T4 (7.5) or V100 (7.0) has none, so bf16 tests only run on 8.x+.
BF16_OK = ON_GPU and torch.cuda.get_device_capability()[0] >= 8
```

- `os.environ.setdefault("TRITON_INTERPRET", "1")`: switch on the emulator if there's no GPU.
  This must happen before Triton reads the kernels, which is why it lives here.
- `BF16_OK`: **bf16** is a 16-bit number format that only newer GPUs (A100, RTX 30xx/40xx)
  support in hardware. A GPU's **compute capability** is its generation number (T4 = 7.5,
  A100 = 8.0). The CPU emulator also gets bf16 matrix multiplies wrong, so bf16 tests only
  run on 8.x+ GPUs.

**`tests/helpers.py`**: small shared functions for the tests.

<!-- impl-file: tests/helpers.py part: 1/1 -->
📄 **`tests/helpers.py`** · create this file and paste **exactly** this into it:

```python
import torch
import torch.nn.functional as F

from conftest import BF16_OK, DEVICE, ON_GPU  # noqa: F401


def rand_qkv(B, H, N, D, dtype=torch.float16, requires_grad=False):
    torch.manual_seed(0)
    return [
        torch.randn(B, H, N, D, device=DEVICE, dtype=dtype, requires_grad=requires_grad)
        for _ in range(3)
    ]


def reference_attention(q, k, v, causal=False):
    """fp32 SDPA reference, cast back to the input dtype."""
    o = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    return o.to(q.dtype)
```

- `torch.manual_seed(0)`: makes the "random" numbers identical every run. A test that
  fails randomly is useless.
- `rand_qkv`: makes the three attention inputs Q, K, V, shaped `[B, H, N, D]` =
  [batch, heads, sequence length, head dim].
- `reference_attention`: PyTorch's official attention (`scaled_dot_product_attention`,
  **SDPA**), computed in **fp32** (`.float()`) so the reference is more precise than what we test.

**`tests/test_matmul.py`**:

<!-- impl-file: tests/test_matmul.py part: 1/1 -->
📄 **`tests/test_matmul.py`** · create this file and paste **exactly** this into it:

```python
import pytest
import torch

from helpers import DEVICE, ON_GPU
from kernels.matmul import matmul

SHAPES = (
    [(512, 512, 512), (1024, 1024, 1024), (768, 1024, 512)]
    if ON_GPU
    else [(64, 64, 64), (100, 72, 48), (37, 129, 65)]
)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matmul_correctness(M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(matmul(a, b), ref, rtol=1e-2, atol=1e-2)


def test_matmul_non_contiguous():
    torch.manual_seed(0)
    a = torch.randn((64, 96), device=DEVICE, dtype=torch.float16)
    b = torch.randn((80, 96), device=DEVICE, dtype=torch.float16).t()  # strided view
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(matmul(a, b), ref, rtol=1e-2, atol=1e-2)
```

- `@pytest.mark.parametrize("M,N,K", SHAPES)`: run the same test once per shape.
- Shapes like `(37, 129, 65)` are deliberately awkward. They aren't multiples of the tile
  size, so they exercise the masks.
- `torch.testing.assert_close(got, want, rtol, atol)` passes if, for every element,
  `|got - want| <= atol + rtol * |want|`. Here that's an error of about 1%.
- `test_matmul_non_contiguous`: `.t()` (transpose) makes a strided view. This proves the
  stride arguments are used correctly.

<!-- run -->
▶ Run:

```bash
python -m pytest tests/test_matmul.py -v
```

✅ Expected (on CPU; on a GPU the shapes are bigger):

```text
tests/test_matmul.py::test_matmul_correctness[64-64-64] PASSED
tests/test_matmul.py::test_matmul_correctness[100-72-48] PASSED
tests/test_matmul.py::test_matmul_correctness[37-129-65] PASSED
tests/test_matmul.py::test_matmul_non_contiguous PASSED
============================== 4 passed in 1.07s ===============================
```

<!-- run -->
▶ Run (commit Phase 3):

```bash
git add kernels/_utils.py kernels/matmul.py tests/conftest.py tests/helpers.py tests/test_matmul.py
git commit -m "feat(matmul): tiled fp16 matmul kernel with autotune, passes vs torch"
```

**Check yourself:** (1) What does `tl.program_id(0)` return? (2) Why is `acc` fp32?
(3) What breaks without the store mask? (4) Why pass strides instead of assuming a layout?

---

# Phase 4: Naive attention, the slow baseline (guide Day 3)

Attention, the textbook way, in three kernels:

```text
kernel 1:  S = Q @ K^T * scale   ── write N x N to HBM (GPU main memory)
kernel 2:  P = softmax(S)        ── read N x N, write N x N
kernel 3:  O = P @ V             ── read N x N again
```

That's three trips through slow memory with a huge table. It's the cost FlashAttention
removes. We need this version to (a) check correctness against and (b) measure the speedup against.

> The guide's own version loaded all of K and V into on-chip memory at once. At N = 4096
> that's 512 KB per tensor, and on-chip memory is ~100–200 KB, so it can't compile at the
> sizes we benchmark. That's why this version writes S to HBM instead. It's also what
> "standard attention" means in the paper.

<!-- impl-file: kernels/attn_naive.py part: 1/5 -->
📄 **`kernels/attn_naive.py`** · part 1 of 5. Create this file and paste this in:

```python
"""Day 3: naive ("standard") attention in Triton. This is the baseline.

It does exactly what the textbook formula says and what eager PyTorch does:

    S = Q @ K^T * scale      -> written to HBM   (N x N per head)
    P = softmax(S)           -> read + written   (N x N per head)
    O = P @ V                -> read again

Three kernels, three round trips through HBM for the N x N matrix. That
O(N^2) memory traffic is the cost FlashAttention removes, so this is the
baseline the "speedup vs naive" number is measured against.

Note: the guide's single-kernel version keeps all of K and V for one head in
on-chip memory. At N=4096, D=64 that is 512 KB per tensor, which does not fit
in shared memory, so it fails to compile at the sizes we benchmark. Writing S
to HBM is the honest "standard attention" baseline from the paper.
"""
import torch
import triton
import triton.language as tl

from kernels._utils import is_interpreter
```

Docstring and imports. `is_interpreter` comes from your `_utils.py`.

<!-- impl-file: kernels/attn_naive.py part: 2/5 -->
📄 **`kernels/attn_naive.py`** · part 2 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.jit
def _qk_kernel(
    Q, K, S, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """S[b, h, m, n] = scale * sum_d Q[b, h, m, d] * K[b, h, n, d] for one tile."""
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    s_base = S + off_hz.to(tl.int64) * N_CTX * N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX
    mask_n = offs_n < N_CTX

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=mask_m[:, None], other=0.0)
    # Load K already transposed: [HEAD_DIM, BLOCK_N].
    kT = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                 mask=mask_n[None, :], other=0.0)
    s = tl.dot(q, kT) * sm_scale

    s_ptrs = s_base + offs_m[:, None] * N_CTX + offs_n[None, :]
    tl.store(s_ptrs, s.to(S.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
```

**Kernel 1, scores.** The grid is 3-D: `(row tiles, column tiles, batch × heads)`, so each
program fills one `BLOCK_M × BLOCK_N` tile of S for one head.
- `off_z = off_hz // H`, `off_h = off_hz % H`: unpack "which (batch, head)" from one number.
  (`//` = whole-number division, `%` = remainder.)
- `.to(tl.int64)`: compute big memory offsets in 64-bit. In 32-bit, offsets past ~2 billion
  wrap around and silently read the wrong memory.
- `kT` is K loaded **already transposed** (`[HEAD_DIM, BLOCK_N]`), by swapping which index
  goes down and which goes across. So `tl.dot(q, kT)` = Q·Kᵀ.

<!-- impl-file: kernels/attn_naive.py part: 3/5 -->
📄 **`kernels/attn_naive.py`** · part 3 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.jit
def _softmax_kernel(S, P, N_CTX, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    """Row-wise softmax of one row of S, with the max subtracted for stability."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    valid = offs < N_CTX
    if CAUSAL:
        valid = valid & (offs <= row % N_CTX)

    s = tl.load(S + row * N_CTX + offs, mask=valid, other=float("-inf")).to(tl.float32)
    s = s - tl.max(s, axis=0)
    p = tl.exp(s)
    p = p / tl.sum(p, axis=0)
    tl.store(P + row * N_CTX + offs, p.to(P.dtype.element_ty), mask=offs < N_CTX)
```

**Kernel 2, softmax.** One program per row of S.
- `BLOCK` is the next power of 2 ≥ N (`tl.arange` needs a power of 2), so the extra slots are masked.
- Masked slots load `-inf`. Since `exp(-inf) = 0`, they add nothing to the sum.
- **Causal**: in GPT-style models each token may only look at itself and earlier tokens.
  `offs <= row % N_CTX` keeps only columns up to this row's own position.
- `s - tl.max(s, axis=0)`: subtract the row max before `exp` so nothing overflows. The
  result is mathematically unchanged.
- It writes P over S in the same memory ("in place"), saving one N × N buffer.

<!-- impl-file: kernels/attn_naive.py part: 4/5 -->
📄 **`kernels/attn_naive.py`** · part 4 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.jit
def _pv_kernel(
    P, V, Out,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """O[b, h, m, :] = sum_n P[b, h, m, n] * V[b, h, n, :] for one row tile."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    p_base = P + off_hz.to(tl.int64) * N_CTX * N_CTX
    v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        p = tl.load(p_base + offs_m[:, None] * N_CTX + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
```

**Kernel 3, output.** Each program takes a tile of rows and walks across P and down V,
accumulating `P @ V` in fp32, just like the matmul loop.

<!-- impl-file: kernels/attn_naive.py part: 5/5 -->
📄 **`kernels/attn_naive.py`** · part 5 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def naive_attention(q, k, v, causal=False, sm_scale=None):
    """Standard attention that materializes the N x N score matrix in HBM.

    q, k, v: [B, H, N, D] fp16/bf16 tensors. Returns O with the same shape.
    """
    B, H, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "q, k, v must share a shape"
    assert D in (16, 32, 64, 128), "head dim must be 16, 32, 64 or 128"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # The N x N intermediate. Kept in the input dtype, like eager PyTorch.
    s = torch.empty((B * H, N, N), device=q.device, dtype=q.dtype)
    o = torch.empty_like(q)

    BLOCK_M, BLOCK_N = (32, 16) if is_interpreter() else (64, 64)

    grid_qk = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N), B * H)
    _qk_kernel[grid_qk](
        q, k, s, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    # Softmax in place: S becomes P. One program per row.
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    _softmax_kernel[(B * H * N,)](s, s, N, CAUSAL=causal, BLOCK=BLOCK, num_warps=num_warps)

    grid_pv = (triton.cdiv(N, BLOCK_M), B * H)
    _pv_kernel[grid_pv](
        s, v, o,
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return o
```

The launcher: allocate S (`B*H` tables of `N × N`), launch the three kernels in order.
- `sm_scale = 1/sqrt(D)`: the standard attention scaling.
- `num_warps` grows with row length so long rows get more threads.

**`tests/test_attn_naive.py`**:

<!-- impl-file: tests/test_attn_naive.py part: 1/1 -->
📄 **`tests/test_attn_naive.py`** · create this file and paste **exactly** this into it:

```python
import pytest
import torch

from helpers import ON_GPU, rand_qkv, reference_attention
from kernels.attn_naive import naive_attention

SHAPES = (
    [(1, 4, 256, 64), (2, 8, 512, 64), (1, 4, 1000, 128)]
    if ON_GPU
    else [(1, 2, 64, 32), (2, 2, 100, 64), (1, 1, 77, 16)]
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_naive_vs_torch(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    o = naive_attention(q, k, v, causal=causal)
    torch_o = reference_attention(q, k, v, causal=causal)
    torch.testing.assert_close(o, torch_o, rtol=1e-2, atol=1e-2)
```

Two `parametrize` lines multiply: 3 shapes × 2 (causal or not) = 6 tests.

<!-- run -->
▶ Run:

```bash
python -m pytest tests/test_attn_naive.py -v
```

✅ Expected: `6 passed`.

<!-- run -->
▶ Run:

```bash
git add kernels/attn_naive.py tests/test_attn_naive.py
git commit -m "feat(attn): naive Triton attention that materializes QK^T in HBM"
```

**Check yourself:** (1) How many bytes is S for B=1, H=16, N=8192 in fp16? (Answer: 2 GiB.)
(2) Why load masked slots as `-inf` and not `0`?

---

# Phase 5: FlashAttention forward, the main kernel (guide Day 4)

Each program owns `BLOCK_M` query rows of one head. It loads them once, then walks over K
and V one tile at a time, keeping `m` (running max), `l` (running sum) and `acc` (running
output) in fast on-chip registers. The N × N table never exists.

<!-- impl-file: kernels/attn_fwd.py part: 1/9 -->
📄 **`kernels/attn_fwd.py`** · part 1 of 9. Create this file and paste this in:

```python
"""Day 4: the FlashAttention v1 forward kernel.

One program owns one tile of BLOCK_M query rows for one (batch, head). It
walks over K and V in tiles of BLOCK_N rows and keeps three running values
per query row in registers:

    m_i  - the largest score seen so far       (for numerical stability)
    l_i  - the softmax denominator so far       (sum of exp(score - m_i))
    acc  - the un-normalized output so far      (sum of exp(score - m_i) * V)

When a new tile raises the max, the old l_i and acc are rescaled by
exp(m_old - m_new) so everything stays in the same units. After the last
tile, acc / l_i is the exact softmax(QK^T)V. The N x N matrix never exists.

Scores are kept in base-2 units (multiplied by log2(e)) so the kernel can use
exp2, which maps to a single hardware instruction. The saved log-sum-exp
(LSE) is in those same base-2 units; the backward pass reads it as-is.
"""
import math

import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg

LOG2E = 1.4426950408889634  # log2(e): exp(x) == exp2(x * LOG2E)
```

`LOG2E`: `exp(x) = exp2(x × 1.4427)`. GPUs have a fast native `exp2`, so we fold 1.4427
into the scale once and use `exp2` everywhere.

<!-- impl-file: kernels/attn_fwd.py part: 2/9 -->
📄 **`kernels/attn_fwd.py`** · part 2 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base, stride_kn, stride_kk, stride_vn, stride_vk,
    offs_m, qk_scale, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Loop over K/V tiles and update (acc, l_i, m_i) with the online softmax."""
    start_m = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)

    hi = N_CTX
    if CAUSAL:
        # Keys after the last query row in this tile are all masked: skip them.
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
```

**The inner loop's setup.** It's a separate `@triton.jit` function so the main kernel stays
readable. Triton inlines it, so it costs nothing.
- `hi` = where the loop over keys stops. Normally all N keys.
- With **causal**, a query tile ending at row `(start_m+1)*BLOCK_M - 1` can't see any key
  after that row, so the loop stops there. That skips about half the work.

<!-- impl-file: kernels/attn_fwd.py part: 3/9 -->
📄 **`kernels/attn_fwd.py`** · part 3 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        # K tile loaded transposed: [HEAD_DIM, BLOCK_N].
        kT = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk,
                     mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, kT) * qk_scale                       # [BLOCK_M, BLOCK_N]

        valid = mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))
```

**Per key tile, part 1: scores.**
- Load the K tile transposed, `qk = q @ kT × qk_scale` → a `[BLOCK_M, BLOCK_N]` tile of scores.
- `valid` = "this key exists" AND (if causal) "key position ≤ query position".
- `tl.where(valid, qk, -inf)`: invalid scores become `-inf`, so their `exp2` is 0.

> **The guide's causal bug, fixed here:** the guide only stopped the loop early. But the
> *last* tile straddles the diagonal, and without this per-element mask the early rows in
> it would see a few future tokens. The per-element mask fixes that.

<!-- impl-file: kernels/attn_fwd.py part: 4/9 -->
📄 **`kernels/attn_fwd.py`** · part 4 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
        # Online softmax update.
        m_ij = tl.maximum(m_i, tl.max(qk, 1))               # new running max
        p = tl.math.exp2(qk - m_ij[:, None])                # [BLOCK_M, BLOCK_N]
        alpha = tl.math.exp2(m_i - m_ij)                    # rescale old stats
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
```

**Per key tile, part 2: the online softmax.** This is the whole FlashAttention trick, line
for line the recurrence from Phase 2:

| Code | Meaning |
|---|---|
| `m_ij = max(m_i, rowmax(qk))` | new running max |
| `p = exp2(qk - m_ij)` | this tile's un-normalized probabilities |
| `alpha = exp2(m_i - m_ij)` | how much to shrink the old totals (1 if the max didn't change) |
| `l_i = l_i * alpha + rowsum(p)` | update the running sum |
| `acc = acc * alpha` | shrink the old output **(forget this and everything is wrong)** |
| `acc = dot(p, v, acc)` | add this tile's contribution |
| `m_i = m_ij` | remember the new max |

<!-- impl-file: kernels/attn_fwd.py part: 5/9 -->
📄 **`kernels/attn_fwd.py`** · part 5 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    return acc, l_i, m_i


@triton.autotune(
    configs=autotune_configs(
        [
            cfg(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
            cfg(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3),
            cfg(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=4),
            cfg(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=4),
            cfg(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=3),
        ],
        # Small, unequal blocks in the interpreter so tests hit the
        # multi-tile, padding and causal-diagonal paths at tiny N.
        cfg(BLOCK_M=32, BLOCK_N=16),
    ),
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, LSE, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
```

The inner function returns its three running values. Below it is the main kernel's header:
- The autotune list has 5 GPU settings. The CPU-emulator setting `BLOCK_M=32, BLOCK_N=16` is
  deliberately small and unequal, so tiny test inputs still cross several tiles and the diagonal.
- `key=["N_CTX", "HEAD_DIM", "CAUSAL"]`: re-tune whenever these change.
- Strides for Q, K, V and Out are passed separately, so each may have its own memory layout.

<!-- impl-file: kernels/attn_fwd.py part: 6/9 -->
📄 **`kernels/attn_fwd.py`** · part 6 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    """Grid: (cdiv(N, BLOCK_M), B * H). Writes O and the base-2 LSE per row."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # int64 base offsets: B*H*N*D elements overflows int32 at large shapes.
    q_base = Q + off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_base = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_base = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_base = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    # m starts at -inf, so the first tile's alpha = exp2(-inf) = 0 wipes l and acc.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # fold log2(e) in so we can use exp2

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=mask_m[:, None], other=0.0)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        k_base, v_base, stride_kn, stride_kk, stride_vn, stride_vk,
        offs_m, qk_scale, N_CTX,
        HEAD_DIM, BLOCK_M, BLOCK_N, CAUSAL,
    )
```

**Setup.** Grid = `(number of query tiles, B × H)`.
- Base addresses for this (batch, head), computed in int64.
- `m_i` starts at `-inf` and `l_i` at `0`. On the first tile, `alpha = exp2(-inf) = 0` wipes
  the old (empty) values. (The guide starts `l_i` at 1.0. Because of that `alpha = 0`, it
  makes no difference.)
- Q is loaded **once** and stays on-chip for the whole loop.

<!-- impl-file: kernels/attn_fwd.py part: 7/9 -->
📄 **`kernels/attn_fwd.py`** · part 7 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    # Epilogue: normalize once, save LSE for the backward pass.
    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)
    tl.store(LSE + off_hz.to(tl.int64) * N_CTX + offs_m, lse, mask=mask_m)
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
```

**Finish.** Divide by the total once (`acc / l_i`), then save:
- `lse = m_i + log2(l_i)`: the **log-sum-exp**, one number per row, in base 2. The backward
  pass rebuilds any probability from it: `p = exp2(score - lse)`.
- The output, converted back to fp16/bf16.

<!-- impl-file: kernels/attn_fwd.py part: 8/9 -->
📄 **`kernels/attn_fwd.py`** · part 8 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def _flash_fwd(q, k, v, causal, sm_scale):
    """Launch the forward kernel. Returns (O, LSE in base-2 units, shape [B*H, N])."""
    B, H, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "q, k, v must share a shape"
    assert q.dtype in (torch.float16, torch.bfloat16), "use fp16 or bf16 inputs"
    assert D in (16, 32, 64, 128), "head dim must be 16, 32, 64 or 128"

    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)
    _attn_fwd_kernel[grid](
        q, k, v, sm_scale, lse, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return o, lse
```

`_flash_fwd` (the leading `_` means "internal use") checks the inputs and launches the kernel.
`grid` is a lambda because `BLOCK_M` is chosen by autotune.

<!-- impl-file: kernels/attn_fwd.py part: 9/9 -->
📄 **`kernels/attn_fwd.py`** · part 9 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def flash_attn_fwd(q, k, v, causal=False, sm_scale=None):
    """FlashAttention forward (no autograd). Use `flash_attention` for training.

    q, k, v: [B, H, N, D] fp16/bf16. Returns (O, LSE) where LSE has shape
    [B, H, N] and equals torch.logsumexp(Q @ K^T * sm_scale, dim=-1).
    """
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o, lse2 = _flash_fwd(q, k, v, causal, sm_scale)
    return o, (lse2 * math.log(2.0)).view(B, H, N)
```

The public function. It converts the LSE from base 2 to natural log (`× ln 2`), so you can
compare it directly with `torch.logsumexp`.

**`tests/test_attn_fwd.py`**:

<!-- impl-file: tests/test_attn_fwd.py part: 1/1 -->
📄 **`tests/test_attn_fwd.py`** · create this file and paste **exactly** this into it:

```python
import pytest
import torch

from helpers import BF16_OK, DEVICE, ON_GPU, rand_qkv, reference_attention
from kernels.attn_fwd import flash_attn_fwd

SHAPES = (
    [(1, 4, 512, 64), (2, 8, 1024, 64), (1, 16, 2048, 64), (1, 8, 1024, 128), (1, 4, 1000, 64)]
    if ON_GPU
    else [(1, 2, 64, 32), (2, 2, 100, 64), (1, 1, 77, 16), (1, 1, 48, 128)]
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_flash_fwd(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    o, _ = flash_attn_fwd(q, k, v, causal=causal)
    torch.testing.assert_close(o, reference_attention(q, k, v, causal), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_fwd_lse(causal):
    """Debug-playbook checkpoint 2: m + log(l) must equal torch.logsumexp(S)."""
    B, H, N, D = (1, 2, 256, 64) if ON_GPU else (1, 1, 70, 32)
    q, k, v = rand_qkv(B, H, N, D)
    _, lse = flash_attn_fwd(q, k, v, causal=causal)
    s = (q.float() @ k.float().transpose(-1, -2)) / D ** 0.5
    if causal:
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, dim=-1), rtol=1e-3, atol=1e-3)


def test_flash_fwd_strided():
    """q/k/v as non-contiguous views of a [B, N, H, D] tensor (common in real models)."""
    B, H, N, D = (2, 4, 512, 64) if ON_GPU else (1, 2, 64, 32)
    torch.manual_seed(0)
    q, k, v = [torch.randn(B, N, H, D, device=DEVICE, dtype=torch.float16).transpose(1, 2)
               for _ in range(3)]
    o, _ = flash_attn_fwd(q, k, v)
    torch.testing.assert_close(o, reference_attention(q, k, v), rtol=1e-2, atol=1e-2)


# Triton's CPU interpreter computes tl.dot on bf16 inputs incorrectly (it is a
# numpy emulation with no bf16 type), and pre-Ampere GPUs have no bf16 tensor
# cores, so this only runs on an Ampere-or-newer GPU.
@pytest.mark.skipif(not BF16_OK, reason="needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16")
@pytest.mark.parametrize("causal", [False, True])
def test_flash_fwd_bf16(causal):
    q, k, v = rand_qkv(2, 4, 512, 64, dtype=torch.bfloat16)
    o, _ = flash_attn_fwd(q, k, v, causal=causal)
    torch.testing.assert_close(o, reference_attention(q, k, v, causal), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not ON_GPU, reason="PASS gate size; needs a GPU")
def test_pass_gate_forward_n2048():
    """Appendix C, criterion 1: N=2048, D=64, fp16, rtol=atol=1e-2."""
    q, k, v = rand_qkv(1, 16, 2048, 64)
    o, _ = flash_attn_fwd(q, k, v)
    torch.testing.assert_close(o, reference_attention(q, k, v), rtol=1e-2, atol=1e-2)
```

- `test_flash_fwd`: 4 shapes × causal/not = 8 tests, including N = 77, which isn't a
  multiple of any tile size.
- `test_flash_fwd_lse`: checks the saved LSE against `torch.logsumexp` (the guide's debug checkpoint 2).
- `test_flash_fwd_strided`: inputs that are transposed views.
- `@pytest.mark.skipif(...)`: skip a test under a condition (bf16 without a capable GPU,
  and the full-size PASS-gate test on CPU).

<!-- run -->
▶ Run:

```bash
python -m pytest tests/test_attn_fwd.py -v -rs
```

✅ Expected (`-rs` prints the reasons for skips): `11 passed, 3 skipped`. The 3 skips are
the 2 bf16 tests and the N=2048 PASS-gate test. **Skipped means not run.** Those three run in Phase 9 on a GPU.

**Prove the test can catch a bug (do it!):** in `kernels/attn_fwd.py` change
`acc = acc * alpha[:, None]` to `acc = acc`, then rerun the command. Many tests should now
FAIL. (When this was tried while building the repo: 9 failed.) Then **change it back** and
rerun. Everything must pass again.

<!-- run -->
▶ Run:

```bash
git add kernels/attn_fwd.py tests/test_attn_fwd.py
git commit -m "feat(attn): flash forward kernel with online softmax tiling"
```

**Check yourself:** (1) Why is `alpha` needed? (2) What is `lse` used for?
(3) Why does the causal loop stop at `(start_m+1)*BLOCK_M` *and* still need a mask?

---

# Phase 6: FlashAttention backward (guide Day 5)

Training needs **gradients**: how much each input (Q, K, V) should change to reduce the
error. Computing them is the **backward pass**. Given `dO` (the gradient of the output):

```text
D   = rowsum(dO * O)          one number per row
P   = exp2(S*log2e - LSE)     rebuilt from the saved LSE, never stored
dV  = P^T @ dO
dP  = dO @ V^T
dS  = P * (dP - D)            the softmax "backward" rule
dQ  = dS @ K   * scale
dK  = dS^T @ Q * scale
```

We **recompute** P tile by tile instead of storing it. Recomputing costs some math;
storing it would cost N × N memory, exactly what we avoided.

<!-- impl-file: kernels/attn_bwd.py part: 1/9 -->
📄 **`kernels/attn_bwd.py`** · part 1 of 9. Create this file and paste this in:

```python
"""Day 5: the FlashAttention v1 backward pass, by recomputation.

The forward pass threw away the N x N probability matrix P. Instead of
storing it, the backward pass rebuilds each tile of P on the fly from Q, K
and the saved log-sum-exp (LSE):  P_ij = exp2(S_ij * log2(e) - LSE_i).

With D_i = rowsum(dO_i * O_i), the gradients are:

    dV = P^T @ dO
    dP = dO @ V^T
    dS = P * (dP - D)            # softmax backward
    dQ = dS @ K * scale
    dK = dS^T @ Q * scale

Three kernels:
    _attn_bwd_preprocess  D = rowsum(O * dO), one pass.
    _attn_bwd_dkdv        one program per K/V tile, loops over Q tiles.
    _attn_bwd_dq          one program per Q tile, loops over K/V tiles.

Splitting dK/dV from dQ means every output tile has exactly one writer, so
no atomics are needed. The price is computing QK^T and dO V^T twice.
"""
import math

import torch
import triton
import triton.language as tl

from kernels._utils import autotune_configs, cfg
from kernels.attn_fwd import _flash_fwd
```

Docstring (the formulas above) and imports. Note `from kernels.attn_fwd import _flash_fwd`:
the backward reuses your forward.

<!-- impl-file: kernels/attn_bwd.py part: 2/9 -->
📄 **`kernels/attn_bwd.py`** · part 2 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Delta[bh, m] = sum_d O[bh, m, d] * dO[bh, m, d]. Inputs are contiguous."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    base = off_hz * N_CTX * HEAD_DIM
    ptrs = base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    o = tl.load(O + ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=mask_m)
```

**Kernel 1: `D = rowsum(O × dO)`.** Simple: load a tile of rows of O and dO, multiply
element-wise, sum each row. The inputs are **contiguous** (the wrapper guarantees it), so a
row is just `HEAD_DIM` numbers in a row: address = `row × HEAD_DIM + column`.

<!-- impl-file: kernels/attn_bwd.py part: 3/9 -->
📄 **`kernels/attn_bwd.py`** · part 3 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
_BWD_CONFIGS = autotune_configs(
    [
        cfg(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2),
        cfg(BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2),
        cfg(BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=3),
        cfg(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=3),
        cfg(BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=3),
    ],
    cfg(BLOCK_M=16, BLOCK_N=32),
)


@triton.autotune(configs=_BWD_CONFIGS, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, sm_scale, DO, DK, DV, LSE, Delta,
    stride_z, stride_h, stride_n, stride_d,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
```

The tile settings, shared by both gradient kernels, and the header of **kernel 2 (dK, dV)**.
Q, K, V, dO, dK, dV all have the same shape and are all contiguous, so they share one set of strides.

<!-- impl-file: kernels/attn_bwd.py part: 4/9 -->
📄 **`kernels/attn_bwd.py`** · part 4 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    """Grid: (cdiv(N, BLOCK_N), B * H). Accumulates dK and dV for one K/V tile.

    All of Q, K, V, dO, dK, dV are contiguous [B, H, N, D], so they share strides.
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h
    row_base = off_hz.to(tl.int64) * N_CTX

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_CTX

    # K_j and V_j stay in SRAM for the whole program.
    kv_ptrs = base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
    k = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    lo = 0
    if CAUSAL:
        # Query rows before this key tile cannot see it: start at its diagonal.
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
```

**Kernel 2 setup.** Each program owns one tile of **keys** (K_j, V_j), loads it once, and
keeps fp32 accumulators `dk`, `dv`.
- With causal, queries *before* this key tile can't see it, so the loop starts at the
  diagonal (`lo`) instead of 0.

<!-- impl-file: kernels/attn_bwd.py part: 5/9 -->
📄 **`kernels/attn_bwd.py`** · part 5 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX
        q_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
        q = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(LSE + row_base + offs_m, mask=mask_m, other=0.0)
        delta = tl.load(Delta + row_base + offs_m, mask=mask_m, other=0.0)

        # Recompute this tile of P from the saved LSE.
        qk = tl.dot(q, tl.trans(k))                                  # [BLOCK_M, BLOCK_N]
        p = tl.math.exp2(qk * qk_scale - lse[:, None])
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)

        # dV_j += P^T @ dO
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)
        # dP = dO @ V^T ;  dS = P * (dP - D)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        # dK_j += dS^T @ Q   (scale applied once at the end)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)

    dk = dk * sm_scale
    tl.store(DK + kv_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV + kv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])
```

**Kernel 2 loop**, over all query tiles:
1. Load Q, dO, LSE and D for these rows.
2. Rebuild P for this tile: `exp2(q·kᵀ × qk_scale − lse)`.
3. Zero out invalid entries (past the end, or future tokens if causal).
4. `dv += Pᵀ @ dO`.
5. `dp = dO @ Vᵀ`, `ds = P × (dp − D)`.
6. `dk += dSᵀ @ Q`. The `sm_scale` is applied once at the end.

`tl.trans(x)` = transpose (swap rows and columns).

<!-- impl-file: kernels/attn_bwd.py part: 6/9 -->
📄 **`kernels/attn_bwd.py`** · part 6 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
@triton.autotune(configs=_BWD_CONFIGS, key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_bwd_dq(
    Q, K, V, sm_scale, DO, DQ, LSE, Delta,
    stride_z, stride_h, stride_n, stride_d,
    H, N_CTX,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Grid: (cdiv(N, BLOCK_M), B * H). Accumulates dQ for one Q tile."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    base = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h
    row_base = off_hz.to(tl.int64) * N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    q_ptrs = base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    q = tl.load(Q + q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + q_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(LSE + row_base + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + row_base + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        kv_ptrs = base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d
        k = tl.load(K + kv_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(V + kv_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        p = tl.math.exp2(qk * qk_scale - lse[:, None])
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        # dQ_i += dS @ K
        dq = tl.dot(ds.to(k.dtype), k, dq)

    dq = dq * sm_scale
    tl.store(DQ + q_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
```

**Kernel 3 (dQ)**: the mirror image. Each program owns a tile of **queries**, loops over
key tiles, rebuilds P the same way, and accumulates `dq += dS @ K`.

Why separate kernels for dK/dV and dQ? If one program computed all three, many programs
would have to add into the same dQ rows at the same time. That needs **atomic** adds
(slow, and results vary slightly run to run). Separate kernels mean each output tile has
exactly one owner. The cost: scores are computed twice.

> **The guide's backward was "abridged" and wrong:** V reused K's address
> (`v_ptrs = k_ptrs # placeholder`), and the dQ kernel was just "similarly...". All three
> kernels are complete here.

<!-- impl-file: kernels/attn_bwd.py part: 7/9 -->
📄 **`kernels/attn_bwd.py`** · part 7 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def _flash_bwd(do, q, k, v, o, lse, causal, sm_scale):
    """Launch the three backward kernels. All tensors must be contiguous."""
    B, H, N, D = q.shape
    delta = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    PRE_BLOCK = 64
    _attn_bwd_preprocess[(triton.cdiv(N, PRE_BLOCK), B * H)](
        o, do, delta, N, HEAD_DIM=D, BLOCK_M=PRE_BLOCK,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))

    grid_kv = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B * H)
    _attn_bwd_dkdv[grid_kv](
        q, k, v, sm_scale, do, dk, dv, lse, delta,
        *strides, H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    grid_q = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)
    _attn_bwd_dq[grid_q](
        q, k, v, sm_scale, do, dq, lse, delta,
        *strides, H, N,
        HEAD_DIM=D, CAUSAL=causal,
    )
    return dq, dk, dv
```

Launches the three kernels in order: D first (the other two need it), then dK/dV, then dQ.

<!-- impl-file: kernels/attn_bwd.py part: 8/9 -->
📄 **`kernels/attn_bwd.py`** · part 8 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
class FlashAttnFunc(torch.autograd.Function):
    """autograd wrapper: forward saves (Q, K, V, O, LSE), never P."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        o, lse = _flash_fwd(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _flash_bwd(do.contiguous(), q, k, v, o, lse, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None
```

**Plugging into PyTorch's training system.** **Autograd** is PyTorch's automatic gradient
machinery. A `torch.autograd.Function` has two parts:
- `forward(ctx, ...)`: compute the output and save what backward will need with
  `ctx.save_for_backward(...)`. We save Q, K, V, O and LSE, **never P**.
- `backward(ctx, do)`: receives dO and returns **one gradient per forward input**:
  `dq, dk, dv` for q, k, v and `None, None` for `causal` and `sm_scale` (not tensors).
- `.contiguous()` makes a compact copy if a tensor is a strided view, so the backward
  kernels can share one set of strides.

<!-- impl-file: kernels/attn_bwd.py part: 9/9 -->
📄 **`kernels/attn_bwd.py`** · part 9 of 9. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def flash_attention(q, k, v, causal=False, sm_scale=None):
    """Differentiable FlashAttention. q, k, v: [B, H, N, D] fp16/bf16.

    Drop-in for F.scaled_dot_product_attention(q, k, v, is_causal=causal).
    """
    return FlashAttnFunc.apply(q, k, v, causal, sm_scale)
```

The one-line public function. Use it exactly like PyTorch's
`F.scaled_dot_product_attention(q, k, v, is_causal=...)`.

Now fill in **`kernels/__init__.py`**. It was empty since Phase 1; **replace** its contents with:

<!-- impl-file: kernels/__init__.py part: 1/1 -->
📄 **`kernels/__init__.py`** · create this file and paste **exactly** this into it:

```python
"""flash-att-triton: FlashAttention v1 (Dao et al., 2022) in Triton."""
from kernels.attn_bwd import flash_attention
from kernels.attn_fwd import flash_attn_fwd
from kernels.attn_naive import naive_attention
from kernels.matmul import matmul

__all__ = ["flash_attention", "flash_attn_fwd", "naive_attention", "matmul"]
__version__ = "0.1.0"
```

Now `from kernels import flash_attention` works. `__all__` lists the public names.

**`tests/test_attn_bwd.py`**:

<!-- impl-file: tests/test_attn_bwd.py part: 1/1 -->
📄 **`tests/test_attn_bwd.py`** · create this file and paste **exactly** this into it:

```python
import pytest
import torch
import torch.nn.functional as F

from helpers import BF16_OK, ON_GPU, rand_qkv
from kernels.attn_bwd import flash_attention

SHAPES = (
    [(1, 4, 256, 64), (2, 4, 512, 64), (1, 4, 1000, 64), (1, 2, 512, 128)]
    if ON_GPU
    else [(1, 1, 64, 32), (1, 2, 100, 64), (1, 1, 50, 16)]
)


def _grads(fn, q, k, v, do):
    q, k, v = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    o = fn(q, k, v)
    o.backward(do)
    return o, q.grad, k.grad, v.grad


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,N,D", SHAPES)
def test_bwd_matches_torch(B, H, N, D, causal):
    q, k, v = rand_qkv(B, H, N, D)
    do = torch.randn_like(q)

    o, dq, dk, dv = _grads(lambda a, b, c: flash_attention(a, b, c, causal=causal), q, k, v, do)
    # Reference in fp32, so we compare against the true gradient, not fp16 noise.
    ref = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float(), is_causal=causal),
        q, k, v, do.float(),
    )

    torch.testing.assert_close(o, ref[0].to(o.dtype), rtol=1e-2, atol=1e-2)
    for name, got, want in zip("qkv", (dq, dk, dv), ref[1:]):
        torch.testing.assert_close(got, want.to(got.dtype), rtol=2e-2, atol=2e-2,
                                   msg=lambda m, n=name: f"d{n}: {m}")


def test_bwd_sum_loss_as_in_guide():
    """The guide's exact check: o.sum().backward() at N=256, D=64 (smaller on CPU)."""
    B, H, N, D = (1, 4, 256, 64) if ON_GPU else (1, 1, 64, 64)
    q, k, v = rand_qkv(B, H, N, D, requires_grad=True)
    qr, kr, vr = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    F.scaled_dot_product_attention(qr, kr, vr).sum().backward()
    flash_attention(q, k, v).sum().backward()
    for got, want in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not BF16_OK, reason="needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16")
@pytest.mark.parametrize("causal", [False, True])
def test_bwd_bf16(causal):
    q, k, v = rand_qkv(1, 4, 512, 64, dtype=torch.bfloat16)
    do = torch.randn_like(q)
    _, *got = _grads(lambda a, b, c: flash_attention(a, b, c, causal=causal), q, k, v, do)
    _, *want = _grads(
        lambda a, b, c: F.scaled_dot_product_attention(a.float(), b.float(), c.float(), is_causal=causal),
        q, k, v, do.float(),
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w.to(g.dtype), rtol=5e-2, atol=5e-2)
```

- `_grads`: makes fresh copies of q, k, v that track gradients (`requires_grad_(True)`),
  runs the function, calls `.backward(do)` with a **random** dO, and returns the gradients.
- The reference runs in fp32. The tolerance is 2e-2, as the guide specifies.
- `test_bwd_sum_loss_as_in_guide`: the guide's exact check, `o.sum().backward()`.
  A sum gives dO = 1 everywhere, which can hide bugs. That's why the random-dO test exists too.

<!-- run -->
▶ Run (the backward tests, then **everything**):

```bash
python -m pytest tests/test_attn_bwd.py -v
python -m pytest tests/ -rs
```

✅ Expected, first command: `7 passed, 2 skipped`. Second command:

```text
SKIPPED [2] tests/test_attn_bwd.py:52: needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16
SKIPPED [2] tests/test_attn_fwd.py:47: needs a GPU with bf16 (A100/RTX 30xx+); interpreter tl.dot is wrong for bf16
SKIPPED [1] tests/test_attn_fwd.py:55: PASS gate size; needs a GPU
======================== 28 passed, 5 skipped in 8.19s =========================
```

**28 passed, 5 skipped** is the target on a CPU. Anything that says `failed` means a
mistake in your files. Compare them with the parts above, character by character
(indentation included).

**Prove it (do it!):** in `_attn_bwd_dq` change `dq = dq * sm_scale` to `dq = dq * 1.0`,
rerun `python -m pytest tests/test_attn_bwd.py`. It should fail (when tried: 7 failed).
Change it back.

<!-- run -->
▶ Run:

```bash
git add kernels/attn_bwd.py kernels/__init__.py tests/test_attn_bwd.py
git commit -m "feat(attn): backward pass via recomputation, gradients match torch within 2e-2"
```

**Check yourself:** (1) Why is P not saved in forward? (2) What is D and why is it needed?
(3) Why does `backward` return five things?

---

# Phase 7: Benchmark and plot scripts (guide Day 6)

A **benchmark** measures speed. **Latency** is the time per call in milliseconds (ms).
Lower is better.

**`bench/run.py`**:

<!-- impl-file: bench/run.py part: 1/5 -->
📄 **`bench/run.py`** · part 1 of 5. Create this file and paste this in:

```python
"""Day 6: latency of naive Triton vs our flash kernel vs torch SDPA.

    python bench/run.py --seqlen 1024,2048,4096,8192
    python bench/run.py --mode fwd_bwd          # training step (no naive: it has no backward)

Writes bench/results/results_<mode>.json and prints a markdown table you can
paste into the README. Needs a CUDA GPU; the CPU interpreter is for
correctness only and its timings mean nothing.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton  # noqa: E402

from kernels.attn_bwd import flash_attention  # noqa: E402
from kernels.attn_fwd import flash_attn_fwd  # noqa: E402
from kernels.attn_naive import naive_attention  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
```

The `sys.path.insert(...)` line lets the script import `kernels` even though it lives in
the `bench/` folder.

<!-- impl-file: bench/run.py part: 2/5 -->
📄 **`bench/run.py`** · part 2 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def bench_ms(fn):
    """Median ms per call. do_bench handles warmup and flushes L2 between runs."""
    try:
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")


def attn_tflops(B, H, N, D, ms, causal, mode):
    flops = 4 * B * H * N * N * D  # two matmuls of 2*N*N*D each
    if causal:
        flops //= 2
    if mode == "fwd_bwd":
        flops = int(flops * 3.5)  # bwd = 2.5x fwd (5 matmuls vs 2)
    return flops / (ms * 1e-3) / 1e12
```

- `triton.testing.do_bench(fn)` runs `fn` many times (after a warm-up), clears the cache
  between runs, and returns the median time. It's more trustworthy than a hand-made timer.
- If the GPU runs **out of memory** (OOM), the result becomes `nan` ("not a number")
  instead of a crash. Naive attention at N=8192 needs 2 GB just for S.
- `attn_tflops`: converts time into **TFLOP/s** (trillions of operations per second),
  a GPU-independent measure of efficiency.

<!-- impl-file: bench/run.py part: 3/5 -->
📄 **`bench/run.py`** · part 3 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def run(seqlens, B=1, H=16, D=64, causal=False, mode="fwd", dtype=torch.float16):
    rows = []
    for N in seqlens:
        torch.manual_seed(0)
        q, k, v = [torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=(mode == "fwd_bwd"))
                   for _ in range(3)]
        if mode == "fwd":
            t_naive = bench_ms(lambda: naive_attention(q, k, v, causal=causal))
            t_ours = bench_ms(lambda: flash_attn_fwd(q, k, v, causal=causal))
            t_torch = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        else:
            do = torch.randn_like(q)
            t_naive = float("nan")
            t_ours = bench_ms(lambda: flash_attention(q, k, v, causal=causal).backward(do))
            t_torch = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal).backward(do))
        rows.append({
            "seqlen": N, "naive_ms": t_naive, "ours_ms": t_ours, "torch_ms": t_torch,
            "speedup_vs_naive": t_naive / t_ours,
            "ours_vs_torch": t_ours / t_torch,
            "ours_tflops": attn_tflops(B, H, N, D, t_ours, causal, mode),
        })
        del q, k, v
        torch.cuda.empty_cache()
    return rows
```

For each sequence length: make random inputs, time naive vs ours vs PyTorch SDPA.
In `fwd_bwd` mode it times a full training step; naive is skipped there because it has no backward.
`lambda: ...` wraps each call into a no-argument function for `do_bench`.

<!-- impl-file: bench/run.py part: 4/5 -->
📄 **`bench/run.py`** · part 4 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def markdown_table(rows):
    lines = [
        "| seqlen | naive Triton (ms) | this kernel (ms) | torch SDPA (ms) | speedup vs naive | ours / SDPA | TFLOP/s |",
        "|-------:|------------------:|-----------------:|----------------:|-----------------:|------------:|--------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seqlen']} | {r['naive_ms']:.3f} | {r['ours_ms']:.3f} | {r['torch_ms']:.3f} "
            f"| {r['speedup_vs_naive']:.2f}x | {r['ours_vs_torch']:.2f}x | {r['ours_tflops']:.1f} |"
        )
    return "\n".join(lines)
```

Formats the results as a Markdown table, ready to paste into the README.

<!-- impl-file: bench/run.py part: 5/5 -->
📄 **`bench/run.py`** · part 5 of 5. Leave one empty line after the previous part, then paste this **at the end** of the file:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=str, default="1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench/run.py needs a CUDA GPU (see README: Kaggle T4 is free).")

    seqlens = [int(x) for x in args.seqlen.split(",")]
    rows = run(seqlens, args.batch, args.heads, args.head_dim, args.causal, args.mode)

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "triton": triton.__version__,
        "B": args.batch, "H": args.heads, "D": args.head_dim,
        "causal": args.causal, "mode": args.mode, "dtype": "float16",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_{args.mode}{'_causal' if args.causal else ''}.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)

    print(f"{meta['gpu']} | B={args.batch} H={args.heads} D={args.head_dim} "
          f"causal={args.causal} mode={args.mode} fp16\n")
    print(markdown_table(rows))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
```

**argparse** reads command-line options (`--seqlen 1024,2048`, `--causal`, …). Results
are saved as JSON (a standard text format for data) in `bench/results/`, along with the
GPU name and library versions, so numbers can always be traced back.

**`bench/plot.py`**: turns the JSON into a chart:

<!-- impl-file: bench/plot.py part: 1/1 -->
📄 **`bench/plot.py`** · create this file and paste **exactly** this into it:

```python
"""Plot latency vs seqlen from bench/results/results_fwd.json.

    python bench/run.py          # produces the JSON
    python bench/plot.py         # produces bench/results/latency.png
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(RESULTS_DIR, "results_fwd.json"))
    ap.add_argument("--output", default=os.path.join(RESULTS_DIR, "latency.png"))
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    meta, rows = data["meta"], data["rows"]
    xs = [r["seqlen"] for r in rows]

    plt.figure(figsize=(7, 4.5))
    for key, style, label in (
        ("naive_ms", "o-", "naive Triton (materializes N x N)"),
        ("ours_ms", "s-", "ours (flash)"),
        ("torch_ms", "^-", "torch SDPA"),
    ):
        ys = [r[key] for r in rows]
        if all(math.isnan(y) for y in ys):
            continue
        plt.plot(xs, ys, style, label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length")
    plt.ylabel("latency (ms, lower is better)")
    plt.title(f"FlashAttention v1 in Triton, {meta['mode']}, fp16, "
              f"B={meta['B']} H={meta['H']} D={meta['D']}\n{meta['gpu']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
```

`matplotlib.use("Agg")` = draw to a file, not a window (works on servers without a screen).
Log scales on both axes, because times grow roughly with N².

**`scripts/run_all_gpu.sh`**: one command for the GPU machine:

<!-- impl-file: scripts/run_all_gpu.sh part: 1/1 -->
📄 **`scripts/run_all_gpu.sh`** · create this file and paste **exactly** this into it:

```bash
#!/usr/bin/env bash
# One command on a GPU box (Kaggle / Colab / RunPod): test, benchmark, plot.
#   bash scripts/run_all_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"

echo "== 1/4 tests (GPU sizes, includes the N=2048 PASS-gate test) =="
python -m pytest tests/ -v

echo "== 2/4 forward benchmark =="
python bench/run.py --seqlen 1024,2048,4096,8192 --mode fwd

echo "== 3/4 forward+backward benchmark =="
python bench/run.py --seqlen 1024,2048,4096,8192 --mode fwd_bwd

echo "== 4/4 plot =="
python bench/plot.py

echo "Done. Paste the forward table into README.md and blog/post.md, commit bench/results/."
```

`set -euo pipefail` = stop at the first error instead of carrying on with broken results.

<!-- run -->
▶ Run (make the script executable, then commit):

```bash
chmod +x scripts/run_all_gpu.sh
git add bench/run.py bench/plot.py scripts/run_all_gpu.sh
git commit -m "bench: latency vs naive Triton and torch SDPA, JSON results and plot"
```

`chmod +x` = mark a file as runnable.

Try it on your laptop:

```bash
python bench/run.py
```

✅ Expected on a CPU-only machine: `bench/run.py needs a CUDA GPU (see README: Kaggle T4 is free).`
That's correct. Benchmarks are meaningless on the emulator, so the script refuses to run.

---

# Phase 8: Automatic tests on GitHub (CI)

**CI** (Continuous Integration) = GitHub runs your tests on its own computers every time you
push. If you break something, you see a red ❌ on GitHub. It uses the CPU emulator, so it's free.

📄 The file must be at exactly `.github/workflows/tests.yml`. The folder name starts with a dot.

<!-- impl-file: .github/workflows/tests.yml part: 1/1 -->
📄 **`.github/workflows/tests.yml`** · create this file and paste **exactly** this into it:

```yaml
name: tests (CPU, Triton interpreter)

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      TRITON_INTERPRET: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install CPU torch + triton
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install triton numpy pytest
      - name: Run tests in the interpreter
        run: python -m pytest tests/ -v
```

This is **YAML**, a settings format where **indentation matters** (spaces only). In words:
"on every push and pull request, on a fresh Ubuntu machine with `TRITON_INTERPRET=1`:
check out the code, install Python 3.12, install CPU PyTorch + Triton, run pytest".

<!-- run -->
▶ Run:

```bash
git add .github/workflows/tests.yml
git commit -m "ci: run correctness tests in the Triton CPU interpreter"
```

▶ Run (upload everything so far):

```bash
git push
```

On github.com open your repo → **Actions** tab. Within a few minutes the run should show a
green ✅. If it's red, click it and read the log.

---

# Phase 9: Run on a real GPU (guide Day 6, untested here)

> ⚠️ **Honesty:** nothing in this phase has been executed. No GPU was available. Expect to
> debug a little.

Full click-by-click Kaggle instructions are in **NOTES.md, Step 9**. Short version, in a
Kaggle notebook with **Accelerator = GPU T4 x2** and **Internet = On** (`!` runs a terminal
command inside a notebook cell, `%cd` changes folder):

```bash
!git clone https://github.com/<you>/flash-att-triton.git
%cd flash-att-triton
!pip install -q pytest matplotlib
!python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
!bash scripts/run_all_gpu.sh
```

- Kaggle already has PyTorch and Triton installed. **Don't** reinstall torch there.
- The first line of output must say `True`. If not, the GPU isn't switched on.
- The first run is slow: Triton compiles each kernel and autotune tries every setting.

What you must check in the output:
1. **pytest: `0 failed`.** On a T4: bf16 tests skip (no bf16 hardware) and everything else
   runs, including the PASS-gate test at N=2048.
2. **The forward table, row 2048, column "speedup vs naive": must be ≥ 1.5.**
3. Copy both printed tables and download `bench/results/` (Kaggle right panel → Output).

**Brutal truth:** you'll probably be **slower than torch SDPA**. PyTorch uses
FlashAttention-2, hand-written in CUDA by experts. Being within 1.2–2× of it is a decent
result for a from-scratch Triton kernel. Report what you measure, including where you lose.

---

# Phase 10: README, blog, pass/fail gate, release (guide Day 7)

### 10.1 README and blog

<!-- impl-starter: README.md -->
📄 **`README.md`** · start from this skeleton, then fill it in:

```markdown
# flash-att-triton

FlashAttention v1 (Dao et al., 2022) in Triton: forward with online-softmax tiling,
backward by recomputation, benchmarked against torch SDPA.

## Quickstart

    pip install -r requirements.txt
    pytest tests/ -v
    python bench/run.py --seqlen 1024,2048,4096,8192
    python bench/plot.py

## Results

(paste the table printed by bench/run.py; write the GPU name)

![latency](bench/results/latency.png)

## Reference

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135
```

<!-- impl-starter: blog/post.md -->
📄 **`blog/post.md`** · write it in your own words, using the guide's outline:

```markdown
# FlashAttention v1 in Triton, from scratch

## 1. Why I built this            (max 30 lines)
## 2. The IO problem               (max 40 lines: HBM vs SRAM, why memory is the bottleneck)
## 3. The online softmax           (max 60 lines: the recurrence, why it's exact, the backward)
## 4. The forward kernel           (max 40 lines: tiles, masks, exp2, causal)
## 5. Results                      (max 30 lines: your table, your plot, honest comparison with SDPA)
```

Full reference versions of both are in the original repo. Write yours first, compare after.

<!-- run -->
▶ Run:

```bash
git add README.md blog/post.md
git commit -m "docs: README and blog post"
git log --oneline
```

✅ Expected: **9 commits**, newest first (the IDs will differ):

```text
f00d123 docs: README and blog post
...
1a2b3c4 chore: project scaffold, requirements, MIT license
```

Then, after the GPU run, paste your real numbers into both files, add
`bench/results/latency.png` and the JSON files, commit, and `git push`.

### 10.2 The PASS/FAIL gate (from the guide, Appendix C)

Tick every box **honestly** before you tag:

- [ ] Forward matches PyTorch at N=2048, D=64, fp16, tolerance 1e-2 (the `test_pass_gate_forward_n2048` test, **on a GPU**)
- [ ] dQ, dK, dV match PyTorch within 2e-2 **on a GPU**
- [ ] ≥ 1.5× faster than naive at seqlen 2048 (**measured**)
- [ ] `pytest tests/` on the GPU: **0 failed**
- [ ] README has the filled table and plot
- [ ] Blog published somewhere public and linked from the README

**Do not tag if any box is empty.** Don't loosen a tolerance to make a test pass. That
hides a bug, it doesn't fix it.

### 10.3 Tag the release

```bash
git tag -a v0.1.0 -m "FlashAttention v1 in Triton - forward+backward, benchmarked vs torch SDPA"
git push origin v0.1.0
```

A **tag** is a permanent label on one commit. On GitHub → **Releases** → **Draft a new
release** → pick `v0.1.0` → write the notes **with your measured numbers**. Not the guide's
"2.5×" or "within 8%": those are the guide's guesses, not your results.

---

# If something doesn't match

| Problem | Most likely cause | Fix |
|---|---|---|
| `IndentationError` / `unexpected indent` | Spaces got changed while pasting | Re-paste that part; check the editor uses 4 spaces |
| `NameError` / `is not defined` | A part was skipped or pasted in the wrong order | Check every part number 1 … n of that file is present, in order |
| `ModuleNotFoundError: torch` | venv not active | `source .venv/bin/activate` |
| `ModuleNotFoundError: kernels` | Running from the wrong folder, or `python kernels/x.py` | `cd ~/flash-att-triton`; use `python -m pytest` / `python -m kernels.matmul` |
| Phase 3–5 tests fail with `ImportError` from `kernels/__init__.py` | You filled `__init__.py` too early | Keep it empty until Phase 6 (it imports files that don't exist yet) |
| A test says `FAILED ... Mismatched elements` | A real bug in your kernel code | Compare that kernel with this document line by line; it's usually one character |
| Tests take minutes on CPU | Normal-ish on slow laptops | The verified run took ~8 s for all 33 tests on a cloud CPU |
| `pip install triton` → "no matching distribution" | macOS or plain Windows | Use WSL2 or Kaggle |
| `git push` rejects your password | GitHub needs a token | See Phase 1.7 |

When you're completely stuck: delete the file and re-create it from this document,
part by part. That's faster than hunting for a typo for an hour.

---

# Final checklist

- [ ] 9 commits in `git log --oneline`
- [ ] `python -m pytest tests/` → 28 passed, 5 skipped (CPU) / 0 failed (GPU)
- [ ] Green ✅ in the GitHub **Actions** tab
- [ ] You did the three "prove it" experiments (break, see the failure, fix)
- [ ] You can answer every "Check yourself" question without looking
- [ ] GPU numbers in the README (Phase 9), PASS gate ticked, tag pushed (Phase 10)

The last two boxes are where the value is. Everything before them is typing.
