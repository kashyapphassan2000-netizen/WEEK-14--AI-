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
