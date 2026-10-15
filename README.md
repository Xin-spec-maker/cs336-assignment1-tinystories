# CS336 Assignment 1: TinyStories from scratch

Personal learning project based on the Spring 2026 v26.0.3 handout. The core
pipeline has been run on real TinyStories text: our BPE/tokenizer → token IDs →
from-scratch Transformer → training/validation → checkpoint → generation.

The recorded run used Apple M5 / 16 GiB / MPS, float32, 3,344,064 parameters,
2,000 updates and 4,096,000 training tokens. Final validation NLL is **2.633829**
and perplexity **13.926994**, evaluated on all 202,721 next-token targets in the
selected validation subset. This is a small learning run, not the handout's
full-data benchmark or a claim to have completed every assignment experiment.

- [Measured results, exact samples and loss curves](reports/tinystories-core/RESULTS.md)
- [Chinese walkthrough and PPT outline](docs/CORE_WALKTHROUGH.md)
- [Training configuration](configs/tinystories-m5.json)

## Reproduce the core pipeline

Python 3.12.15 is pinned. Create a fresh environment on each machine:

```sh
uv sync --locked --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run --locked pytest
uv run --locked python -m cs336_basics.prepare_tinystories \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
```

Preparation selects 20,000 complete train stories and 1,000 valid stories from
the independent official splits. BPE learns 4096 tokens from only the first
2,000 train stories. The manifest records source URLs, hashes, counts, overlap
checks and tokenizer hashes. It never uses validation text to learn merges.
Prepared output is not silently overwritten; choose `--output-dir` for a new dataset.

Run the real-data smoke test, including checkpoint/resume:

```sh
uv run --locked python -m cs336_basics.train \
  --config configs/tinystories-smoke.json --data-dir data/tinystories-core \
  --output-dir runs/tinystories-smoke --stop-after 6
uv run --locked python -m cs336_basics.train \
  --config configs/tinystories-smoke.json --data-dir data/tinystories-core \
  --output-dir runs/tinystories-smoke --resume runs/tinystories-smoke/last.pt
```

Then run the modest formal training configuration:

```sh
uv run --locked python -m cs336_basics.train \
  --config configs/tinystories-m5.json --data-dir data/tinystories-core \
  --output-dir runs/tinystories-m5
uv run --locked python -m cs336_basics.generate \
  --checkpoint runs/tinystories-m5/last.pt \
  --vocab data/tinystories-core/vocab.json --merges data/tinystories-core/merges.json \
  --prompt 'Once upon a time, there was a little rabbit' \
  --device mps --temperature 0.8 --top-p 0.9 --seed 336 --max-new-tokens 256 \
  --output-json runs/tinystories-m5/sample.json
```

To resume an interrupted run, use the same training command with
`--resume runs/tinystories-m5/last.pt`. If its log contains steps newer than the
checkpoint, choose a new output directory for the resumed branch. Resume checks
the model/training configuration and data manifest before loading the optimizer
and RNG state. `--stop-after` stops at a chosen global step without changing the
planned learning-rate schedule. An already-completed run is not trained further.

Training logs are local JSONL; no W&B account is required. Change the configuration's
`device` to `cpu` if MPS is unavailable. Timings and exact samples may differ across
hardware or software versions; the published artifacts are the actual recorded run.

To regenerate the plot from the committed measurements without changing project dependencies:

```sh
uv run --with matplotlib --no-project python scripts/plot_training.py
```

## Versioned artifacts and local files

`data/`, `runs/`, virtual environments, caches, credentials and all `*.pt`/`*.pth`
checkpoints are ignored. Only code, configs, small result records and plots are
versioned. Back up your local `data/tinystories-core/` and `runs/tinystories-m5/`
separately if you need to preserve the exact trained checkpoint.

The original official test reference file `tests/fixtures/ts_tests/model.pt` is
also intentionally excluded by this checkpoint rule. Restore that fixture from
your original assignment package before running the full official test suite in
a fresh clone. All tests were run with the existing local reference fixture present.
The training and generation commands above do not depend on this test fixture.

## Original assignment setup reference

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

