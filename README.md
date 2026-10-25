# Mini-Eval

Evaluation of [Mini-LLM](https://github.com/Mini-llm-lab/Mini-LLM) models against reference models (Pythia, ...)
with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
This repository is separate from the training repository on purpose: it only asks the
models for probabilities, and `lm-eval` computes the scores.

> **Server only.** Reported numbers follow protocol v2: they are produced on the
> Linux GPU server, on CUDA, from a committed tree. The package refuses to import on
> macOS — there is no local path to results, including for `uv run pytest`, which
> also runs on the server.

**Full documentation: [Mini-Eval documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-Eval)**

The rules that make every number comparable are in [Protocol.md](Protocol.md) (also
mirrored in the
[documentation](https://github.com/Mini-llm-lab/Documentation/blob/main/Mini-Eval/protocol.md)).

## Setup

On the server, next to a checkout of the training repository:

```
uv sync --extra native   # reference-only installs can use plain `uv sync`
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training
repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on the machine and is
ignored by Git.

## Typical run

```
uv run mini-eval list
uv run mini-eval check mini-llm-40M-35096steps --device cuda
uv run mini-eval run smoke mini-llm-40M-35096steps pythia-70m-step128 --device cuda
uv run mini-eval run small_lm --all --device cuda --require-clean
uv run mini-eval report small_lm
uv run pytest
```

`run` always rebuilds `reports/<suite>.md` plus the two plots from the latest runs,
archives the previous report with its plots under `reports/history/<suite>/<date>/`,
and writes a self-contained HTML version (`reports/<suite>.html`). Tests use fake
models and need no network (they still require Linux). Everything else
— command reference, adding a model, the native backend, results layout — lives in
the
[documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-Eval).
