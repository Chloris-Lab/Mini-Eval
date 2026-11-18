# Evaluation protocol v3

These choices are fixed so that every number in this repository is comparable with the others. Anything that changes a score changes the protocol version (`PROTOCOL_VERSION` in `src/mini_eval/__init__.py`), and every model, references included, is then evaluated again.

**What changed in v3:** the suite gains `gsm8k_cot_zeroshot` (0-shot GSM8K — `Q: …` / `A: Let's think step by step.` prompt, greedy until the next `Q:` marker, `exact_match,flexible-extract`), the Mini line-up is reduced to a single native-equivalent model (`mini-llm-150M-base`, `GautierP/MiniLLM-150M` under `base/`, pinned at `6c218a1`), the legacy native entries are removed, and the training corpus is decontaminated with an n-gram filter before any published number is produced (see [Data preparation](#data-preparation)). The report is generated in two renderings — `_private.md` (all models) and `_public.md` (references plus the Petal release) — from the same runs, both committed in the repository.

## Fixed settings

| Setting   | Value                                               | Why                                                                                              |
| --------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Harness   | `lm-eval` 0.4.13, pinned in `uv.lock`               | Tasks and prompts change between versions                                                        |
| Few-shot  | 0                                                   | Small models gain little from examples, and published zero-shot numbers exist for the references |
| Precision | float32                                             | No rounding differences between models                                                           |
| Examples  | Full test sets for reported numbers (`limit: null`) | A limit is only for smoke tests                                                                  |
| Decoding  | Greedy, for the few generative uses                 | Deterministic                                                                                    |
| Device    | `cuda`, on the Linux GPU server; recorded in every run | The code refuses to run on macOS, so no local numbers can enter the results                    |
| Context   | Model-native (up to 8192)                           | v1 capped SmolLM2-135M at 2048 because the eval machine could not fit its native 8192           |
| Batch     | 4                                                   | Memory-safe at 8192 tokens; batch size must not change scores anyway (checked by `mini-eval check`) |

## Suite `small_lm`

The primary metric of each task is declared in `suites/small_lm.yaml` before any result is looked at:

| Task           | Primary metric | Chance                           |
| -------------- | -------------- | -------------------------------- |
| hellaswag      | acc_norm       | 0.25                             |
| arc_easy       | acc_norm       | 0.25                             |
| arc_challenge  | acc_norm       | 0.25                             |
| piqa           | acc_norm       | 0.50                             |
| winogrande     | acc            | 0.50                             |
| openbookqa     | acc_norm       | 0.25                             |
| sciq           | acc_norm       | 0.25                             |
| lambada_openai | acc            | 0                                |
| wikitext       | bits_per_byte  | not applicable (lower is better) |
| gsm8k_cot_zeroshot | exact_match,flexible-extract | 0.0                           |

`bits_per_byte` does not depend on the tokenizer, so it is the only language-modeling number that can be compared between models with different vocabularies. Perplexity per token must never be compared across tokenizers.

## Models

- **References** are Hugging Face models pinned to a revision. Pythia publishes intermediate checkpoints (`step64`, `step128`, ...) at 2,097,152 tokens per step, which allows comparisons at an equal number of training tokens; v3 keeps `step143000` for both Pythia-70M and Pythia-160M.
- `google/gemma-3-270m` is the **base** checkpoint, not the instruction-tuned `-it`
  variant that Ollama ships as `gemma3:270m`: this suite scores base models, so an IT
  model would be measured on a chat behaviour it was never given here. It is gated, so
  the Hugging Face account running the eval must have accepted the Gemma license. Its
  native context is 32K; `models.yaml` caps it at 8192, the ceiling the suite batch
  size of 4 is memory-safe at.
- **Native models** are identified by the SHA-256 of the checkpoint and of `tok.json`,
  stored in `models.yaml`. A run is refused if the files on disk do not match.
- `mini-llm-150M-base` evaluates the repository's own 150M from the Hugging Face Hub (`GautierP/MiniLLM-150M`, `base/` subfolder, pinned at `6c218a122ce1a410bbe3cf5afe4b8504ae6e8684`, `trust_remote_code: true`) — legacy 1.0 model, kept for reference. The mini models of this protocol are the native **`petal-1-150M`** (base, `digit32k/150M/petal-1-150M-base-53375steps.pt`) and **`petal-1-150M-assistant`** (SFT-final, `digit32k/150M/petal-1-150M.pt`), each pinned by the SHA-256 of the checkpoint and of `digit32k/tok.json` in `models.yaml` — the base model for comparison with the references, the assistant for the artifact actually shipped. The native legacy Mini checkpoints are not evaluated anymore.
- Every model runs at its **native context length**, or at 8192 tokens when its native context is wider (`gemma-3-270m`); the value is recorded in the run metadata.
- Size is described by **non-embedding parameters** as well as total parameters, because embeddings are a very different share of the total from one model to another.

The registry entries carry a `group`: `references` (the HF line-up) or `mini` (the lab's models), plus an optional `public: true` flag. `report small_lm` writes two renderings of the same runs, committed in the repository and archiving together on rerun:

- `reports/small_lm_private.md` (+ `_vs_tokens.png`, `_vs_params.png`, `.html`) — all registered models;
- `reports/small_lm_public.md` (+ same suffixes) — references plus Petal (`public: true`; mini-llm, its internal alpha, is excluded).

## Reading results

- Every score comes with its standard error. A difference smaller than two standard errors is not a result.
- Two views are reported: score against tokens seen, and score against non-embedding parameters. Comparing a model trained on 0.2B tokens with a reference trained on 300B tokens says more about the data than about the model.
- A report warns when the runs in a table differ in harness version, protocol version, few-shot count, example limit or device, or come from a modified working tree.

## Data preparation

The training corpus (`prep` in the training repository) and the SFT mix are
decontaminated against this suite before any model used for it is trained:

- Normalization: lower case, punctuation folded to spaces, whitespace collapsed.
- Every document or supervised example is checked against the **13-gram exact** sets of
  each test set above; a document sharing even one 13-gram is dropped. The same filter
  runs on tokenizer training data and on SFT examples (question side, including the
  multi-turn conversations).
- The number of skipped documents is recorded in `data/meta.json` (`skipped_contaminated_docs`).
- GSM8K `train` examples are filtered against `test` near-duplicates the same way.

Auditing the corpus against the benchmark sets a second time, after the training run, is
not possible for a multi-hundred-billion-token ingest without saving a copy; the guarantee
comes from the filter at ingestion, not from a post-hoc scan.

## Before trusting a number

1. **Reproduce a published result** with the Hugging Face path (for example Pythia-70M on HellaSwag) and compare it with the values in the model card or paper. A mismatch points to a configuration error.
2. Run `mini-eval check MODEL --device cuda` on every native model: batch size must not change scores, cuda must agree with cpu within tolerance, and an untrained model must stay near chance.
3. Run reported numbers with `--require-clean`, from a committed working tree, on the server.

## Known gaps

- Contamination is mitigated at ingestion time (13-gram filter on the pretraining corpus and on the SFT mix, see [Data preparation](#data-preparation)). A separate whole-corpus post-hoc audit of the train/test overlap for the v3 suite has not been published; the guarantee comes from the drop-at-ingest rule.
- The assistant (chat) evaluation is not part of this protocol yet. The adapter implements generation and the training prompt format, but no generative suite is defined.
- Benchmark datasets are downloaded from the Hugging Face Hub at run time and are not pinned beyond the `lm-eval` task version.
