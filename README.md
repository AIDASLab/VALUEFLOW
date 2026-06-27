<div align="center">

<img src="docs/static/valueflow_icon.png" alt="VALUEFLOW" width="120"/>

# VALUEFLOW

**Toward Pluralistic and Steerable Value-based Alignment in Large Language Models**

*ICML 2026*

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2602.03160)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://aidaslab.github.io/VALUEFLOW/)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://TODO-huggingface-model-link)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](snu-aidas/vidb-filtered)

</div>

---

VALUEFLOW is a unified framework for **value-based alignment** that spans extraction, evaluation, and
intensity-controlled steering of LLMs. It integrates three components:

- **HIVES** — a hierarchical value embedding space capturing intra- and cross-theory value structure.
- **VIDB** (Value Intensity DataBase) — a large-scale resource of value-labeled texts with intensity
  estimates derived from ranking-based aggregation.
- **Anchor-based evaluator** — produces calibrated intensity scores by ranking model outputs against VIDB panels.

---

## 0) Environment Setup

```bash
conda create -n llm_value python==3.12
conda activate llm_value
pip install -r requirements.txt
```

---

## 1) Embedding Model (HIVES)

HIVES is a hierarchical value embedding model trained on top of `Qwen/Qwen3-Embedding-0.6B`.

**Download** — available on the Hugging Face Hub: [`snu-aidas/HiVES-2`](https://huggingface.co/snu-aidas/HiVES-2)

**Usage** — encode texts and save per-theory embeddings:

```bash
python embedding/save_embedding.py \
  --model_name_or_path <hives-model-path-or-hf-name> \
  --input_csv data/values.csv \
  --output_dir ./embeddings \
  --batch_size 64
```

The input CSV needs at least `text` and `theory` columns; one NPZ of `(text, embedding)` pairs is written per theory.

---

## 2) VIDB Download (Value Intensity DataBase)

VIDB is the reference anchor set used for evaluation and steering. Two versions are released:

- **Full set** — complete value-labeled corpus with intensity estimates. *(link: TODO)*
- **Filtered set** — a curated subset where LLM and human ratings agree: [`snu-aidas/vidb-filtered`](https://huggingface.co/datasets/snu-aidas/vidb-filtered)

Place the downloaded files under `data/` (e.g. `data/final_ratings/`) before running evaluation.

---

## 3) Text Value Evaluation

Score open-ended responses by ranking them against VIDB anchors, producing calibrated per-value
intensities in `[-10, 10]`.

**First download VIDB** (Section 2), then run:

```bash
bash scripts/evaluate_responses.sh
```

Key settings inside the script:

| Variable | Meaning |
| --- | --- |
| `VALUES` | Target values to evaluate (e.g. benevolence, power, …). |
| `INTENSITIES` | Steering intensity levels of the responses being scored (`-2, -1, 0, 1, 2`). |
| `EVAL_LLM` | Judge model used for ranking (default `google/gemma-3-27b-it`). |
| `THEORY` | Value theory: `pvq` \| `mft` \| `duty` \| `right`. |
| `K` | Texts per ranking window (1 response + `K-1` sampled VIDB anchors). |
| `M` | Number of ranking windows (comparisons) per response. |
| `ITER` | Epochs for the 1-D Plackett–Luce (PL) utility optimization. |
| `LR` | Learning rate for the PL utility ascent. |
| `SAMPLING_METHOD` | Anchor sampling strategy (`bucket` \| `random` \| `fixed`). |
| `PROMPT_FORMAT` | Ranking prompt format (`default` \| `oneshot`). |

The script loops over every value × intensity, ranks each response against VIDB panels, runs PL
optimization, and writes calibrated intensity scores to the output directory.

---

## 4) Profiling Prompts → OpinionQA Generation + Evaluation

Use demographic value profiles to steer generation on OpinionQA and evaluate behavior-prediction accuracy.

```bash
bash scripts/run_opinionqa_final.sh {qwen3|phi4|glm4}
```

Each preset runs `profling/generate_response_opinionqa.py` with a model-specific configuration:

| Preset | Model | Theory | `num_prompts` | `threshold` |
| --- | --- | --- | --- | --- |
| `qwen3` | `Qwen/Qwen3-32B` | duty | 4 | 0.65 |
| `phi4` | `microsoft/phi-4` | pvq | 6 | 0.17 |
| `glm4` | `zai-org/GLM-4-32B-0414` | duty | 4 | 0.5 |

Main arguments:

- `--profile_dir` — directory of per-group profile prompts (`profling/prompts/`, e.g. `POLPARTY_Democrat.json`).
- `--relevance_csv` — value/duty relevance scores per question (used with `--threshold`).
- `--theory` — value theory used to build the profile prompt (`pvq` \| `mft` \| `duty` \| `rights`).
- `--num_prompts` — number of profile prompts combined per query.
- `--threshold` — relevance cutoff for selecting value dimensions.

Outputs are written under `outputs/opinionqa_final/`.

---

## Data Creation

**Value categorization.** Each text is mapped onto a theory hierarchy via human–LLM collaboration.
At each level, a panel of seven LLMs votes on the best category; a label is accepted if ≥5 agree or
the leader is ahead by ≥2 votes, otherwise a *Neutral* option is re-prompted. The process descends
the chosen child node until a neutral stop or leaf is reached, and the final label is the root-to-leaf
path. Per-theory scripts live in [`categorization/`](categorization/) (`categorize_values_pvq.py`,
`categorize_values_mft.py`, `categorize_duties.py`, `categorize_rights.py`).

---

## Embedding Model Training

HIVES is trained in two stages:

- **Stage 1 — Intra-theory alignment** ([`embedding/train_stage1.py`](embedding/train_stage1.py)):
  a hierarchical contrastive loss pulls together texts that share hierarchy prefix and direction
  label, preserving the taxonomy structure within each theory.
- **Stage 2 — Inter-theory & anchor alignment** ([`embedding/train_stage2.py`](embedding/train_stage2.py)):
  an InfoNCE objective over cross-theory anchors and user-friendly value instances unifies the
  heterogeneous theories into a shared semantic space.

```bash
python embedding/train_stage1.py   # intra-theory
python embedding/train_stage2.py   # cross-theory
```

---

## Related Resources

For a curated, continually updated collection of recent work on human values and pluralistic
alignment in LLMs, check out
[**Awesome-LLM-Values-and-Pluralistic-Alignment**](https://github.com/AIDASLab/Awesome-LLM-Values-and-Pluralistic-Alignment).

---

## License

The VALUEFLOW code is released under the [MIT License](LICENSE).

VIDB is built on top of several source corpora, each of which retains its original license. When
using VIDB, please comply with the terms of the underlying dataset licenses:

| Dataset | License |
| --- | --- |
| MFRC | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Social Chemistry | Creative Commons Attribution–ShareAlike 4.0 International (CC BY-SA 4.0) |
| ValueNet | Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA) |
| ValueEval | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| ValuePrism | AI2 ImpACT License, Medium Risk Artifacts ("MR Agreement") |

Note that ValueNet (CC BY-NC-SA) restricts use to non-commercial purposes, and ValuePrism is
governed by the AI2 ImpACT Medium Risk Agreement.

## Citation

```bibtex
@inproceedings{kim2026valueflow,
  title     = {VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models},
  author    = {Kim, Woojin and Hyeon, Sieun and Oh, Jusang and Do, Jaeyoung},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
