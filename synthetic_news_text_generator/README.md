# Synthetic News Text Generator

Generates diverse, synthetic Romanian-language news articles for training **deep-fake text detection** models.  
Each article is produced by a local Ollama LLM and saved with full `TextMetadata` annotations.

---

## Table of Contents

- [Requirements](#requirements)
- [Architecture](#architecture)
- [Configuration](#configuration)
  - [Models & Quantizations](#models--quantizations)
  - [Generation parameters](#generation-parameters)
  - [Topics, personas, and style axes](#topics-personas-and-style-axes)
  - [Wikipedia seed](#wikipedia-seed)
  - [Output directory](#output-directory)
- [Generation types](#generation-types)
- [Running — single model (`main.py`)](#running--single-model-mainpy)
- [Running — multi-model (`run_multi_model.py`)](#running--multi-model-run_multi_modelpy)
  - [All flags](#all-flags)
  - [Examples](#examples)
  - [`--delete-after` behaviour](#--delete-after-behaviour)
- [Output format](#output-format)

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- At least one model pulled — e.g. `ollama pull llama3.2:3b`

> No `OPENAI_API_KEY` is needed. Scripts default to a dummy key for the Ollama-compatible `/v1` endpoint.

---

## Architecture

```
synthetic_news_text_generator/
├── config.yaml          # Models, topics, personas, and all generation settings
├── config_parser.py     # Pydantic config loader
├── wiki_scraper.py      # Fetches random Romanian Wikipedia extracts (factual seeds)
├── prompt_engine.py     # Builds Romanian prompts for 3 generation types
├── llm_interface.py     # OpenAI-compatible client → Ollama /v1 endpoint
├── pipeline.py          # Post-processes LLM output → validated TextMetadata
├── main.py              # Single-model CLI
└── run_multi_model.py   # Multi-model CLI (parallelism, retries, --delete-after)
```

---

## Configuration

All settings are in [`config.yaml`](config.yaml).

### Models & Quantizations

Models are listed under `llms:`. Each entry merges from the shared `&default_params` anchor and a quantization anchor, overriding only what differs:

```yaml
# Minimal — inherits all defaults
- <<: [*q4, *default_params]
  name: "llama3.2:3b"

# With overrides
- <<: [*q8, *default_params]
  name: "mistral:7b-instruct-q8_0"
  temperature_range: [0.2, 0.5, 0.7, 1.0]
  max_tokens: 2048
```

Three quantization anchors are pre-defined:

| Anchor | `quantization` field | Typical VRAM |
|---|---|---|
| `*q4` | `q4_K_M` | lowest |
| `*q8` | `q8_0` | ~2× q4 |
| `*fp16` | `fp16` | ~4× q4 |

**Included architectures** (25 entries total):

| Architecture | q4 | q8 | fp16 |
|---|:---:|:---:|:---:|
| gemma4:e4b (Google) | ✓ | — | — |
| llama3.2:3b (Meta) | ✓ | ✓ | ✓ |
| llama3.1:8b (Meta) | ✓ | ✓ | ✓ |
| mistral:7b (Mistral AI) | ✓ | ✓ | ✓ |
| qwen2.5:7b (Alibaba) | ✓ | ✓ | ✓ |
| phi4:14b (Microsoft) | ✓ | ✓ | — |
| deepseek-r1:7b (DeepSeek) | ✓ | ✓ | — |
| aya-expanse:8b (Cohere) | ✓ | ✓ | ✓ |
| gemma3:4b (Google) | ✓ | ✓ | ✓ |
| mistral-nemo:12b (Mistral+NVIDIA) | ✓ | ✓ | — |
| solar-pro:22b (Upstage) | ✓ | — | — |

> Verify exact Ollama tags at https://ollama.com/library/\<model\> before pulling.

To disable a model, remove or comment out its entry. To add a new one, copy any existing block and change `name:`.

### Generation parameters

The `&default_params` anchor defines the shared sampling defaults:

```yaml
_default_params: &default_params
  endpoint: "http://localhost:11434/v1"
  quantization: "q4_K_M"
  temperature_range: [0.2, 0.5, 0.7, 0.9, 1.0]   # sampled randomly per article
  top_p_range: [0.85, 0.9, 0.95, 1.0]
  frequency_penalty_range: [0.0, 0.1, 0.3]
  presence_penalty_range: [0.0, 0.1, 0.3]
  max_tokens: 2000
```

Each value in a `*_range` list is a candidate — one is drawn at random for every article, maximising sampling diversity in the dataset.

### Topics, personas, and style axes

The following axes are randomly combined for each article:

| Key | Examples |
|---|---|
| `topics` | 19 domains (Politică, Economie, Sănătate, Sport…) each with 3–4 sub-domains |
| `personas` | jurnalist de investigație, redactor de tabloid, influencer, bot agregator… |
| `target_countries` | România, Republica Moldova |
| `viewpoints` | pozitivă, neutră, negativă |
| `lengths` | scurtă, medie, lungă |
| `platforms` | știri normale, portal web, Telegram, Reddit, TikTok, Facebook |
| `grammar_levels` | impecabilă → slabă |
| `manipulation_levels` | obiectivă → propagandă agresivă |
| `propaganda_techniques` | Whataboutism, Apelul la frică, Cherry-picking… |
| `target_audiences` | Public general, Pensionari, Tineri / Gen Z, Părinți… |
| `urgency_levels` | Știre de rutină, Breaking News, Analiză aprofundată… |
| `formatting_styles` | Standard, Clickbait agresiv, Academic, Emotiv… |

### Wikipedia seed

```yaml
use_wikipedia_seed: "random"   # true | false | "random" (50/50 per article)
```

When enabled, a random extract from Romanian Wikipedia (`ro.wikipedia.org`) is fetched and used as a factual seed, enabling the `based_on_real_article` generation type.

### Output directory

```yaml
output_directory: "../generated_dataset/"   # relative to config.yaml
```

---

## Generation types

One type is picked randomly per article (Wikipedia-based type only available when `use_wikipedia_seed` is enabled):

| Type | Prompt strategy |
|---|---|
| `complete` | Fully synthetic — persona + topic + all style axes |
| `based_on_topic` | Same as `complete` but with a topic-driven narrative framing |
| `based_on_real_article` | Romanian Wikipedia extract injected as a factual seed; the model writes a brand-new article inspired by it |

---

## Running — single model (`main.py`)

Uses a **randomly selected** model from `config.yaml` for every article.

```bash
# From the project root
python -m synthetic_news_text_generator.main --count 10

# Custom config
python -m synthetic_news_text_generator.main --config path/to/config.yaml --count 50
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to the YAML configuration file |
| `--count` | `1` | Number of articles to generate |

---

## Running — multi-model (`run_multi_model.py`)

Distributes generation across **all** (or a chosen subset of) models in `config.yaml`.

```bash
python synthetic_news_text_generator/run_multi_model.py --count 100
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--count N` | — | Total articles distributed round-robin across models *(mutually exclusive with `--per-model`)* |
| `--per-model N` | — | N articles per model *(mutually exclusive with `--count`)* |
| `--models M [M …]` | all | Restrict to these model names (must match `name:` in config) |
| `--workers N` | `1` | Parallel worker threads (`1` = sequential) |
| `--retries N` | `2` | Retry attempts per failed job (exponential back-off: 2 s, 4 s, …) |
| `--output-dir PATH` | from config | Override the output directory |
| `--delete-after` | off | **Sequential only** — `ollama rm <model>` after each model's batch |
| `--dry-run` | off | Print the generation plan without calling Ollama |
| `--seed N` | — | Fix the random seed for reproducibility |
| `--config PATH` | `synthetic_news_text_generator/config.yaml` | Path to config YAML |

### Examples

```bash
# Preview the plan without generating anything
python synthetic_news_text_generator/run_multi_model.py --count 50 --dry-run

# 20 articles per model, delete each from disk after its batch finishes
python synthetic_news_text_generator/run_multi_model.py --per-model 20 --delete-after

# 200 total articles, 3 parallel workers
python synthetic_news_text_generator/run_multi_model.py --count 200 --workers 3

# Only two specific models
python synthetic_news_text_generator/run_multi_model.py --count 60 \
    --models "aya-expanse:8b" "llama3.1:8b-instruct-q8_0"

# Reproducible run
python synthetic_news_text_generator/run_multi_model.py --per-model 10 --seed 42
```

### `--delete-after` behaviour

When enabled (requires `--workers 1`), all jobs for a model run to completion before `ollama rm` is called, then the next model loads:

```
[BATCH] llama3.2:3b  (20 jobs)
  Progress: 1/60 (1.7%) | ✓ 1  ✗ 0 | rate=0.12/s ETA=491s
  ...
  [DELETE] Removing llama3.2:3b from disk...
  [DELETE] ✓ llama3.2:3b removed.

[BATCH] mistral:7b  (20 jobs)
  ...
```

> ⚠️ `--delete-after` is silently ignored when `--workers > 1`.

---

## Output format

Each article produces two files:

**`<uuid>.txt`** — plain text:
```
Guvernul anunță măsuri de urgență pentru reducerea inflației

Premierul a declarat marți că pachetul de măsuri vizează...

Experții economici au avertizat că efectele se vor simți...
```

**`<uuid>.json`** — full `TextMetadata`:
```json
{
    "id": "3f2a1c...",
    "real_fake": "fake",
    "based_on": "complete",
    "title": "Guvernul anunță măsuri de urgență...",
    "generation": {
        "model": "llama3.2:3b",
        "quantization": "q4_K_M",
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,
        "max_tokens": 2000,
        "generation_type": "complete",
        "prompt": "Vei adopta rolul de..."
    },
    "annotations": {
        "word_count": 312,
        "char_count": 1843,
        "sentences": 18,
        "paragraph_count": 4,
        "lang_ro": "yes",
        "lang_ru": "no",
        "is_public_figure": "no",
        "summary": "Articol despre măsurile economice..."
    },
    "source": {
        "country": "România",
        "category": "Politică - Alegeri",
        "author": "jurnalist de investigație",
        "language": "ro",
        "from_dataset": "synthetic"
    }
}
```
