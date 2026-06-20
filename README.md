# DeepNewsDef

Resources and tools for the **DeepNewsDef** project — a dataset and toolchain for deep-fake Romanian news detection.

---

## Modules

| Module | Description |
|---|---|
| [`synthetic_news_text_generator/`](synthetic_news_text_generator/) | Generates synthetic Romanian news articles using local Ollama models |
| [`metadata_validator/`](metadata_validator/) | Validates dataset JSON files against the `TextMetadata` / image schemas |

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Synthetic News Text Generator

→ **[Full documentation](synthetic_news_text_generator/README.md)**

```bash
# Single model, 10 articles
python -m synthetic_news_text_generator.main --count 10

# All models, 20 articles each, free disk after each batch
python synthetic_news_text_generator/run_multi_model.py --per-model 20 --delete-after
```

---

## Metadata Validator

```bash
# Single file
python -m metadata_validator.validate --type text --file path/to/metadata.json

# Entire folder
python -m metadata_validator.validate --type image --folder path/to/dataset/
```

A report is written to `validation_report.json` (override with `--report`).
