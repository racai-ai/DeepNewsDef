# DeepNewsDef
Resources for the DeepNewsDef project

## Metadata Validator

Validate dataset JSON metadata files against text or image schemas.

```bash
# Single file
python -m metadata_validator.validate --type text --file path/to/metadata.json

# Entire folder
python -m metadata_validator.validate --type image --folder path/to/dataset/
```

A detailed report is written to `validation_report.json` (override with `--report`).
