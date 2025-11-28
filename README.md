## glinlp

Minimal spaCy-inspired wrapper around [GLiNER2](https://github.com/fastino-ai/GLiNER2).
It packages a schema-first workflow in Python, notebooks, or FastAPI deployments.

### Quick start

```bash
uv sync
glinlp serve examples/schema.example.yaml
```

### Notebook usage

```python
from glinlp import NLP

nlp = NLP.from_schema("examples/schema.example.yaml")

doc = nlp("Apple CEO Tim Cook announced the Vision Pro launch.")
print(doc.text)
print(doc.classes["sentiment"])
print(doc.entities)
print(doc.structures)

for result in nlp.pipe(["text 1", "text 2"]):
    print(result.classes)
```

### CLI helpers

```bash
glinlp validate examples/schema.example.yaml
glinlp infer-jsonl examples/schema.example.yaml data/input.jsonl data/output.jsonl
```

`infer-jsonl` reads one document per line and writes:

```json
{
  "text": "...",
  "predictions": {
    "classes": {"sentiment": {"label": "positive", "confidence": 0.92}},
    "entities": [{"person": ["Tim Cook"], "company": ["Apple"]}],
    "structures": {"announcement": [{"subject": "Vision Pro", "date": "yesterday"}]}
  }
}
```

so downstream jobs can cleanly access classes vs entities vs structured chunks.

### Schema Example

```yaml
model_name: fastino/gliner2-base-v1
batch_size: 2
include_confidence: true
entities:
  - person
  - company
  - location: "Countries, cities or regions"
classifications:
  - name: sentiment
    labels:
      - label: positive
      - label: neutral
      - label: negative
    threshold: 0.55
structures:
  - product:
      - name: name
        dtype: str
        description: Product name
      - name: price
        dtype: str
        description: Currency and amount
      - name: features
        dtype: list
        description: Bulleted list of features
```

### Marimo notebook

There is an interactive marimo notebook in `notebooks/demo.py`. Launch it with:

```bash
uv run marimo run notebooks/demo.py
```
