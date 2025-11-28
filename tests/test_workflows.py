from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml
from typer.testing import CliRunner

from glinlp.cli import app
from glinlp.nlp import NLP
from glinlp.schema import NLPSchema


def _write_schema(path: Path) -> Path:
    schema = {
        "model_name": "fastino/gliner2-base-v1",
        "entities": ["person", {"company": "Company names"}],
        "classifications": [
            {"sentiment": {"labels": ["positive", "negative"]}},
            {"intent": {"labels": ["sales", "support"], "multi_label": True}},
        ],
        "structures": [
            {
                "deal": [
                    {"name": "company", "dtype": "str", "description": "Company referenced"},
                    {"name": "value", "dtype": "str", "description": "Transaction size"},
                ]
            }
        ],
    }
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    return path


class DummyStructureBuilder:
    def __init__(self, schema: "DummySchema", name: str):
        self.schema = schema
        self.name = name

    def field(self, name: str, dtype: str = "list", **meta):
        self.schema.structures.setdefault(self.name, []).append(
            {"name": name, "dtype": dtype, **meta}
        )
        return self


class DummySchema:
    def __init__(self):
        self.entities_config: Dict[str, Dict] = {}
        self.classifications: List[Dict] = []
        self.structures: Dict[str, List[Dict]] = {}

    def entities(self, payload: Dict[str, Dict]):
        self.entities_config = payload
        return self

    def classification(self, name: str, labels, **cfg):
        self.classifications.append({"name": name, "labels": labels, **cfg})
        return self

    def structure(self, name: str):
        return DummyStructureBuilder(self, name)


class TinyExtractor:
    """
    Lightweight stand-in for GLiNER2 used strictly for local tests.
    Keeps the tests fast while exercising the NLP orchestration code.
    """

    def create_schema(self):
        return DummySchema()

    def _fake_prediction(self, text: str) -> Dict:
        return {
            "text": text,
            "entities": [
                {
                    "person": ["Sam"],
                    "company": ["ACME Corp"],
                }
            ],
            "sentiment": ("positive", 0.92),
            "intent": [("sales", 0.81)],
            "deal": [
                {
                    "company": "ACME Corp",
                    "value": "$100M",
                }
            ],
        }

    def extract(self, text, schema, **kwargs):
        return self._fake_prediction(text)

    def batch_extract(self, texts, schema, **kwargs):
        return [self._fake_prediction(text) for text in texts]


def test_nlp_wrapper_behaves_like_spacy_pipe(tmp_path):
    """
    Tutorial:
        schema = NLPSchema.from_path("schema.yaml")
        nlp = NLP(schema, extractor)
        doc = nlp("text")
        docs = list(nlp.pipe([...]))
    """
    schema_path = _write_schema(tmp_path / "schema.yaml")
    schema = NLPSchema.from_path(schema_path)

    extractor = TinyExtractor()
    nlp = NLP(schema=schema, extractor=extractor)

    doc = nlp("Apple acquired Beats.")
    assert doc.text == "Apple acquired Beats."
    assert doc.classes["sentiment"]["label"] == "positive"

    texts = ["Deal one", "Deal two"]
    docs = list(nlp.pipe(texts))
    assert [d.text for d in docs] == texts
    assert "deal" in docs[0].structures


def test_cli_validate_schema_outputs_json(tmp_path):
    """
    CLI mini tutorial:
        $ glinlp validate schema.yaml
    """
    schema_path = _write_schema(tmp_path / "schema.yaml")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(schema_path)])
    assert result.exit_code == 0
    assert "Validated schema:" in result.stdout
    assert "JSON schema" in result.stdout


def test_cli_infer_jsonl_appends_predictions(tmp_path, monkeypatch):
    schema_path = _write_schema(tmp_path / "schema.yaml")
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    records = [
        {"id": 1, "text": "Apple acquired Beats."},
        {"id": 2, "text": "Google launched a new phone."},
    ]
    input_file.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    def _from_schema(cls, path):
        schema = NLPSchema.from_path(path)
        return NLP(schema=schema, extractor=TinyExtractor())

    monkeypatch.setattr(NLP, "from_schema", classmethod(_from_schema))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "infer-jsonl",
            str(schema_path),
            str(input_file),
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    lines = [
        json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    preds = lines[0]["predictions"]
    assert set(preds.keys()) == {"classes", "entities", "structures"}
    assert preds["classes"]["sentiment"]["label"] == "positive"
    assert "deal" in preds["structures"]
