from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml

from glinlp.schema import FieldDType, NLPSchema


def write_schema(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_schema_handles_multiple_input_shortcuts(tmp_path):
    """
    Mini tutorial:
    - entities accept bare strings and dicts with descriptions
    - classifications can be expressed as {task: [labels]}
    - structured fields use more explicit dictionaries
    """
    schema_path = write_schema(
        tmp_path,
        """
        model_name: fastino/gliner2-base-v1
        entities:
          - person
          - organization: "Named organizations"
          - product:
              description: "Merchandise"
              threshold: 0.7
        classifications:
          - sentiment:
              labels: [positive, neutral, negative]
          - topic:
              labels:
                finance: "Financial content"
                science: "Scientific content"
        structures:
          - recap:
              - name: headline
                dtype: str
                description: Main headline
              - name: highlights
                dtype: list
                description: Important bullets
        """,
    )

    schema = NLPSchema.from_path(schema_path)

    assert [ent.label for ent in schema.entities] == ["person", "organization", "product"]
    assert schema.entities[2].threshold == 0.7

    assert schema.classifications[0].name == "sentiment"
    assert [label.label for label in schema.classifications[1].labels] == ["finance", "science"]

    recap_fields = schema.structures[0].fields
    assert recap_fields[0].dtype == FieldDType.STR
    assert recap_fields[1].dtype == FieldDType.LIST


def test_json_schema_export_contains_helpful_metadata(tmp_path):
    schema_path = write_schema(
        tmp_path,
        """
        entities: [person]
        """,
    )

    schema = NLPSchema.from_path(schema_path)
    json_schema = schema.model_json_schema()

    # Document that JSON Schema exposes field definitions for editor support.
    assert "entities" in json_schema["properties"]
    entity_ref = json_schema["properties"]["entities"]["items"]["$ref"]
    assert "EntitySchema" in entity_ref


def test_schema_can_be_defined_in_json(tmp_path):
    data = {
        "model_name": "fastino/gliner2-base-v1",
        "entities": ["person"],
        "classifications": [
            {"sentiment": {"labels": ["positive", "negative"]}},
        ],
        "structures": [
            {
                "summary": [
                    {"name": "title", "dtype": "str"},
                    {"name": "bullets", "dtype": "list"},
                ]
            }
        ],
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    schema = NLPSchema.from_path(path)
    assert schema.classifications[0].labels[0].label == "positive"
    assert schema.structures[0].fields[0].dtype == FieldDType.STR


def test_json_schema_includes_enums_and_can_be_saved(tmp_path):
    schema_path = write_schema(tmp_path, "entities: [product]\n")
    schema = NLPSchema.from_path(schema_path)

    json_schema = schema.model_json_schema()
    field_enum = json_schema["$defs"]["FieldDType"]["enum"]
    assert {"list", "str"} == set(field_enum)

    schema_store = tmp_path / "vscode-schema.json"
    schema_store.write_text(json.dumps(json_schema, indent=2), encoding="utf-8")
    assert schema_store.stat().st_size > 100


def test_schema_supports_multilabel_classification(tmp_path):
    schema_path = write_schema(
        tmp_path,
        """
        include_confidence: true
        classifications:
          - tags:
              labels:
                feature: "Feature request"
                bug: "Bug report"
              multi_label: true
        """,
    )

    schema = NLPSchema.from_path(schema_path)
    task = schema.classifications[0]

    assert task.multi_label is True
    assert schema.include_confidence is True
    assert task.labels[0].label == "feature"
