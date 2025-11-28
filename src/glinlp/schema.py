from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class EntitySchema(BaseModel):
    label: str = Field(..., alias="name")
    description: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            return {"label": value}

        if isinstance(value, dict):
            if "label" in value or "name" in value:
                return value
            if len(value) == 1:
                label, config = next(iter(value.items()))
                if isinstance(config, str):
                    return {"label": label, "description": config}
                if isinstance(config, dict):
                    data = config.copy()
                    data["label"] = label
                    return data
        raise TypeError(f"Invalid entity specification: {value!r}")

    def to_gliner(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.description:
            data["description"] = self.description
        if self.threshold is not None:
            data["threshold"] = self.threshold
        return data


class ClassificationLabel(BaseModel):
    label: str = Field(..., alias="name")
    description: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            return {"label": value}
        if isinstance(value, dict):
            if "label" in value or "name" in value:
                return value
            if len(value) == 1:
                label, desc = next(iter(value.items()))
                if isinstance(desc, str):
                    return {"label": label, "description": desc}
        raise TypeError(f"Invalid classification label: {value!r}")


class ClassificationTask(BaseModel):
    name: str
    labels: List[ClassificationLabel]
    multi_label: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_confidence: bool = True
    class_act: Literal["auto", "sigmoid", "softmax"] = "auto"

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict) and "name" not in value:
            if len(value) != 1:
                raise TypeError(f"Ambiguous classification task: {value!r}")
            name, config = next(iter(value.items()))
            if isinstance(config, (list, tuple)):
                value = {"name": name, "labels": list(config)}
            elif isinstance(config, dict):
                value = {"name": name, **config}
            else:
                raise TypeError(f"Invalid classification task config: {config!r}")

        if isinstance(value, dict) and "labels" in value and isinstance(value["labels"], dict):
            data = value.copy()
            label_items = []
            for label, desc in data["labels"].items():
                entry: Dict[str, Any] = {"label": label}
                if isinstance(desc, str):
                    entry["description"] = desc
                elif isinstance(desc, dict):
                    entry.update(desc)
                label_items.append(entry)
            data["labels"] = label_items
            value = data

        return value

    def gliner_labels(self) -> Union[List[str], Dict[str, str]]:
        if any(label.description for label in self.labels):
            return {
                lbl.label: lbl.description or ""
                for lbl in self.labels
            }
        return [lbl.label for lbl in self.labels]


class FieldDType(str, Enum):
    LIST = "list"
    STR = "str"


def _parse_field_string(value: str) -> Dict[str, Any]:
    parts = value.split("::")
    name = parts[0].strip()
    dtype: Optional[str] = None
    description_parts: List[str] = []
    choices: Optional[List[str]] = None

    for part in parts[1:]:
        token = part.strip()
        if token in {"str", "list"} and dtype is None:
            dtype = token
        elif token.startswith("[") and token.endswith("]"):
            choices = [choice.strip() for choice in token[1:-1].split("|") if choice.strip()]
        else:
            description_parts.append(token)

    description = " ".join(description_parts) or None
    return {
        "name": name,
        "dtype": dtype or "list",
        "choices": choices,
        "description": description,
    }


class StructuredField(BaseModel):
    name: str
    dtype: FieldDType = FieldDType.LIST
    description: Optional[str] = None
    choices: Optional[List[str]] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            return _parse_field_string(value)
        if isinstance(value, dict):
            return value
        raise TypeError(f"Invalid structured field: {value!r}")


class StructuredSchema(BaseModel):
    name: str
    fields: List[StructuredField]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict) and "name" not in value:
            if len(value) != 1:
                raise TypeError(f"Ambiguous structured schema: {value!r}")
            name, fields = next(iter(value.items()))
            if not isinstance(fields, list):
                raise TypeError(f"Structure fields must be a list for {name}")
            return {"name": name, "fields": fields}
        return value


class NLPSchema(BaseModel):
    model_name: str = "fastino/gliner2-base-v1"
    default_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=4, gt=0)
    include_confidence: bool = True

    entities: List[EntitySchema] = Field(default_factory=list)
    classifications: List[ClassificationTask] = Field(default_factory=list)
    structures: List[StructuredSchema] = Field(default_factory=list)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "NLPSchema":
        data = load_data(path)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Failed to load schema from {path}: {exc}") from exc

    def entity_payload(self) -> Dict[str, Dict[str, Any]]:
        payload = {}
        for entity in self.entities:
            payload[entity.label] = entity.to_gliner()
        return payload


def load_data(path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(content) or {}
    return json.loads(content)
