from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from gliner2 import GLiNER2
from gliner2.inference.engine import Schema as GLiNERExtractionSchema

from .schema import NLPSchema

logger = logging.getLogger(__name__)


@dataclass(init=False)
class Document:
    text: str
    classes: Dict[str, Any]
    entities: List[Dict[str, Any]]
    structures: Dict[str, Any]

    def __init__(
        self,
        text: str,
        classes: Dict[str, Any],
        entities: List[Dict[str, Any]],
        structures: Dict[str, Any],
        raw: Any | None = None,
    ):
        self.text = text
        self.classes = classes
        self.entities = entities
        self.structures = structures
        # Ignore `raw` to keep memory footprint small while staying backward compatible.

    def to_payload(self, include_text: bool = True) -> Dict[str, Any]:
        payload = {
            "classes": self.classes,
            "entities": self.entities,
            "structures": self.structures,
        }
        if include_text:
            payload["text"] = self.text
        return payload

    @property
    def predictions(self) -> Dict[str, Any]:
        return self.to_payload(include_text=False)


class NLP:
    """
    Thin wrapper that loads a GLiNER2 model from an :class:`NLPSchema`
    definition and exposes a spaCy-like API.
    """

    def __init__(self, schema: NLPSchema, extractor: GLiNER2):
        self.schema = schema
        self.extractor = extractor
        self._gliner_schema = self._build_schema()

    @classmethod
    def from_schema(cls, schema_source) -> "NLP":
        """
        Load an :class:`NLP` instance from a schema path, dict, or ``NLPSchema``.
        """
        schema = _load_schema(schema_source)
        extractor = _load_extractor(schema.model_name)
        return cls(schema=schema, extractor=extractor)

    def _build_schema(self) -> GLiNERExtractionSchema:
        schema_builder = self.extractor.create_schema()

        if self.schema.entities:
            schema_builder.entities(self.schema.entity_payload())

        for task in self.schema.classifications:
            schema_builder.classification(
                task.name,
                task.gliner_labels(),
                multi_label=task.multi_label,
                cls_threshold=task.threshold,
                class_act=task.class_act,
            )

        for structure in self.schema.structures:
            builder = schema_builder.structure(structure.name)
            for field in structure.fields:
                builder.field(
                    field.name,
                    dtype=field.dtype.value,
                    choices=field.choices,
                    description=field.description,
                    threshold=field.threshold,
                )

        return schema_builder

    def __call__(self, text: str) -> Document:
        if not text:
            return Document(
                text="",
                classes={},
                entities=[],
                structures={},
            )
        raw = self.extractor.extract(
            text,
            self._gliner_schema,
            threshold=self.schema.default_threshold,
            include_confidence=self.schema.include_confidence,
        )
        return self._build_document(text, raw)

    def pipe(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
    ) -> Iterator[Document]:
        size = batch_size or self.schema.batch_size
        iterator = iter(texts)
        while True:
            batch = list(islice(iterator, size))
            if not batch:
                break
            results = self.extractor.batch_extract(
                batch,
                self._gliner_schema,
                batch_size=size,
                threshold=self.schema.default_threshold,
                include_confidence=self.schema.include_confidence,
            )
            for text_value, result in zip(batch, results):
                yield self._build_document(text_value, result)

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        from .server import serve_nlp

        serve_nlp(self, host=host, port=port)

    def _build_document(self, text: str, raw: Dict[str, Any]) -> Document:
        classes: Dict[str, Any] = {}
        for task in self.schema.classifications:
            if task.name in raw:
                classes[task.name] = _normalize_class_result(
                    raw[task.name],
                    multi=task.multi_label,
                )

        structures: Dict[str, Any] = {}
        for structure in self.schema.structures:
            if structure.name in raw:
                structures[structure.name] = raw[structure.name]

        entities = raw.get("entities", [])

        return Document(
            text=text,
            classes=classes,
            entities=entities,
            structures=structures,
        )


def _load_schema(source) -> NLPSchema:
    if isinstance(source, NLPSchema):
        return source
    if isinstance(source, (str, Path)):
        return NLPSchema.from_path(source)
    if isinstance(source, dict):
        return NLPSchema.model_validate(source)
    raise TypeError("Unsupported schema input")


def _load_extractor(model_name: str) -> GLiNER2:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        extractor = GLiNER2.from_pretrained(model_name)

    suppressed = stdout_buffer.getvalue().strip()
    suppressed_err = stderr_buffer.getvalue().strip()
    if suppressed or suppressed_err:
        logger.debug("Suppressed GLiNER2 loader output for %s", model_name)
    return extractor


def _normalize_class_result(value: Any, multi: bool) -> Any:
    def convert(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if isinstance(item, (tuple, list)) and item:
            entry = {"label": item[0]}
            if len(item) > 1 and isinstance(item[1], (int, float)):
                entry["confidence"] = item[1]
            elif len(item) > 1:
                entry["confidence"] = item[1]
            return entry
        return {"label": item}

    if multi:
        if isinstance(value, list):
            return [convert(item) for item in value]
        return [convert(value)]
    return convert(value)
