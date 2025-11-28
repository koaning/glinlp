from __future__ import annotations

from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .nlp import NLP


class TextPayload(BaseModel):
    text: str


class PipePayload(BaseModel):
    texts: List[str]


def create_app(nlp: NLP) -> FastAPI:
    app = FastAPI(
        title="glinlp",
        description="Minimal server for GLiNER2",
        version="0.1.0",
    )

    @app.get("/")
    @app.get("health")
    @app.get("healthz")
    def health():
        return {"status": "alive"}

    @app.get("/schema")
    def get_schema():
        return nlp.schema.model_json_schema()

    @app.post("/extract")
    def extract(payload: TextPayload):
        doc = nlp(payload.text)
        return doc.to_payload()

    @app.post("/pipe")
    def pipe(payload: PipePayload):
        return [doc.to_payload() for doc in nlp.pipe(payload.texts)]

    return app


def serve_nlp(nlp: NLP, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start a uvicorn server backed by the provided NLP wrapper."""
    app = create_app(nlp)
    uvicorn.run(app, host=host, port=port)
