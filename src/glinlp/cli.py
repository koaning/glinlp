from __future__ import annotations

import json
from pathlib import Path
import typer

from .nlp import NLP
from .schema import NLPSchema

app = typer.Typer(help="spaCy-style interface on top of GLiNER2")


@app.command()
def serve(
    schema: Path = typer.Argument(..., help="Path to a YAML or JSON schema"),
    host: str = typer.Option("0.0.0.0", help="Host interface to bind"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start a FastAPI service backed by the configured schema."""
    nlp = NLP.from_schema(schema)
    nlp.serve(host=host, port=port)


@app.command()
def validate(schema: Path = typer.Argument(..., help="Schema file to inspect")):
    """Print the validated schema along with its JSON Schema."""
    nlp_schema = NLPSchema.from_path(schema)
    typer.echo("Validated schema:")
    typer.echo(nlp_schema.model_dump_json(indent=2))
    typer.echo("\nJSON schema (for editors / UIs):")
    typer.echo(json.dumps(nlp_schema.model_json_schema(), indent=2))


@app.command("infer-jsonl")
def infer_jsonl(
    schema: Path = typer.Argument(..., help="Path to a YAML or JSON schema"),
    input_jsonl: Path = typer.Argument(..., help="Input JSONL file with a text field"),
    output_jsonl: Path = typer.Argument(..., help="Where to store predictions JSONL"),
    text_field: str = typer.Option("text", help="Field in each JSON object that holds text"),
):
    """
    Run batched inference over a JSONL file and emit predictions to a new JSONL file.
    Each input line must contain a JSON object with a ``text_field`` entry (default: ``text``).
    The output file mirrors the inputs but adds a ``predictions`` key.
    """
    if not input_jsonl.exists():
        raise typer.BadParameter(f"Input file {input_jsonl} does not exist.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    nlp = NLP.from_schema(schema)

    processed = 0
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_field)
            if not isinstance(text, str):
                raise typer.BadParameter(
                    f"Record missing text field '{text_field}': {line[:80]}"
                )
            doc = nlp(text)
            record["predictions"] = doc.predictions
            dst.write(json.dumps(record))
            dst.write("\n")
            processed += 1

    typer.echo(f"Wrote {processed} predictions to {output_jsonl}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
