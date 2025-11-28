from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .nlp import NLP
from .schema import NLPSchema

app = typer.Typer(
    help="spaCy-style interface on top of GLiNER2",
    no_args_is_help=True,
)
schema_app = typer.Typer(help="Schema utilities", no_args_is_help=True)
app.add_typer(schema_app, name="schema")


@app.command()
def serve(
    schema: Path = typer.Argument(..., help="Path to a YAML or JSON schema"),
    host: str = typer.Option("0.0.0.0", help="Host interface to bind"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start a GLiNER2 server."""
    nlp = NLP.from_schema(schema)
    nlp.serve(host=host, port=port)


@app.command()
def infer(
    schema: Path = typer.Argument(..., help="Path to a YAML or JSON schema"),
    input_jsonl: Path = typer.Argument(..., help="Input JSONL file with a text field"),
    output_jsonl: Path = typer.Argument(..., help="Where to store predictions JSONL"),
    text_field: str = typer.Option("text", help="Field in each JSON object that holds text"),
):
    """
    Run batched inference over a JSONL file.
    """
    if not input_jsonl.exists():
        raise typer.BadParameter(f"Input file {input_jsonl} does not exist.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    nlp = NLP.from_schema(schema)

    with input_jsonl.open("r", encoding="utf-8") as src:
        total_records = sum(1 for line in src if line.strip())
    if total_records == 0:
        typer.echo("No records found in the input file.")
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )

    processed = 0
    with progress, input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        task_id = progress.add_task("Predicting", total=total_records)
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
            progress.advance(task_id)

    typer.echo(f"Wrote {processed} predictions to {output_jsonl}")


@schema_app.command("autocomplete")
def schema_json(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save the NLPSchema JSON schema definition.",
    )
):
    """
    Create a JSON schema for IDE autocompletion.
    """
    spec = NLPSchema.model_json_schema()
    payload = json.dumps(spec, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Wrote JSON schema to {output}")
    else:
        typer.echo(payload)


@schema_app.command("new")
def schema_create(
    output: Path = typer.Argument(..., help="Where to write the new schema YAML"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite the output file if it already exists.",
    ),
):
    """
    Create a new schema file, optionally from a template, otherwise using a default skeleton.
    """
    if output.exists() and not force:
        raise typer.BadParameter(f"{output} already exists. Use --force to overwrite.")

    skeleton = {
            "model_name": "fastino/gliner2-base-v1",
            "include_confidence": True,
            "entities": ["person"],
            "classifications": [
                {"sentiment": {"labels": ["positive", "negative", "neutral"], "threshold": 0.5}}
            ],
            "structures": [
                {
                    "record": [
                        {"name": "field_name", "dtype": "str", "description": "Describe the value"}
                    ]
                }
            ],
        }
    content = yaml.safe_dump(skeleton, sort_keys=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    typer.echo(f"Schema written to {output}")

def main() -> None:
    app()


if __name__ == "__main__":
    main()
