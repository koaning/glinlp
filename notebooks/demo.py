# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from glinlp import NLP, NLPSchema

    schema_path = Path("examples/schema.example.yaml")
    schema = NLPSchema.from_path(schema_path)
    return NLP, schema, schema_path


@app.cell
def _(nlp):
    nlp.schema.classifications
    return


@app.cell
def _(NLP, schema_path):
    nlp = NLP.from_schema(schema_path)
    sample_text = "Apple CEO Tim Cook introduced the Vision Pro headset in Cupertino."
    doc = nlp(sample_text)
    return (nlp,)


@app.cell
def _(nlp):
    texts = [
        "Meta is reportedly acquiring a robotics startup from Berlin.",
        "Microsoft unveiled new Azure pricing tiers in Seattle.",
    ]
    batch = list(nlp.pipe(texts))
    batch
    return


@app.cell
def _(schema):
    json_schema = schema.model_json_schema()
    return


if __name__ == "__main__":
    app.run()
