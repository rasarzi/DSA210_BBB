"""Local peptide screening interface. Requires an authorized trained model artifact."""

import re

import gradio as gr
import pandas as pd

from src.optimizer import scan_single_mutations
from src.predictor import get_predictor


def _sequences(text):
    sequences = [item.strip() for item in re.split(r"[,;\n]+", text or "") if item.strip()]
    if not sequences:
        raise gr.Error("Enter at least one peptide sequence.")
    if len(sequences) > 500:
        raise gr.Error("Limit batch predictions to 500 peptides.")
    return sequences


def predict_batch(text):
    try:
        predictions = get_predictor().predict_many(_sequences(text))
        return pd.DataFrame(predictions).sort_values("bbb_score", ascending=False)
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


def scan_mutations(sequence, top_n):
    try:
        result = scan_single_mutations(sequence, get_predictor(), limit=int(top_n))
        return result["original_score"], pd.DataFrame(result["top_variants"])
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="BBB Peptide Screening") as demo:
    gr.Markdown("# BBB Peptide Screening\nSequence-based computational prioritization; **not measured permeability or clinical guidance**. Model scores may be uncalibrated.")
    with gr.Tab("Batch prediction"):
        raw = gr.Textbox(lines=8, label="Peptide sequences", placeholder="DSSHAFTLDELR\nAGYLLGKINLKALAALAKKIL")
        run = gr.Button("Predict and rank")
        ranked = gr.Dataframe(label="Predictions (highest score first)")
        run.click(predict_batch, inputs=raw, outputs=ranked)
    with gr.Tab("Single-residue scan"):
        peptide = gr.Textbox(label="Parent peptide", placeholder="DSSHAFTLDELR")
        count = gr.Slider(1, 100, value=20, step=1, label="Top substitutions")
        scan = gr.Button("Scan single substitutions")
        baseline = gr.Number(label="Parent model score")
        variants = gr.Dataframe(label="Candidate substitutions ranked by score change")
        scan.click(scan_mutations, inputs=[peptide, count], outputs=[baseline, variants])
    gr.Markdown("A positive model score is **not evidence of BBB transport**. Mutations are computational hypotheses and can affect stability, toxicity, and synthesis.")


if __name__ == "__main__":
    demo.launch()
