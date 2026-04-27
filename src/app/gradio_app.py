import sys
import os
import pandas as pd
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import predict_bbb

def run_agent(input_text):
    sequences = [
        s.strip().upper()
        for s in input_text.replace(",", "\n").split("\n")
        if s.strip()
    ]

    results = []

    for seq in sequences:
        try:
            prob, label = predict_bbb(seq)
            results.append({
                "Sequence": seq,
                "BBB_Probability": round(prob, 4),
                "Prediction": label
            })
        except Exception as e:
            results.append({
                "Sequence": seq,
                "BBB_Probability": None,
                "Prediction": f"Error: {e}"
            })

    df = pd.DataFrame(results)

    if "BBB_Probability" in df.columns:
        df = df.sort_values(
            by="BBB_Probability",
            ascending=False,
            na_position="last"
        )

    return df

demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(
        lines=10,
        label="Peptide Sequences",
        placeholder="Paste peptide sequences here, one per line"
    ),
    outputs=gr.Dataframe(label="BBB Prediction Results"),
    title="BBB Peptide Classifier Agent",
    description=(
        "Paste peptide sequences. The model predicts BBB permeability probability. "
        "This is a computational screening tool, not experimental validation."
    )
)

if __name__ == "__main__":
    demo.launch()
