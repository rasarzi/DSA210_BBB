"""Example CLI inference. A local trusted model artifact is required."""

from src.predictor import get_predictor

if __name__ == "__main__":
    sequences = ["YGRKKRRQRRR", "SSSSSSSSSS", "AGYLLGKINLKALAALAKKIL"]
    for row in get_predictor().predict_many(sequences):
        print(f"{row['sequence']}\t{row['bbb_score']:.4f}\t{row['prediction']}")
