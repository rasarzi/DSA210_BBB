from src.predictor import predict_bbb

test_sequences = [
    "YGRKKRRQRRR",
    "RQIKIWFQNRRMKWKK",
    "SSSSSSSSSS",
    "AGYLLGKINLKALAALAKKIL",
]

for seq in test_sequences:
    prob, label = predict_bbb(seq)
    print(f"{seq}\t{prob:.4f}\t{label}")
