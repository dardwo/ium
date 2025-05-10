import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

test_df = pd.read_csv("data/test_data.csv")
feature_cols = [col for col in test_df.columns if col != "cardio"]
X_test = test_df[feature_cols].values
y_test = test_df["cardio"].values

model = tf.keras.models.load_model("data/cardio_model_tf.h5")

y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "prob": y_pred_probs
})
pred_df.to_csv("data/predictions.csv", index=False)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

metrics_path = "data/metrics.csv"
new_row = pd.DataFrame([{
    "build": os.getenv("BUILD_NUMBER", "1"),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

if os.path.exists(metrics_path):
    old_metrics = pd.read_csv(metrics_path)
    metrics_df = pd.concat([old_metrics, new_row], ignore_index=True)
else:
    metrics_df = new_row

metrics_df.to_csv(metrics_path, index=False)

plt.figure(figsize=(10, 6))
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    plt.plot(metrics_df["build"], metrics_df[metric], marker='o', label=metric)

plt.title("Evaluation Metrics over Builds")
plt.xlabel("Build Number")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/plot.png")
plt.close()
