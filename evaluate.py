import os
import json
import numpy as np
import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================
# Patch lỗi Keras
# ==========================
original_bn = keras.layers.BatchNormalization.from_config

@classmethod
def bn_patch(cls, config):
    config.pop("renorm", None)
    config.pop("renorm_clipping", None)
    config.pop("renorm_momentum", None)
    return original_bn(config)

keras.layers.BatchNormalization.from_config = bn_patch


original_dense = keras.layers.Dense.from_config

@classmethod
def dense_patch(cls, config):
    config.pop("quantization_config", None)
    return original_dense(config)

keras.layers.Dense.from_config = dense_patch


# ==========================
# Load dữ liệu test
# ==========================
X_test = np.load("dataset/test/X_test.npy")
y_test = np.load("dataset/test/y_test.npy")

models = {
    "BiLSTM": "Training/BILSTM/best_bilstm_model.keras",
    "GRU": "Training/GRU/best_gru_model.keras",
    "LSTM": "Training/LSTM/best_lstm_model.keras"
}

results = {}

for name, path in models.items():

    if not os.path.exists(path):
        print(f"Không tìm thấy {path}")
        continue

    try:
        model = keras.models.load_model(
            path,
            compile=False
        )

        y_pred = (
            model.predict(X_test, verbose=0) > 0.5
        ).astype(int).flatten()

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "CM": confusion_matrix(y_test, y_pred).tolist()
        }

        print(f"✅ {name} OK")

    except Exception as e:
        print(f"❌ {name}: {e}")

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Đã lưu evaluation_results.json")