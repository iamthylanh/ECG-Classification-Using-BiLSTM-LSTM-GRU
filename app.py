import os
import json
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from flask_cors import CORS

import keras

# ==================================================
# PATCH KERAS 3
# ==================================================

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


# ==================================================
# APP
# ==================================================

app = Flask(__name__)
CORS(app)

# ==================================================
# LOAD MODELS
# ==================================================

print("=" * 60)
print("Loading ECG Models...")
print("=" * 60)

model_paths = {
    "BiLSTM": "Training/BILSTM/best_bilstm_model.keras",
    "GRU": "Training/GRU/best_gru_model.keras",
    "LSTM": "Training/LSTM/best_lstm_model.keras"
}

models = {}

for name, path in model_paths.items():

    try:

        if not os.path.exists(path):
            print(f"❌ Không tìm thấy: {path}")
            continue

        model = keras.models.load_model(
            path,
            compile=False
        )

        models[name] = model

        print(f"✅ {name} loaded")

    except Exception as e:

        print(f"❌ {name}")
        print(e)

print("=" * 60)
print(f"Loaded {len(models)} models")
print("=" * 60)


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# PREDICT ECG FILE
# ==================================================

@app.route("/predict_file", methods=["POST"])
def predict_file():

    file = request.files.get("file")

    if file is None:
        return jsonify({
            "status": "error",
            "message": "Chưa chọn file CSV"
        }), 400

    try:

        signal = (
            pd.read_csv(
                file,
                header=None
            )
            .values
            .flatten()
            .astype(float)
        )

        if len(signal) != 140:

            return jsonify({
                "status": "error",
                "message":
                f"Cần đúng 140 điểm ECG. Hiện tại: {len(signal)}"
            }), 400

        x = signal.reshape(1, 140, 1)

        predictions = {}

        for model_name, model in models.items():

            prob = float(
                model.predict(
                    x,
                    verbose=0
                )[0][0]
            )

            anomaly = prob > 0.5

            predictions[model_name] = {

                "label":
                    "Bất thường"
                    if anomaly
                    else "Bình thường",

                "score": round(
                    prob if anomaly else 1 - prob,
                    4
                )
            }

        return jsonify({

            "status": "success",

            "signal_data":
                signal.tolist(),

            "predictions":
                predictions
        })

    except Exception as e:

        return jsonify({

            "status": "error",
            "message": str(e)

        }), 500


# ==================================================
# EVALUATION RESULTS
# ==================================================

@app.route("/api/evaluation")
def get_evaluation():

    try:

        with open(
            "evaluation_results.json",
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return jsonify({

            "status": "success",
            "data": data

        })

    except Exception as e:

        return jsonify({

            "status": "error",
            "message": str(e)

        }), 500


# ==================================================
# API STATUS
# ==================================================

@app.route("/api/status")
def api_status():

    return jsonify({

        "status": "running",

        "loaded_models":
            list(models.keys()),

        "model_count":
            len(models)

    })


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health")
def health():

    return jsonify({

        "message":
            "ECG API Running",

        "models":
            list(models.keys())

    })


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )