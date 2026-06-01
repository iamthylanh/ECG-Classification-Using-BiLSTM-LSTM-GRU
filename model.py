"""
model.py

Nhiệm vụ:
- Nghiên cứu RNN/LSTM/GRU
- Xây dựng kiến trúc mô hình Deep Learning
- Cấu hình Input Layer và Output Layer

Dữ liệu đầu vào:
    Shape = (140, 1)

Ý nghĩa:
    140 time-steps ECG
    1 giá trị biên độ tại mỗi bước

Đầu ra:
    Sigmoid
    0 = Bình thường
    1 = Bất thường
----------------------------------------------------
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    Bidirectional
)

# ==================================================
# MÔ HÌNH 1: LSTM
# ==================================================

def build_lstm_model(
        input_shape=(140, 1),
        units=64,
        dropout_rate=0.3):

    model = Sequential([
        Input(shape=input_shape),

        # LSTM tầng 1
        LSTM(
            units=units,
            return_sequences=True
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        # LSTM tầng 2
        LSTM(
            units=units // 2,
            return_sequences=False
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        # Dense Layer
        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        # Output Layer
        Dense(
            1,
            activation="sigmoid"
        )
    ],
    name="LSTM_ECG_Classifier")

    return model


# ==================================================
# MÔ HÌNH 2: GRU
# ==================================================

def build_gru_model(
        input_shape=(140, 1),
        units=64,
        dropout_rate=0.3):

    model = Sequential([
        Input(shape=input_shape),

        GRU(
            units=units,
            return_sequences=True
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        GRU(
            units=units // 2,
            return_sequences=False
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ],
    name="GRU_ECG_Classifier")

    return model


# ==================================================
# MÔ HÌNH 3: BIDIRECTIONAL LSTM
# ==================================================

def build_bilstm_model(
        input_shape=(140, 1),
        units=64,
        dropout_rate=0.3):

    model = Sequential([
        Input(shape=input_shape),

        Bidirectional(
            LSTM(
                units=units,
                return_sequences=True
            )
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        Bidirectional(
            LSTM(
                units=units // 2,
                return_sequences=False
            )
        ),

        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ],
    name="BiLSTM_ECG_Classifier")

    return model


# ==================================================
# KIỂM TRA KIẾN TRÚC
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ECG CLASSIFICATION MODELS")
    print("PHẦN THỰC HIỆN: NI + QUYÊN")
    print("=" * 60)

    models = [
        build_lstm_model(),
        build_gru_model(),
        build_bilstm_model()
    ]

    for model in models:
        print("\n")
        print("=" * 60)
        print(model.name)
        print("=" * 60)
        model.summary()