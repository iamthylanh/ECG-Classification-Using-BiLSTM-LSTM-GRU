import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_gru_model

def main():
    # 1. Tải và tiền xử lý dữ liệu
    dataset_dir = "THDeep_Lan2/dataset"
    
    X_train = np.load(os.path.join(dataset_dir, "train", "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "train", "y_train.npy"))
    
    X_val = np.load(os.path.join(dataset_dir, "val", "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "val", "y_val.npy"))

    # 2. Khởi tạo mô hình
    # Tín hiệu đầu vào ECG có shape = (140, 1)
    model = build_gru_model(input_shape=(140, 1))

    # 3. COMPILE MODEL
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. CẤU HÌNH CALLBACKS (EARLYSTOPPING & MODELCHECKPOINT)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_gru_model.keras', 
            save_best_only=True, 
            monitor='val_loss'
        )
    ]

    # 5. TRAIN MODEL
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks
    )

    # 6. ACCURACY/LOSS (Vẽ đồ thị)
    plot_accuracy_loss(history)

def plot_accuracy_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
