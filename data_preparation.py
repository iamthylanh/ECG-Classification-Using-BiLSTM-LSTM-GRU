import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

def clean_and_prepare_raw_data(train_raw_path, test_raw_path):
    print("--- BẮT ĐẦU DỌN DẸP VÀ GOM NHÓM NHÃN ---")
    
    try:
        train_df = pd.read_csv(train_raw_path, header=None, sep=r'\s+')
        test_df = pd.read_csv(test_raw_path, header=None, sep=r'\s+')
    except Exception as e:
        print(f"Lỗi đọc file dữ liệu: {e}")
        return None, None

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Gộp dữ liệu thô để xử lý nhãn đồng bộ
    combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    raw_labels = combined_df.iloc[:, 0].values.astype(int)
    signals = combined_df.iloc[:, 1:].values

    # GOM NHÓM NHÃN: Nhãn 1 -> 0 (Bình thường) | Nhãn 2,3,4,5 -> 1 (Bất thường)
    binary_labels = np.where(raw_labels == 1, 0, 1)

    print(f" Đã dọn dẹp {combined_df.shape[0]} chuỗi tín hiệu ECG thô thành công.")
    return signals, binary_labels

def split_and_create_structure(signals, binary_labels, output_dir="dataset"):
    print("\n--- BẮT ĐẦU CHIA TẬP DỮ LIỆU VÀ CHUẨN HÓA THEO TỪNG CHUỖI SÓNG TIM ---")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    sample_dir = os.path.join(output_dir, "samples_to_test")
    os.makedirs(sample_dir, exist_ok=True)

    # BƯỚC 1: CHIA DỮ LIỆU ĐỘC LẬP (Tỷ lệ 80 - 10 - 10)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        signals, binary_labels, test_size=0.1, random_state=42, stratify=binary_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=42, stratify=y_train_val
    )

    # BƯỚC 2: CHUẨN HÓA THEO TỪNG DÒNG (Row-wise MinMax Scaling)
    def scale_row_wise(matrix):
        min_vals = matrix.min(axis=1, keepdims=True)
        max_vals = matrix.max(axis=1, keepdims=True)
        denominator = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        return (matrix - min_vals) / denominator

    X_train_scaled = scale_row_wise(X_train)
    X_val_scaled = scale_row_wise(X_val)
    X_test_scaled = scale_row_wise(X_test)

    # BƯỚC 3: CHUẨN HÓA ĐƯA VỀ DẠNG ARRAY 3D CHO RNN
    X_train_3D = np.expand_dims(X_train_scaled, axis=-1)
    X_val_3D = np.expand_dims(X_val_scaled, axis=-1)
    X_test_3D = np.expand_dims(X_test_scaled, axis=-1)

    def save_data_splits(X_data, y_data, split_name):
        np.save(os.path.join(output_dir, split_name, f"X_{split_name}.npy"), X_data)
        np.save(os.path.join(output_dir, split_name, f"y_{split_name}.npy"), y_data)
        return len(y_data)

    len_train = save_data_splits(X_train_3D, y_train, 'train')
    len_val = save_data_splits(X_val_3D, y_val, 'val')
    len_test = save_data_splits(X_test_3D, y_test, 'test')

    # BƯỚC 4: XUẤT FILE .CSV MẪU SẠCH TUYỆT ĐỐI KHÔNG LỆCH BIÊN ĐỘ 
    normal_indices = np.where(y_test == 0)[0][:3]
    abnormal_indices = np.where(y_test == 1)[0][:3]

    for i, idx in enumerate(normal_indices):
        pd.DataFrame(X_test_scaled[idx]).T.to_csv(os.path.join(sample_dir, f"mau_ecg_binh_thuong_{i+1}.csv"), index=False, header=False)
    for i, idx in enumerate(abnormal_indices):
        pd.DataFrame(X_test_scaled[idx]).T.to_csv(os.path.join(sample_dir, f"mau_ecg_bat_thuong_{i+1}.csv"), index=False, header=False)

    print(f"\n--- TÁI CẤU TRÚC PIPELINE THÀNH CÔNG (CHUẨN HÓA ĐỘC LẬP TỪNG MẪU) ---")
    print(f" 📂 'dataset/train/': {len_train} mẫu.")
    print(f" 📂 'dataset/val/':   {len_val} mẫu.")
    print(f" 📂 'dataset/test/':  {len_test} mẫu.")

if __name__ == "__main__":
    RAW_TRAIN = os.path.join("raw_data", "ECG5000_TRAIN.txt")
    RAW_TEST = os.path.join("raw_data", "ECG5000_TEST.txt")
    
    if os.path.exists(RAW_TRAIN) and os.path.exists(RAW_TEST):
        signals, labels = clean_and_prepare_raw_data(RAW_TRAIN, RAW_TEST)
        if signals is not None:
            split_and_create_structure(signals, labels, output_dir="dataset")