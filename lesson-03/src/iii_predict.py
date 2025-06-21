import os
import numpy as np
from PIL import Image
from i_implement_PCA import PCA

# Nếu có log từ assignment trước
import logging
logger = logging.getLogger("FaceRecognition")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.hasHandlers():
    logger.addHandler(ch)

# ======== Cấu hình ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(TRAIN_DIR, "test")
IMAGE_SIZE = (64, 64)
N_COMPONENTS = 20

# ======== Hàm load dữ liệu huấn luyện ========
def load_face_dataset(base_path, image_size=(64, 64)):
    X, y = [], []
    label_map = {}
    logger.info(f"Đang load ảnh từ: {base_path}")
    
    for idx, folder in enumerate(sorted(os.listdir(base_path))):
        if folder == "test":  # Bỏ qua thư mục test
            continue
        person_path = os.path.join(base_path, folder)
        if not os.path.isdir(person_path) or not folder.startswith("person-"):
            continue

        label_map[idx] = folder
        logger.info(f"[{folder}] → Label: {idx}")

        for fname in os.listdir(person_path):
            fpath = os.path.join(person_path, fname)
            try:
                img = Image.open(fpath).convert("L")
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32).flatten() / 255.0
                X.append(img_array)
                y.append(idx)
            except Exception as e:
                logger.warning(f"Lỗi đọc ảnh {fpath}: {e}")

    logger.info(f"✅ Tổng ảnh train: {len(X)} | Kích thước mỗi ảnh: {image_size}")
    return np.array(X), np.array(y), label_map

# ======== Hàm load ảnh test ========
def load_test_data(test_path, image_size=(64, 64)):
    X_test, file_names = [], []
    for fname in sorted(os.listdir(test_path)):
        fpath = os.path.join(test_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert("L")
            img = img.resize(image_size)
            img_array = np.asarray(img, dtype=np.float32).flatten() / 255.0
            X_test.append(img_array)
            file_names.append(fname)
        except Exception as e:
            logger.warning(f"[Lỗi] {fname}: {e}")
    logger.info(f"✅ Tổng ảnh test: {len(X_test)}")
    return np.array(X_test), file_names

# ======== Hàm dự đoán k-NN với k=1 ========
def predict_nearest(X_train_pca, y_train, X_test_pca):
    y_pred = []
    for vec in X_test_pca:
        dists = np.linalg.norm(X_train_pca - vec, axis=1)
        nearest = np.argmin(dists)
        y_pred.append(y_train[nearest])
    return np.array(y_pred)

def calculate_accuracy(y_true, y_pred):
    """
    Tính toán độ chính xác (accuracy) cho các nhãn dự đoán mà không dùng sklearn.
    Args:
        y_true (np.ndarray): Nhãn thực tế.
        y_pred (np.ndarray): Nhãn dự đoán.
    Returns:
        float: Độ chính xác (accuracy).
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total if total > 0 else 0.0

# ======== MAIN ========
def main():
    # Load dữ liệu train/test
    X_train, y_train, label_map = load_face_dataset(TRAIN_DIR, IMAGE_SIZE)
    X_test, test_names = load_test_data(TEST_DIR, IMAGE_SIZE)

    # PCA
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Dự đoán nhãn cho tập test
    y_pred = predict_nearest(X_train_pca, y_train, X_test_pca)

    # Hiển thị kết quả dự đoán
    print("\n📢 Kết quả dự đoán:")
    for fname, label in zip(test_names, y_pred):
        print(f"{fname} → {label_map[label]}")

    # ====== Lấy nhãn thật từ tên file test ======
    label_map_inv = {v: k for k, v in label_map.items()}
    y_test = []
    for fname in test_names:
        true_label_str = fname.split('.')[0]  # Lấy phần trước dấu '.'
        if true_label_str in label_map_inv:
            y_test.append(label_map_inv[true_label_str])
        else:
            raise ValueError(f"Nhãn '{true_label_str}' không tồn tại trong tập train")

    y_test = np.array(y_test)

    # Tính và hiển thị accuracy
    acc = calculate_accuracy(y_test, y_pred)
    print(f"\n✅ Độ chính xác (Accuracy): {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
