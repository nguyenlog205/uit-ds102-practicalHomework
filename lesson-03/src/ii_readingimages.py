import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging 
from i_implement_PCA import PCA
# Định nghĩa đường dẫn gốc của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("FaceDatasetLoader")

logger.setLevel(logging.INFO)  # hoặc DEBUG để chi tiết hơn

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Format log
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)

# Gắn handler nếu chưa có
if not logger.hasHandlers():
    logger.addHandler(ch)



def load_face_dataset(base_path="data", image_size=(64, 64)):
    """
    Load ảnh từ thư mục person-* và chuyển thành vector dữ liệu + nhãn.
    
    Args:
        base_path (str): Thư mục chứa dữ liệu ảnh.
        image_size (tuple): Resize ảnh về kích thước đồng nhất.
    
    Returns:
        X (np.ndarray): Mảng dữ liệu ảnh dạng vector (n_samples, n_features)
        y (np.ndarray): Mảng label ứng với từng ảnh (n_samples,)
        label_map (dict): Ánh xạ label -> tên thư mục (0: person-1, ...)
    """
    X = []
    y = []
    label_map = {}

    logger.info(f"Đang load ảnh từ: {base_path}")

    for idx, person in enumerate(sorted(os.listdir(base_path))):
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            continue
        if not person.startswith("person-"):
            logger.debug(f"Bỏ qua thư mục: {person}")
            continue

        label_map[idx] = person
        logger.info(f"[{person}] → Label: {idx}")

        for fname in os.listdir(person_path):
            fpath = os.path.join(person_path, fname)
            try:
                img = Image.open(fpath).convert("L")
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32).flatten() / 255.0

                X.append(img_array)
                y.append(idx)

                logger.debug(f"Đã load ảnh: {fpath}")
            except Exception as e:
                logger.warning(f"Lỗi đọc ảnh {fpath}: {e}")

    logger.info(f"✅ Tổng số ảnh: {len(X)} | Kích thước mỗi ảnh: {image_size}")
    return np.array(X), np.array(y), label_map

def apply_pca(X, n_components=20):
    """
    Áp dụng PCA để giảm chiều dữ liệu ảnh.
    
    Args:
        X (np.ndarray): Mảng dữ liệu ảnh dạng vector (n_samples, n_features)
        n_components (int): Số thành phần chính cần giữ lại.
    
    Returns:
        X_pca (np.ndarray): Dữ liệu sau khi giảm chiều (n_samples, n_components)
        pca (PCA): Đối tượng PCA đã fit
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def visualize_pca_bases(pca, image_size=(64, 64), n_bases=20):
    """
    Hiển thị các vector cơ sở chính (eigenfaces) của không gian PCA.
    """
    expected_size = image_size[0] * image_size[1]
    if pca.components_.shape[1] != expected_size:
        raise ValueError(
            f"❌ PCA components có độ dài {pca.components_.shape[1]} "
            f"không thể reshape thành ảnh có kích thước {image_size} = {expected_size}"
        )

    plt.figure(figsize=(15, 4))
    for i in range(n_bases):
        plt.subplot(2, n_bases // 2, i + 1)
        base_img = pca.components_[i].reshape(image_size)
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
        plt.imshow(base_img, cmap='gray')
        plt.title(f"PC {i+1}")
        plt.colorbar()
        plt.axis('off')

    plt.suptitle("Các vector cơ sở của Principle Space (Eigenfaces)")
    plt.tight_layout()
    plt.show()

def main():
    # Đọc ảnh huấn luyện
    X, y, label_map = load_face_dataset(base_path=os.path.join(BASE_DIR, "data"), image_size=(64, 64))

    print("Dữ liệu shape:", X.shape)
    print("Label shape:", y.shape)
    print("Label mapping:", label_map)

    # Hiển thị một ví dụ ảnh
    if len(X) > 0:
        img_example = X[0].reshape(64, 64)
        plt.imshow(img_example, cmap='gray')
        plt.title(f"Label: {y[0]} - {label_map[y[0]]}")
        plt.axis('off')
        plt.show()
    
    # Áp dụng PCA với 20 chiều
    X_pca, pca = apply_pca(X, n_components=20)
    print("Dữ liệu sau PCA shape:", X_pca.shape)

    # Hiển thị các vector cơ sở của Principle Space
    visualize_pca_bases(pca, image_size=(64, 64), n_bases=20)

if __name__ == "__main__":
    main()
