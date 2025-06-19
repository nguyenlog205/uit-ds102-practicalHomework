
from implement_GMM import GMM

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load ảnh màu
img = cv2.imread(r'C:\Users\VICTUS\Documents\developer\uit-practicalLesson-sML\lesson-05\cow.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Chuyển ảnh từ (H, W, 3) → (H*W, 3)
h, w, c = img_rgb.shape
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
x_coords = x_coords.reshape(-1, 1) / w   # chuẩn hóa
y_coords = y_coords.reshape(-1, 1) / h

rgb = img_rgb.reshape(-1, 3).astype(np.float32)
pixels = np.hstack((rgb, x_coords, y_coords))

gmm = GMM(n_components=2, max_iter=100000)  # 2 cụm: foreground và background
gmm.fit(pixels)
labels = gmm.predict(pixels)

# Đếm số pixel thuộc mỗi cụm
counts = np.bincount(labels)
bg_label = np.argmax(counts)  # Label của background là cụm lớn hơn

# Mask: 0 = background, 255 = foreground
mask = (labels != bg_label).astype(np.uint8) * 255
mask = mask.reshape(h, w)

foreground = img_rgb.copy()
foreground[mask == 0] = [0, 0, 0]  # Xoá background

# Hiển thị
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Mask (foreground)")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Removed background")
plt.imshow(foreground)
plt.axis('off')

plt.tight_layout()
plt.show()
