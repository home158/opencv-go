import cv2
import numpy as np

# 假設已經有一張二值化的圖片
image = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(image, (50, 50), 20, 255, -1)  # 在影像中繪製白色圓

# 指定中心點與半徑
center = (50, 50)  # 圓心座標 (x, y)
radius = 10        # 半徑

# 建立圓形遮罩
mask = np.zeros_like(image, dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# 使用遮罩提取感興趣區域
region = cv2.bitwise_and(image, image, mask=mask)

# 計算圓形區域內的平均值
mean_value = cv2.mean(image, mask=mask)[0]  # 只取第一個通道的均值

# 顯示結果
print("Circle Area Average Value:", mean_value)

# 顯示影像與遮罩
cv2.imshow("Binary Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Region of Interest", region)
cv2.waitKey(0)
cv2.destroyAllWindows()