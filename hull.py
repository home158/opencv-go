import cv2
import numpy as np

# 創建測試圖像
image = np.zeros((400, 400), dtype=np.uint8)
points = np.array([[50, 300], [200, 50], [350, 300], [150, 150], [250, 150]])
cv2.drawContours(image, [points], -1, 255, -1)
# 顯示結果
cv2.imshow("draw Contours", image)
cv2.waitKey(0)


# 找到輪廓
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]



# 繪製原始輪廓和凸包
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)  # 綠色：原始輪廓

# 顯示結果
cv2.imshow("Convex Hull", result)
cv2.waitKey(0)


# 計算凸包
hull = cv2.convexHull(contour)
cv2.drawContours(result, [hull], -1, (255, 0, 0), 2)     # 藍色：凸包

# 顯示結果
cv2.imshow("Convex Hull", result)
cv2.waitKey(0)