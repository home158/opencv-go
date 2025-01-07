import cv2
import numpy as np

# 創建測試圖像
image = np.zeros((400, 400), dtype=np.uint8)
p1 = np.array([[10, 50], [10, 100], [100, 50]])
p2 = np.array([[50, 30], [90, 200], [100, 90]])
points = [p1,p2]

cv2.drawContours(image, points, -1, 255, -1)
# 顯示結果
cv2.imshow("draw Contours", image)
cv2.waitKey(0)


# 找到輪廓
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contour=[]
for c in contours:
    contour.append(c)
    


# 繪製原始輪廓
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, contour, 0, (0, 255, 0), 1)  # 綠色：原始輪廓

# 顯示結果
cv2.imshow("Convex Hull", result)
cv2.waitKey(0)

# 繪製凸包
for item in contours:
    hull = cv2.convexHull(item) # 寻找凸包
    cv2.drawContours(result, [hull], -1, (255, 0, 0), 2)     # 藍色：凸包

# 顯示結果
cv2.imshow("Convex Hull", result)
cv2.waitKey(0)