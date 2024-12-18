import cv2
import numpy as np

# 读取图像
img = cv2.imread('rect.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测圆形
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.1,
    minDist=30,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=30
)

# 确保检测到圆形
if circles is not None:
    circles = np.uint16(np.around(circles))  # 转为整型并四舍五入
    for circle in circles[0, :]:
        x, y, r = circle
        # 绘制圆形
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # 绘制圆心
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# 显示结果
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
