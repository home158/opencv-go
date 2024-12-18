import cv2
import numpy as np

# 创建空白图像
image = np.zeros((400, 400, 3), dtype=np.uint8)

# 画几个圆形
cv2.circle(image, (100, 100), 40, (255, 255, 255), -1)  # 白色实心圆
cv2.circle(image, (300, 100), 50, (255, 255, 255), -1)  # 白色实心圆
cv2.circle(image, (200, 300), 60, (255, 255, 255), -1)  # 白色实心圆

# 保存为文件
cv2.imwrite('circles.jpg', image)

# 显示图像
cv2.imshow("Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
