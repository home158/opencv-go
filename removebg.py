import cv2
import numpy as np

# 读取图片
pic_file = r'pic/13x13/1.jpg'

image = cv2.imread(pic_file)

cv2.imshow('原始圖檔', image)
cv2.waitKey(0)

# 转换为 HSV 色彩空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV 色彩空间', hsv)
cv2.waitKey(0)
# 定义蓝色的 HSV 范围 (可根据具体图片调整)
lower_blue = np.array([100, 50, 50])  # 蓝色下界
upper_blue = np.array([140, 255, 255])  # 蓝色上界

# 创建蓝色遮罩
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 反转遮罩（保留非蓝色部分）
mask_inv = cv2.bitwise_not(mask)

# 应用遮罩，保留原图中的非蓝色部分
result = cv2.bitwise_and(image, image, mask=mask_inv)

# 如果需要生成透明背景的 PNG 文件
b, g, r = cv2.split(result)  # 拆分通道
alpha = mask_inv  # 将反转的遮罩作为 alpha 通道
rgba = cv2.merge([b, g, r, alpha])  # 合并成带透明通道的图像

# 保存结果
cv2.imwrite("output.png", rgba)

print("蓝底去除完成，结果保存在 output.png")
