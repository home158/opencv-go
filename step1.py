import cv2
import numpy as np

pic_file = r'demo4.jpg'
im_bgr = cv2.imread(pic_file) # 读入图像
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度
im_gray = cv2.GaussianBlur(im_gray, (7,7), 0) # 滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测
cv2.imshow('Go', im_edge) # 显示边缘检测结果
cv2.waitKey(0)
cv2.destroyAllWindows()
