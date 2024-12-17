import cv2
import numpy as np

pic_file = r'demo4.jpg'
im_bgr = cv2.imread(pic_file) # 读入图像
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测

contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 提取轮廓
rect, area = None, 0 # 找到的最大四边形及其面积
for item in contours:
	hull = cv2.convexHull(item) # 寻找凸包
	epsilon = 0.1 * cv2.arcLength(hull, True) # 忽略弧长10%的点
	approx = cv2.approxPolyDP(hull, epsilon, True) # 将凸包拟合为多边形
	if len(approx) == 4 and cv2.isContourConvex(approx): # 如果是凸四边形
		ps = np.reshape(approx, (4,2))
		ps = ps[np.lexsort((ps[:,0],))]
		lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
		rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
		a = cv2.contourArea(approx) # 计算四边形面积
		if a > area:
			area = a
			rect = (lt, lb, rt, rb)

if rect is None:
	print('在图像文件中找不到棋盘！')
else:
	print('棋盘坐标：')
	print('\t左上角：(%d,%d)'%(rect[0][0],rect[0][1]))
	print('\t左下角：(%d,%d)'%(rect[1][0],rect[1][1]))
	print('\t右上角：(%d,%d)'%(rect[2][0],rect[2][1]))
	print('\t右下角：(%d,%d)'%(rect[3][0],rect[3][1]))

	
im = np.copy(im_bgr)
for p in rect:
	im = cv2.line(im, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
	im = cv2.line(im, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)


# Resize to 50% of the original size
width = int(im.shape[1] * 0.5)
height = int(im.shape[0] * 0.5)
new_dim = (width, height)

resized_image = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)

# Display the resized image
cv2.imshow('Resized Image', resized_image)

cv2.waitKey(0)

cv2.destroyAllWindows()