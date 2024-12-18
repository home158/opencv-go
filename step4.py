import cv2
import numpy as np
from blue_background_remover import BlueBackgroundRemover

remover = BlueBackgroundRemover()
input_file = r'pic/13x13/1.jpg'
output_file = "output.png"
remover.remove_blue_background(input_file, output_file)

im_bgr = cv2.imread(output_file) # 读入图像




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

#红色的十字标注在原始的彩色图像上
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

if not rect is None:
    pts1 = np.float32([(10,10), (10,650), (650,10), (650,650)]) # 预期的棋盘四个角的坐标
    pts2 = np.float32(rect) # 当前找到的棋盘四个角的坐标
    m = cv2.getPerspectiveTransform(pts2, pts1) # 生成透视矩阵
    board_gray = cv2.warpPerspective(im_gray, m, (660, 660)) # 执行透视变换
    board_bgr = cv2.warpPerspective(im_bgr, m, (660, 660)) # 执行透视变换

cv2.imshow('go', board_gray)
cv2.imwrite('rect.jpg', board_gray)

cv2.waitKey(0)


circles = cv2.HoughCircles(board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=90, param2=16, minRadius=10, maxRadius=20) # 圆检测
xs, ys = circles[0,:,0], circles[0,:,1] # 所有棋子的x坐标和y坐标
xs.sort()
ys.sort()

k = 1
while xs[k]-xs[:k].mean() < 15:
    k += 1
x_min = int(round(xs[:k].mean()))

k = 1
while ys[k]-ys[:k].mean() < 15:
    k += 1

y_min = int(round(ys[:k].mean()))

k = -1
while xs[k:].mean() - xs[k-1] < 15:
    k -= 1
x_max = int(round(xs[k:].mean()))

k = -1
while ys[k:].mean() - ys[k-1] < 15:
    k -= 1
y_max = int(round(ys[k:].mean()))

if abs(600-(x_max-x_min)) < abs(600-(y_max-y_min)):
    v_min, v_max = x_min, x_max
else:
    v_min, v_max = y_min, y_max
    
pts1 = np.float32([[22, 22], [22, 598], [598, 22], [598, 598]])  # 棋盘四个角点的最终位置
pts2 = np.float32([(v_min, v_min), (v_min, v_max), (v_max, v_min), (v_max, v_max)])
m = cv2.getPerspectiveTransform(pts2, pts1)
board_gray = cv2.warpPerspective(board_gray, m, (620, 620))
board_bgr = cv2.warpPerspective(board_bgr, m, (620, 620))
cv2.line(im, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
cv2.imshow('go', board_gray)
cv2.imwrite('rect.jpg', board_gray)

cv2.waitKey(0)

cv2.destroyAllWindows()