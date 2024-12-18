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


cv2.line(board_gray, (330-10,330), (330+10,330), (255,0,0), 1)
cv2.line(board_gray, (330,330-10), (330,330+10), (255,0,0), 1)

cv2.imshow('rect.jpg', board_gray)

cv2.waitKey(0)

series = np.linspace(36, 620, 13, dtype=np.int64)

for i in series:
	im = cv2.line(board_bgr, (36, i), (620, i), (0,255,0), 1)
	im = cv2.line(board_bgr, (i, 36), (i, 620), (0,255,0), 1)
	
cv2.imshow('go', im)

cv2.waitKey(0)

mesh = np.linspace(36, 620, 13, dtype=np.int64)
rows, cols = np.meshgrid(mesh, mesh)

print(rows)
print(cols)
cv2.destroyAllWindows()