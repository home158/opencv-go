# -*- coding: utf-8 -*-

"""
识别图像中的围棋局面
"""

import cv2
import numpy as np
from blue_background_remover import BlueBackgroundRemover
from sgfmill import sgf

from stats import show_phase, stats
class MoveInfo:
    def __init__(self, colour, move, comment=None):
        if(colour == 1):
            self.colour = 'b'  # 'b' for black, 'w' for white
        if(colour == 2):
            self.colour = 'w'  # 'b' for black, 'w' for white
        #np陣列與sgf相反
        self.move = (13 - 1 - move[0],move[1])     # tuple like (x, y), or None for a pass
        self.comment = comment  # optional comment


class GoPhase:
    """从图片中识别围棋局面"""
    
    def __init__(self, pic_path):
        self.game = sgf.Sgf_game(size=13)

        self.im_bgr = None          # 棋盤原圖
        self.im_removebg_bgr = None # 棋盤去背彩圖
        self.board_gray = None #平面校正的棋盤
        self.board_gray_threshold = None #平面校正的棋盤
        self.board_bgr = None #平面校正的棋盤
        self.im_rect_line = None #標記棋盤四角
        self.board_bgr_line = None #標記棋盤線
        self.im_bgr = cv2.imread(pic_path) # 原始的彩色图像文件，BGR模式

        #去除背景
        remover = BlueBackgroundRemover()
        self.im_removebg_bgr = remover.remove_blue_background(self.im_bgr)

        self.im_gray = cv2.cvtColor(self.im_removebg_bgr, cv2.COLOR_BGR2GRAY) # 转灰度
        self.im_gray = cv2.GaussianBlur(self.im_gray, (3,3), 0) # 滤波降噪
        self.im_edge = cv2.Canny(self.im_gray, 30, 50) # 边缘检测
        self.phase = np.zeros((13, 13), dtype=np.ubyte)

        self._find_chessboard()
        self._location_grid()
        self._identify_chessman()
        self._phase_to_sgf()
    def _find_chessboard(self):
        """找到棋盘"""
        contours, hierarchy = cv2.findContours(self.im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 提取轮廓
        area = 0 # 找到的最大四边形及其面积
        for item in contours:
            hull = cv2.convexHull(item) # 寻找凸包
            epsilon = 0.1 * cv2.arcLength(hull, True) # 忽略弧长10%的点
            approx = cv2.approxPolyDP(hull, epsilon, True) # 将凸包拟合为多边形

            if len(approx) == 4 and cv2.isContourConvex(approx): # 如果是凸四边形
                ps = np.reshape(approx, (4,2)) # 四个角的坐标
                ps = ps[np.lexsort((ps[:,0],))] # 排序区分左右
                lt, lb = ps[:2][np.lexsort((ps[:2,1],))] # 排序区分上下
                rt, rb = ps[2:][np.lexsort((ps[2:,1],))] # 排序区分上下
                
                a = cv2.contourArea(approx)
                if a > area:
                    area = a
                    self.rect = (lt, lb, rt, rb)
        if self.rect is None:
            print('在图像文件中找不到棋盘！')
        else:
            print('棋盘坐标：')
            print('\t左上角：(%d,%d)'%(self.rect[0][0],self.rect[0][1]))
            print('\t左下角：(%d,%d)'%(self.rect[1][0],self.rect[1][1]))
            print('\t右上角：(%d,%d)'%(self.rect[2][0],self.rect[2][1]))
            print('\t右下角：(%d,%d)'%(self.rect[3][0],self.rect[3][1]))
        self.im_rect_line = np.copy(self.im_bgr)
        for p in self.rect:
            self.im_rect_line = cv2.line(self.im_rect_line, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
            self.im_rect_line = cv2.line(self.im_rect_line, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)
        edgegap = 10
        if not self.rect is None:
            pts1 = np.float32([(edgegap,edgegap), (edgegap,660+edgegap), (660+edgegap,edgegap), (660+edgegap,660+edgegap)]) # 预期的棋盘四个角的坐标
            pts2 = np.float32(self.rect) # 当前找到的棋盘四个角的坐标
            m = cv2.getPerspectiveTransform(pts2, pts1) # 生成透视矩阵
            self.board_gray = cv2.warpPerspective(self.im_gray, m, (660, 660)) # 执行透视变换
            self.board_bgr = cv2.warpPerspective(self.im_bgr, m, (660, 660)) # 执行透视变换
            #應用二值化 (Binary Threshold)
            _, self.board_gray_threshold = cv2.threshold(self.board_gray, 127, 255, cv2.THRESH_BINARY)

        

    def _location_grid(self):
        """定位棋盘格子"""
        series = np.linspace(25, 630, 13, dtype=np.int64)
        self.board_bgr_line = self.board_bgr.copy()

        for i in series:
            cv2.line(self.board_bgr_line, (25, i), (630, i), (0,255,0), 1)
            cv2.line(self.board_bgr_line, (i, 25), (i, 630), (0,255,0), 1)

    def _identify_chessman(self):
        """识别棋子"""
        mesh = np.linspace(25, 630, 13, dtype=np.int64)
        rows, cols = np.meshgrid(mesh, mesh)
        # 获取第一个点的坐标
        x, y = cols[0, 0], rows[0, 0]
        print(f"坐标: ({x}, {y})")

        # 获取其他点的坐标，例如第 5 行第 3 列
        x, y = cols[12, 11], rows[12, 11]
        print(f"坐标: ({x}, {y})")

        
        # 检测圆形
        circles = cv2.HoughCircles(
            self.board_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=30
        )

        self.board_bgr_circles = self.board_bgr.copy()

        # 确保检测到圆形
        if circles is not None:
            circles = np.uint16(np.around(circles))  # 转为整型并四舍五入
            for circle in circles[0, :]:
                x, y, r = circle
                # 绘制圆形
                cv2.circle(self.board_bgr_circles, (x, y), r, (0, 255, 0), 2)
                # 绘制圆心
                cv2.circle(self.board_bgr_circles, (x, y), 2, (0, 0, 255), 3)

                # 随机给定一个坐标
                circle_point = (x, y)  # 示例随机坐标 (x, y)
                x_circle, y_circle = circle_point

                # 计算欧几里得距离
                distances = np.sqrt((cols - x_circle) ** 2 + (rows - y_circle) ** 2)

                # 找到距离最小值的索引
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)

                # 输出结果
                row_index, col_index = min_idx

                # 檢查坐標是否有效 (避免超出圖片邊界)
                radius = 10
                if (circle_point[0] - radius < 0 or circle_point[1] - radius < 0 or 
                    circle_point[0] + radius >= self.board_gray_threshold.shape[1] or 
                    circle_point[1] + radius >= self.board_gray_threshold.shape[0]):
                    print("Error: The specified circle exceeds the image boundaries.")
                else:
                    # 建立圓形遮罩
                    mask = np.zeros_like(self.board_gray_threshold, dtype=np.uint8)
                    cv2.circle(mask, circle_point, radius, 255, -1)

                    # 使用遮罩提取圓形區域
                    region = cv2.bitwise_and(self.board_gray_threshold, self.board_gray_threshold, mask=mask)

                    # 計算圓形區域內的平均值
                    mean_value = cv2.mean(self.board_gray_threshold, mask=mask)[0]  # 只取灰階通道的平均值

                    # 輸出平均值


                val = self.board_gray_threshold[circle_point]

                print(f"最接近的点的行索引: {row_index}, 列索引: {col_index} , 坐標 {circle_point}, 顏色{val} {mean_value}" )
                if mean_value < 127:
                    self.phase[row_index,col_index] = 1
                if mean_value > 127:
                    self.phase[row_index,col_index] = 2

                
    def _phase_to_sgf(self):
        phase_0 = np.zeros((13, 13), dtype=np.ubyte)
        phase_1 = self.phase
        diff = phase_1 - phase_0
        # 找出非零位置
        non_zero_indices = np.nonzero(diff)

        # 列印結果
        print("非零位置的索引：")
        move_infos = []
        for row, col in zip(non_zero_indices[0], non_zero_indices[1]):
            value = diff[row, col]
            if(value < 3):
                print(f"位置: ({row}, {col}), 值: {value}")
                move_infos.append(MoveInfo(value, (col,row)))

                
        for move_info in move_infos:
            node = self.game.extend_main_sequence()
            node.set_move(move_info.colour, move_info.move)
            if move_info.comment is not None:
                node.set("C", move_info.comment)
        with open("tmp//record.sgf", "wb") as f:
            f.write(self.game.serialise())



    def _stats(self):
        """统计黑白双方棋子和围空"""

    def show_image(self, name='gray', win="GoPhase"):
        """显示图像"""
        
        if name == 'im_removebg_bgr':
            im = self.im_removebg_bgr
        elif name == 'im_bgr':
            im = self.im_bgr
        elif name == 'im_gray':
            im = self.im_gray
        elif name == 'im_edge':
            im = self.im_edge
        elif name == 'board_gray':
            im = self.board_gray
        elif name == 'board_bgr':
            im = self.board_bgr
        elif name == 'im_rect_line':
            im = self.im_rect_line
        elif name == 'board_bgr_line':
            im = self.board_bgr_line
        elif name == 'board_bgr_circles':
            im = self.board_bgr_circles
        elif name == 'board_gray_threshold':
            im = self.board_gray_threshold

        
        if im is None:
            print('识别失败，无图像可供显示')
        else:
            # Resize to 50% of the original size
            width = int(im.shape[1] * 0.75)
            height = int(im.shape[0] * 0.75)
            new_dim = (width, height)

            resized_image = cv2.resize(im, new_dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite("tmp\\"+name+".jpg", im)      
            cv2.imshow(win, resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_phase(self):
        """显示局面"""
    def show_result(self):
        """显示结果"""
        

if __name__ == '__main__':
    go = GoPhase('pic/13x13/10.jpg')

    go.show_image("im_bgr",'原圖')

    go.show_image("im_removebg_bgr",'去背彩圖')
    go.show_image("im_gray",'去背彩圖')
    go.show_image("im_rect_line",'im_rect_line')
    go.show_image("im_edge",'im_edge')
    go.show_image("board_gray",'board_gray')
    go.show_image("board_bgr",'board_bgr')
    go.show_image("board_bgr_line",'board_bgr_line')
    go.show_image("board_bgr_circles",'board_bgr_circles')
    go.show_image("board_gray_threshold",'board_gray_threshold')
