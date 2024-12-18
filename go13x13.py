# -*- coding: utf-8 -*-

"""
识别图像中的围棋局面
"""

import cv2
import numpy as np
from blue_background_remover import BlueBackgroundRemover

from stats import show_phase, stats


class GoPhase:
    """从图片中识别围棋局面"""
    
    def __init__(self, pic_path):
        self.im_bgr = None          # 棋盤原圖
        self.im_removebg_bgr = None # 棋盤去背彩圖

        self.im_bgr = cv2.imread(pic_path) # 原始的彩色图像文件，BGR模式
        self._find_chessboard() # 找到棋盘


    def _find_chessboard(self):
        """找到棋盘"""
        remover = BlueBackgroundRemover()
        self.im_removebg_bgr =remover.remove_blue_background(self.im_bgr)

    def _location_grid(self):
        """定位棋盘格子"""
    def _identify_chessman(self):
        """识别棋子"""
    def _stats(self):
        """统计黑白双方棋子和围空"""

    def show_image(self, name='gray', win="GoPhase"):
        """显示图像"""
        
        if name == 'im_removebg_bgr':
            im = self.im_removebg_bgr
        elif name == 'im_bgr':
            im = self.im_bgr

        
        if im is None:
            print('识别失败，无图像可供显示')
        else:
            cv2.imshow(win, im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_phase(self):
        """显示局面"""
    def show_result(self):
        """显示结果"""
        

if __name__ == '__main__':
    go = GoPhase('pic/13x13/1.jpg')
    go.show_image("im_bgr",'原圖')

    go.show_image("im_removebg_bgr",'去背彩圖')
