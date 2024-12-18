import cv2
import numpy as np

class BlueBackgroundRemover:
    def __init__(self, lower_blue=None, upper_blue=None):
        """
        初始化蓝色背景去除器。

        :param lower_blue: 蓝色下界（默认：[100, 50, 50]）。
        :param upper_blue: 蓝色上界（默认：[140, 255, 255]）。
        """
        # 设置默认的蓝色范围
        self.lower_blue = lower_blue if lower_blue is not None else np.array([100, 50, 50])
        self.upper_blue = upper_blue if upper_blue is not None else np.array([140, 255, 255])

    def remove_blue_background(self, input_img):
        """
        去除图片的蓝色背景。

        :param input_path: 输入图片路径。
        :param output_path: 输出图片路径（支持 PNG 透明背景）。
        """

        # 转换为 HSV 色彩空间
        hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

        # 创建蓝色遮罩
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # 反转遮罩（保留非蓝色部分）
        mask_inv = cv2.bitwise_not(mask)

        # 应用遮罩，保留原图中的非蓝色部分
        result = cv2.bitwise_and(input_img, input_img, mask=mask_inv)

        # 处理透明背景
        b, g, r = cv2.split(result)  # 拆分通道
        alpha = mask_inv  # 将反转的遮罩作为 alpha 通道
        rgba = cv2.merge([b, g, r, alpha])  # 合并成带透明通道的图像

        # 保存结果
        return rgba
        #cv2.imwrite(output_path, rgba)

    def set_blue_range(self, lower_blue, upper_blue):
        """
        设置蓝色的 HSV 范围。

        :param lower_blue: 蓝色下界。
        :param upper_blue: 蓝色上界。
        """
        self.lower_blue = np.array(lower_blue)
        self.upper_blue = np.array(upper_blue)
        print(f"蓝色范围更新为: lower={self.lower_blue}, upper={self.upper_blue}")
