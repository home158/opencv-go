import numpy as np
"""
使用NumPy二維ubytem數組存儲局面：
0 - 空
1 - 黑子
2 - 白子
定義位置如下
( 0, 0) ( 0, 1) ( 0, 2) ...... ( 0,18)
( 1, 0)
( 2, 0)
.
.
.
.
.
.
(18, 0) (18, 1) (18, 2) ...... (18,18)
"""
# 前一手
phase_0 = np.array([
    [0,0,2,1,1,0,1,1,1,2,0,2,0,2,1,0,1,0,0],
    [0,0,2,1,0,1,1,1,2,0,2,0,2,2,1,1,1,0,0],
    [0,0,2,1,1,0,0,1,2,2,0,2,0,2,1,0,1,0,0],
    [0,2,1,0,1,1,0,1,2,0,2,2,2,0,2,1,0,1,0],
    [0,2,1,1,0,1,1,2,2,2,2,0,0,2,2,1,0,1,0],
    [0,0,2,1,1,1,1,2,0,2,0,2,0,0,2,1,0,0,0],
    [0,0,2,2,2,2,1,2,2,0,0,0,0,0,2,1,0,0,0],
    [2,2,2,0,0,0,2,1,1,2,0,2,0,0,2,1,0,0,0],
    [1,1,2,0,0,0,2,2,1,2,0,0,0,0,2,1,0,0,0],
    [1,0,1,2,0,2,1,1,1,1,2,2,2,0,2,1,1,1,1],
    [0,1,1,2,0,2,1,0,0,0,1,2,0,2,2,1,0,0,1],
    [1,1,2,2,2,2,2,1,0,0,1,2,2,0,2,1,0,0,0],
    [2,2,0,2,2,0,2,1,0,0,1,2,0,2,2,2,1,0,0],
    [0,2,0,0,0,0,2,1,0,1,1,2,2,0,2,1,0,0,0],
    [0,2,0,0,0,2,1,0,0,1,0,1,1,2,2,1,0,0,0],
    [0,0,2,0,2,2,1,1,1,1,0,1,0,1,1,0,0,0,0],
    [0,2,2,0,2,1,0,0,0,0,1,0,0,0,0,1,1,0,0],
    [0,0,2,0,2,1,0,1,1,0,0,1,0,1,0,1,0,0,0],
    [0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
], dtype=np.ubyte)
# 次一手
phase_1 = np.array([
    [1,0,0,1,1,0,1,1,1,2,0,2,0,2,1,0,1,0,0],
    [0,0,2,1,0,1,1,1,2,0,2,0,2,2,1,1,1,0,0],
    [0,0,2,1,1,0,0,1,2,2,0,2,0,2,1,0,1,0,0],
    [0,2,1,0,1,1,0,1,2,0,2,2,2,0,2,1,0,1,0],
    [0,2,1,1,0,1,1,2,2,2,2,0,0,2,2,1,0,1,0],
    [0,0,2,1,1,1,1,2,0,2,0,2,0,0,2,1,0,0,0],
    [0,0,2,2,2,2,1,2,2,0,0,0,0,0,2,1,0,0,0],
    [2,2,2,0,0,0,2,1,1,2,0,2,0,0,2,1,0,0,0],
    [1,1,2,0,0,0,2,2,1,2,0,0,0,0,2,1,0,0,0],
    [1,0,1,2,0,2,1,1,1,1,2,2,2,0,2,1,1,1,1],
    [0,1,1,2,0,2,1,0,0,0,1,2,0,2,2,1,0,0,1],
    [1,1,2,2,2,2,2,1,0,0,1,2,2,0,2,1,0,0,0],
    [2,2,0,2,2,0,2,1,0,0,1,2,0,2,2,2,1,0,0],
    [0,2,0,0,0,0,2,1,0,1,1,2,2,0,2,1,0,0,0],
    [0,2,0,0,0,2,1,0,0,1,0,1,1,2,2,1,0,0,0],
    [0,0,2,0,2,2,1,1,1,1,0,1,0,1,1,0,0,0,0],
    [0,2,2,0,2,1,0,0,0,0,1,0,0,0,0,1,1,0,0],
    [0,0,2,0,2,1,0,1,1,0,0,1,0,1,0,1,0,0,0],
    [0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
], dtype=np.ubyte)

diff = phase_1 - phase_0
# 找出非零位置
non_zero_indices = np.nonzero(diff)

# 列印結果
print("非零位置的索引：")
for row, col in zip(non_zero_indices[0], non_zero_indices[1]):
    value = diff[row, col]
    if(value < 3):
        print(f"位置: ({row}, {col}), 值: {value}")