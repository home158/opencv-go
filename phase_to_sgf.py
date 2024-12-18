import numpy as np
from sgfmill import sgf
# 創建一個新的 19x19 棋盤遊戲
board_size = 19
game = sgf.Sgf_game(size=board_size)

class MoveInfo:
    def __init__(self, colour, move, comment=None):
        if(colour == 1):
            self.colour = 'b'  # 'b' for black, 'w' for white
        if(colour == 2):
            self.colour = 'w'  # 'b' for black, 'w' for white
        #np陣列與sgf相反
        self.move = (board_size - 1 - move[0],move[1])     # tuple like (x, y), or None for a pass
        self.comment = comment  # optional comment

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
phase_0 = np.zeros((13, 13), dtype=np.ubyte)
# 次一手
phase_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.ubyte)

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
        move_infos.append(MoveInfo(value, (row, col)))

        
for move_info in move_infos:
    node = game.extend_main_sequence()
    node.set_move(move_info.colour, move_info.move)
    if move_info.comment is not None:
        node.set("C", move_info.comment)
with open("src/record.sgf", "wb") as f:
    f.write(game.serialise())


