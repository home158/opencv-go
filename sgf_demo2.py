from sgfmill import sgf

# 函數：將棋盤坐標轉換為 SGF 格式
def coord_to_sgf(coord):
    """將棋盤坐標 (row, col) 轉換為 SGF 格式 (如 'dc')"""
    row, col = coord
    return f"{chr(ord('a') + col)}{chr(ord('a') + row)}"

# 創建一個新的 19x19 棋盤遊戲
game = sgf.Sgf_game(size=19)

# 第一步：黑子下在 (3, 3)，無提子
node1 = game.extend_main_sequence()
node1.set_move('b', (3, 3))

# 第二步：白子下在 (4, 3)，無提子
node2 = game.extend_main_sequence()
node2.set_move('w', (4, 3))

# 第三步：黑子下在 (5, 3)，提掉白子在 (4, 3)
node3 = game.extend_main_sequence()
node3.set_move('w', (2, 3))

# 第三步：黑子下在 (5, 3)，提掉白子在 (4, 3)
node3 = game.extend_main_sequence()
node3.set_move('w', (3, 4))

# 第三步：黑子下在 (5, 3)，提掉白子在 (4, 3)
node3 = game.extend_main_sequence()
node3.set_move('w', (3, 2))

# 將結果存入 SGF 文件
with open("capture_example.sgf", "wb") as f:
    f.write(game.serialise())

print("SGF file with capture saved to 'capture_example.sgf'")
