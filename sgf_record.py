from sgfmill import sgf
class MoveInfo:
    def __init__(self, colour, move, comment=""):
        self.colour = colour  # 'b' for black, 'w' for white
        self.move = move      # tuple like (x, y), or None for a pass
        self.comment = comment  # optional comment

move_infos = [
    MoveInfo('w', (3, 3), "Opening move"),
    MoveInfo('b', (4, 3), "Opening move"),
    MoveInfo('b', (2, 3), "Opening move"),
    MoveInfo('b', (3, 4), "Opening move"),
    MoveInfo('b', (3, 2), "Opening move"),
]
# 創建一個新的 19x19 棋盤遊戲
game = sgf.Sgf_game(size=19)
for move_info in move_infos:
    node = game.extend_main_sequence()
    node.set_move(move_info.colour, move_info.move)
    if move_info.comment is not None:
        node.set("C", move_info.comment)
with open("src/record.sgf", "wb") as f:
    f.write(game.serialise())


