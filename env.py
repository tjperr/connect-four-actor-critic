#
# Aim: Env exposes the game where the "player" is always 1 and the opponent is always 0
# but either can go first
#

from game import Game


def encode_board(board, as_player):
    encoded_board = []
    for col in board:
        for cell in col:
            if cell == as_player:
                encoded_board.append(1)
            elif cell == 1 - as_player:
                encoded_board.append(-1)
            else:
                encoded_board.append(0)
    return encoded_board


class Env:
    def __init__(self):
        self.game = Game(verbose=False)
        self.action_space = range(self.game._width)

    def seed(*args):
        pass

    def step(self, action):

        winner, board = self.game.move(action)
    
        if winner is None:

            opponent_move = self.opponent.move(
                encode_board(board, as_player=1 - self.as_player)
            )
            winner, board = self.game.move(opponent_move)

        if winner == self.as_player:
            done = True
            reward = 1
        elif winner == 1 - self.as_player:
            done = True
            reward = -1
        elif winner == -1:
            done = True
            reward = -1
        else:
            done = False
            reward = 0

        return encode_board(self.game.board(), self.as_player), reward, done, None

    def render(self):
        self.game.print(verbose=True)

    def reset(self, opponent, as_player, verbose=False):
        self.opponent = opponent
        self.as_player = as_player
        self.game = Game(verbose=verbose)

        # If we play as 1 then the opponent makes the first move
        if self.as_player == 1:
            board = self.game.board()

            # encode the board for the opponent and get their move
            board = encode_board(board, 1 - self.as_player)
            opponent_move = self.opponent.move(board)

            self.game.move(opponent_move)

        # encode the board for us and return our move
        return encode_board(self.game.board(), self.as_player)
