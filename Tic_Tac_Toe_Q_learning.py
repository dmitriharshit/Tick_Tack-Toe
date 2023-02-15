import random

blank = ' '
human = 'X'
AI = 'O'
epochs = 40000
epsilon = 0.4
win = 10
lose = -10
tie = 0


class Player:
    @staticmethod
    def show_board(board):
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))

# parent class is Player


class HumanPlayer(Player):

    def reward(self, value, board):
        pass

    def make_move(self, board):

        while True:
            try:
                self.show_board(board)
                move = input('Your next move (cell index 1-9):')
                move = int(move)

                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move-1


class AI_Player(Player):
    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9, default_q=1):
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # Discount parameter for future reward
        # if the given move at the given state is not defined yet: we have default Q-value
        self.default_q = default_q
        # Q(s,a) function is a dictionary in this implementation.
        # It returns a value for state s and move a
        self.q = {}
        self.move = None  # previous move during the game
        self.board = (' ',)*9  # previous state of board

    # these are the empty cells on the grid
    def available_cell(self, board):
        return [i for i in range(9) if board[i] == ' ']

    # This function return Q value for state,action pair:
    def get_Q(self, state, action):
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = self.default_q
        return self.q[(state, action)]

    # make a move with epsilon probabilitya(exploration) or pick the action with the highest Q value
    def make_move(self, board):
        self.board = tuple(board)
        action = self.available_cell(board)

        # action with epsilon rpobability
        if random.random() < self.epsilon:
            self.move = random.choice(action)
            return self.move

        # exploitation:
        # we have to find the action with highest Q value
        q_values = [self.get_Q(self.board, a) for a in action]
        max_q = max(q_values)

        # if multiple actions choose one at random
        if q_values.count(max_q) > 1:
            best_action = [i for i in range(
                len(action)) if q_values[i] == max_q]
            best_move = action[random.choice(best_action)]

        else:
            best_move = action[q_values.index(max_q)]

        self.move = best_move
        return self.move

    def reward(self, reward, board):
        if self.move:
            q_prev = self.get_Q(self.board, self.move)
            q_max = max([self.get_Q(tuple(board), a)
                         for a in self.available_cell(self.board)])
            self.q[self.board, self.move] = q_prev + \
                self.alpha*((reward+self.gamma*q_max)-q_prev)


class Tic_Tac_Toe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' ']*9

    def play(self):
        # this is the game loop:
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                tickers = (AI, human)
            else:
                player = self.player2
                other_player = self.player1
                tickers = (human, AI)

            # check the state of the game(win,lose or draw)
            # returns boolean value and name of player who won if its wasn't a draw
            game_over, winner = self.check_state(tickers)

            if game_over:
                if winner == tickers[0]:
                    player.show_board(self.board[:])
                    print('\n %s won!' % player.__class__.__name__)
                    player.reward(win, self.board[:])
                    other_player.reward(lose, self.board[:])
                if winner == tickers[1]:
                    player.show_board(self.board[:])
                    print('\n %s won!' % other_player.__class__.__name__)
                    other_player.reward(win, self.board[:])
                    player.reward(lose, self.board[:])
                else:
                    player.show_board(self.board[:])
                    print('its a TIE!!')
                    other_player.reward(tie, self.board[:])
                    player.reward(tie, self.board[:])
                break

            # next player's turm:
            self.first_player_turn = not self.first_player_turn
            # player's move according to Q values (for AI)
            move = player.make_move(self.board)
            self.board[move] = tickers[0]

    def check_state(self, player_ticker):
        # we consider both player ('X' and 'O')
        for k in player_ticker:

            # checking horizontally:
            for i in range(3):
                if self.board[3*i+0] == k and self.board[3*i+1] == k and self.board[3*i+2] == k:
                    return True, k

            # checking vertically (columns)
            for i in range(3):
                if self.board[3*0+i] == k and self.board[3*1+i] == k and self.board[3*2+i] == k:
                    return True, k

            # check diagonal dimensions (top left to bottom right + top right to bottom left)
            if self.board[0] == k and self.board[4] == k and self.board[8] == k:
                return True, k

            if self.board[2] == k and self.board[4] == k and self.board[6] == k:
                return True, k

        # finally we check draw case
        if self.board.count(' ') == 0:
            return True, None
        else:
            return False, None


if __name__ == '__main__':

    ai_player1 = AI_Player()
    ai_player2 = AI_Player()

    print('Training the algo...')

    for _ in range(epochs):
        game = Tic_Tac_Toe(ai_player1, ai_player2)
        game.play()

    print('Training is Done!')

    # epsilon=0 means it wont explore anymore but will play using Q-values
    ai_player1.epsilon = 0
    human_player = HumanPlayer()
    game = Tic_Tac_Toe(ai_player1, human_player)
    game.play()
