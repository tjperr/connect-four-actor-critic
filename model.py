import random
from copy import deepcopy
from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.special import softmax

import tensorflow as tf
from game import BOARD_HEIGHT, BOARD_WIDTH, Game, run_game
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def move(self, state):

        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = self.call(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]

        return action


class BasicModel:

    _name = "basic_model"

    def move(self, board):
        board = np.reshape(board, (BOARD_WIDTH, BOARD_HEIGHT))

        # sometimes go for the middle
        if np.random.random() < 0.25 and sum(np.abs(board[3])) < BOARD_HEIGHT:
            # print(3)
            return 3
        # sometimes play randomly
        if np.random.random() < 0.25:
            # print('random')
            rand = random.randint(0, BOARD_WIDTH - 1)
            # print(f"rand={rand}")
            # print(sum(np.abs(board[rand])))
            if sum(np.abs(board[rand])) < BOARD_HEIGHT:
                return rand
        # sometimes cover the first 1 you see
        else:
            # print('cover')
            for i in range(len(board)):
                # Find the top value in column i
                top_value = 0
                for x in board[i]:
                    if x != 0:
                        top_value = x

                # print(f"i={i}")
                # print(f"top_value={top_value}")
                # print(sum(np.abs(board[i])))
                if sum(np.abs(board[i])) < BOARD_HEIGHT and top_value == -1:
                    return i
        return random.randint(0, BOARD_WIDTH - 1)

    def fit(self, *args, **kwargs):
        pass


class RandomModel:

    _name = "random_model"

    def move(self, board, as_player):
        return random.randint(0, BOARD_WIDTH - 1)

    def fit(self, *args, **kwargs):
        pass


class Me:
    def move(self, board, as_player):
        return int(input(f"choose column 0-{BOARD_WIDTH-1}: "))


class Model:

    _model = None
    _name = None
    _moves = []

    def __init__(self, load_model_name=None, model_name="model"):
        if load_model_name:
            self._model = keras.models.load_model("models/" + load_model_name)
            self._name = load_model_name
        else:
            self.initialise()
            self._name = model_name

    def move(
        self,
        board,
        as_player,
        stochastic=False,
        print_probs=False,
        valid_moves_only=False,
    ):

        pred = self.predict(board, as_player)

        if valid_moves_only:
            base_smax = [x for x in pred[0]]
            for i in range(BOARD_WIDTH):
                if len(board[i]) >= BOARD_HEIGHT:
                    base_smax[i] = -9999
            smax = softmax(base_smax)
        else:
            smax = softmax([x for x in pred[0]])

        if print_probs:
            print([round(x, 2) for x in pred[0]])
            print([round(x, 2) for x in smax])

        if stochastic:
            move = random.choices(range(len(smax)), smax)[0]
        else:
            move = np.argmax(smax)

        self._moves.append(move)
        return move

    def predict(self, board, as_player):
        return self._model.predict(self.input_encoding(board, as_player))

    def initialise(self):
        self._model = Sequential()
        self._model.add(layers.InputLayer(input_shape=(1, BOARD_WIDTH * BOARD_HEIGHT)))
        self._model.add(layers.Dense(4 * BOARD_WIDTH * BOARD_HEIGHT, activation="tanh"))
        self._model.add(layers.Dense(4 * BOARD_WIDTH * BOARD_HEIGHT, activation="tanh"))
        self._model.add(layers.Dense(BOARD_WIDTH, activation="linear"))
        self._model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            metrics=["mse"],
        )

    def input_encoding(self, board, as_player):
        if as_player == 1:
            input_vector = self.board_to_vec(board)
        else:
            reversed_board = [[1 - cell for cell in col] for col in board]
            input_vector = self.board_to_vec(reversed_board)

        return input_vector

    def board_to_vec(self, board, length=BOARD_HEIGHT):
        copy = deepcopy(board)
        for b in copy:
            b += [None] * (length - len(b))

        input_layer_0 = [tile_encoding(tile) for col in copy for tile in col]

        return np.array([input_layer_0])

    def fit_one(self, board, as_player, y, *args, **kwargs):
        self._model.fit(self.input_encoding(board, as_player), y, *args, **kwargs)

    def save(self, model_name=None):
        if self._name:
            self._model.save("models/" + self._name)
        elif model_name:
            self._model.save("models/" + model_name)
        else:
            print("please provide model name")


def tile_encoding(x):
    if x == 0:
        return 1
    elif x == 1:
        return -1
    else:
        return 0
