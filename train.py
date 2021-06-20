# TODO
# 1. modify the as_player logic to make the bot train as player 0 and 1 randomly
#    make sure it's reported/rendered correctly
#
# 2. implement the adversary and switching. train from scratch each time? E.g.:
#    a. start with two from-scratch models, train one against the other
#    b. pick the best of the two (judged through test on 1000 games), throw the other away
#    c. train a from-scratch against the kept model, go to b.
#

import collections
import random
import statistics
import time
from typing import Any, List, Sequence, Tuple

import numpy as np
import tqdm
from matplotlib import pyplot as plt

import tensorflow as tf
from env import Env
from model import ActorCritic, BasicModel
from tensorflow.keras import layers

# Create the environment
env = Env()

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = env.step(action)
    # print(f"state={state}")
    # print(f"reward={reward}")
    # print(f"done={done}")
    return (
        # state.astype(np.float32)
        np.array(state, np.float32),
        np.array(reward, np.int32),
        np.array(done, np.int32),
    )


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):

        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(
    rewards: tf.Tensor, gamma: float, standardize: bool = True
) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    # whats the rationale behind standardising?
    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (
            tf.math.reduce_std(returns) + eps
        )

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
    action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor
) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def get_train_step_fn():
    @tf.function
    def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int,
    ) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = run_episode(
                initial_state, model, max_steps_per_episode
            )

            # Calculate expected returns
            returns = get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]
            ]

            # Calculating loss values to update our network
            loss = compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
    return train_step

# Set up training loop
min_episodes_criterion = 100
max_episodes = 1000
max_steps_per_episode = 1000
reward_threshold = 0.6  # Threshold to stop training

# Discount factor for future rewards
gamma = 0.99


def run_epoch(model, opponent):
    running_reward = 0
    train_step_fn = get_train_step_fn()

    # Keep last n episodes reward
    episodes_reward: collections.deque = collections.deque(
        maxlen=min_episodes_criterion
    )

    with tqdm.trange(max_episodes) as t:
        for i in t:

            initial_state = tf.constant(
                env.reset(opponent=opponent, as_player=0), dtype=tf.float32
            )

            episode_reward = int(
                train_step_fn(
                    initial_state, model, optimizer, gamma, max_steps_per_episode
                )
            )

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f"Episode {i}")
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

            # Show an episode every 500 episodes
            if i % 500 == 0:
                as_player = random.randint(0, 1)

                action_probs, values, rewards = run_episode(
                    tf.constant(
                        env.reset(opponent=opponent, as_player=as_player, verbose=True),
                        dtype=tf.float32,
                    ),
                    model,
                    max_steps=100,
                )
                print(f"as_player={as_player}")
                print("rewards:")
                tf.print(rewards)
                time.sleep(4)
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > reward_threshold and i >= min_episodes_criterion:
                break

    print(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")


num_actions = len(env.action_space)
num_hidden_units = 128

def performance(model, opponent):
    performance = []
    for _ in tqdm.tqdm(range(200)):
        action_probs, values, rewards = run_episode(
            tf.constant(
                env.reset(opponent=opponent, as_player=random.randint(0, 1), verbose=False),
                dtype=tf.float32,
            ),
            model,
            max_steps=100,
        )
        performance.append(rewards[-1].numpy())

    return np.mean(performance)
    

opponent = ActorCritic(num_actions, num_hidden_units)

for _ in range(10):
    # initialise a new model
    model = ActorCritic(num_actions, num_hidden_units)

    # train the model
    run_epoch(model, opponent)

    # if the model is better, make this the new opponent
    model_perf = performance(model, opponent)
    print(f"Model v opponent: {model_perf}")
    print(f"Model v BasicModel: {performance(model, BasicModel())}")

    if model_perf > 0:
        print("replacing opponent")
        opponent = model
    else:
        print("retaining opponent")

# TODO: save the model
# play the model :D