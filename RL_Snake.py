import os
import gymnasium_snake_game
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# options for snake game
options = {
    'fps': 100,
    'max_step': 500,
    'init_length': 4,
    ## reward values
    'food_reward': 3.0,
    'dist_reward': 0.0,
    'death_penalty': -3.05,
    'living_bonus': 0.05,
    ##
    'width': 10,
    'height': 10,
    'block_size': 100,
    'background_color': (28, 31, 41),
    'food_color': (145, 29, 29),
    'head_color': (24, 66, 13),
    'body_color': (10, 140, 19),
}

# function to create the snake game environment and apply the options listed above. its a function so make_vec_env accepts it.
def create_env():
    return gymnasium.make('Snake-v1', render_mode='human', **options)

# vectorizes multiple environments into one. in this case its only 1 environment 
env = make_vec_env(create_env, n_envs=1)

# The PPO learning model
"""
Short for Proximity Policy Optimization model, The idea of this algorithm is that the new policy made should be closer to the old policy after updates and it uses clipping to 
stop it from changing too much and avoid large updates. 
"""

def PPO_model():

    # selecting the model to use, in this case its PPO
    model = PPO('MlpPolicy', env, verbose=1)

    # define how long should the model train for.
    model.learn(total_timesteps=100000)

    # Use save model to create a new training model file and save it. Otherwise use load if a file already exists. 
    # the whole os line just makes sure the file lands in the same directory as the python file.
    model = PPO.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PPO_model"))

    # reset the environment
    obs = env.reset()
    for i in range(2000):
        # predict the action and take action
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

# The DQN learning model
"""
Short for Deep Q-Network model, it uses a neural network to find out the Q-value function which is the expected reward for each action in a given state. It basically predicts the best action to take
based on the current state.
"""
def DQN_model():

    # selecting the model to use, in this case its DQN
    model = DQN('MlpPolicy', env, verbose=1)

    # define how long should the model train for.
    model.learn(total_timesteps=100000)

    # Use save model to create a new training model file and save it. Otherwise use load if a file already exists. 
    # the whole os line just makes sure the file lands in the same directory as the python file.
#    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DQN_model"))
    model = DQN.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DQN_model"))

    # reset the environment
    obs = env.reset()
    for i in range(2000):
        # predict the action and take action
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

# Pick a model to use
# PPO_model()
DQN_model()

# DQN model has proven to be the more effective one 
