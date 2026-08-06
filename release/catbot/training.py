import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################
def Qnew(a, Qsa,reward, discount, optimal):
    return (1-a)*Qsa+a*(reward+discount*optimal)

def decode_state(state):
    agent_row = state // 1000
    agent_col = (state // 100) % 10
    cat_row   = (state // 10) % 10
    cat_col   = state % 10
    return agent_row, agent_col, cat_row, cat_col


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def reward(prevDistPlayerToCat, currDistPlayerToCat):
    if currDistPlayerToCat == prevDistPlayerToCat:
        return -1
    elif prevDistPlayerToCat - currDistPlayerToCat > 0:
        return 1
    elif prevDistPlayerToCat - currDistPlayerToCat < 0:
        return -5
    return 0


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project

    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    learning_rate = 0.01
    discount = 0.99

    epsilon = 0.5 #
    epsilon_decay = 0.995
    epsilon_min = 0.01


    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        obs, info = env.reset()
        done = False

        while not done:

            current_state = obs
            agent_row, agent_col, cat_row, cat_col = decode_state(obs)
            distance = manhattan_distance(agent_row, agent_col, cat_row, cat_col)
            action = env.action_space.sample()


            ## decide
            random_value = random.random()
            if random_value < epsilon: #explore if less than epsilon
                next_state, _, terminated, truncated, _ = env.step(action)
                next_agent_row, next_agent_col, next_cat_row, next_cat_col = decode_state(next_state)
                next_distance = manhattan_distance(
                    next_agent_row,
                    next_agent_col,
                    next_cat_row,
                    next_cat_col
                )
                done = terminated or truncated
                if done:
                    r = 100
                else:
                    r = reward(distance, next_distance)

                best_future = np.max(q_table[next_state])

                q_table[current_state][action] = Qnew(
                    learning_rate,
                    q_table[current_state][action],
                    r,
                    discount,
                    best_future
                )

                obs = next_state


            else: #exploit if greater than epsilon
                action = np.argmax(q_table[obs])
                next_state, _, terminated, truncated, _ = env.step(action)
                next_agent_row, next_agent_col, next_cat_row, next_cat_col = decode_state(next_state)
                next_distance = manhattan_distance(
                    next_agent_row,
                    next_agent_col,
                    next_cat_row,
                    next_cat_col
                )
                done = terminated or truncated
                if done:
                    r = 100
                else:
                    r = reward(distance, next_distance)


                best_future = np.max(q_table[next_state])

                q_table[current_state][action] = Qnew(
                    learning_rate,
                    q_table[current_state][action],
                    r,
                    discount,
                    best_future
                )
                obs = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table