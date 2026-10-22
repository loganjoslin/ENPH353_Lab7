#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space # 0=line on left; 1=line centered; 2=line on right
    action_space = env.action_space # 0=drive left, 1=drive fwd; 2=drive right

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = loadQ("best_q_table.npy")

    highest_reward = float("-inf")

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print(f"New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table.npy")

            state, _ = env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    if next_state == -1:
        next_state_term = 0
        # handle road going off screen; this terminating state has no associated actions to contribute to reward
    else:
        next_state_term = gamma * np.max(q_table[next_state]) 

    current_value = q_table[state][action]
    new_value = current_value + alpha * (reward + next_state_term - current_value)
    q_table[state][action] = new_value
    return q_table


def choose_action(state, q_table, epsilon, action_space):
    explore = random.random() < epsilon
    if explore: return action_space.sample()
    q_vals = q_table[state]
    max_q = np.max(q_vals)
    max_indeces = []
    for i in range(len(q_vals)):
        if q_vals[i] == max_q:
            max_indeces.append(i)
    return random.choice(max_indeces)

def saveQ(q_table, save_path):
    np.save(save_path, q_table)

def loadQ(load_path):
    return np.load(load_path)

if __name__ == "__main__":
    main()