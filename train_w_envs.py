from agent.agent import Agent
from envs import TradingEnv
from functions import *
import sys
import torch


#############################
# Trains the model from CLI #
#############################


profits_list = [] # Wild hold list of all profits as we go through training

# Given command line input as below

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

# Unpackage data from terminal
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
num_tech_indicators = 1 # Number of technical indicators we include in each state
agent = Agent(window_size + num_tech_indicators)
data = getStockDataVec(stock_name)
env = TradingEnv(data, window_size)
l = len(data) - 1

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = env.get_state(0)

    # We must manually add profit, as that is not taken into account anywhere else in our model

    total_profit = 0
    env.inventory = []

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = env.get_state(t + 1)
        reward = 0

        if action == 1: # buy
            #remembers the price bought at t, and the time bought 
            env.buy(t)
            # print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(env.inventory) > 0: # sell
            reward, profit = env.sell(t)
            total_profit += profit
            # print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))

        done = True if t == l - 1 else False
        # Push all values to memory
        agent.memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            profits_list.append(total_profit)  

        agent.optimize()

    if e % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        torch.save(agent.policy_net, "models/policy_model")
        torch.save(agent.target_net, "models/target_model")
