from agent.agent import Agent
from envs import TradingEnv
from agent.memory import Transition, ReplayMemory
from functions import *
import sys
import torch

profits_list = []

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
env = TradingEnv(data, window_size)
l = len(data) - 1

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = env.get_state(0)

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
