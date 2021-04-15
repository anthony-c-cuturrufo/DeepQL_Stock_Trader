import numpy as np 
from functions import *

class TradingEnv():
    def __init__(self, train_data, window_size):
        #list of all closing values from beg - end
        self.data = train_data
        #size of recent closing price list
        self.window_size = window_size
        #keeps track of buying prices 
        self.inventory = []
        
    def get_reward(self, selling_price, time_sold, bought_price, time_bought):
        delta_t = time_sold - time_bought
        profit = selling_price - bought_price
        reward = max(profit, .0001) // (np.log(delta_t) + 1)
        return reward
        
    # returns an an n-day state representation ending at time t
    def get_state(self, t):
        n = self.window_size + 1
        d = t - n + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1] # pad with t0
        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))

        return np.array([res])
    
    def buy(self, t):
        # keeps track of the price bought and time bought
        self.inventory.append((self.data[t], t))
    
    def sell(self, t):
        #sells the oldest stock in portfolio
        bought_price, time_bought = self.inventory.pop(0)
        selling_price = self.data[t] 
        reward = self.get_reward(selling_price, t, bought_price, time_bought)
        profit = selling_price - bought_price
        return reward, profit
