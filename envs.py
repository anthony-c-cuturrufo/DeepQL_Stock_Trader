from functions import *

class TradingEnv():
    """
    Represents the trading environment used in our model.
    Handles low-level data scraping, retrieval, and calculation
    Adjustable parameters:
        get_reward(params): the reward function of a certain action
        get_state(params): the state that the model is currently in
    """
    def __init__(self, train_data, window_size):
        '''
        Creates a trading environment from data train_data with window size window_size
        :param train_data: data to be trained on, e.g. daily closing prices
        :param window_size: size of the window on which we examine stock trends
        '''
        # List of all daily closing prices
        self.data = train_data
        # List of Simple Moving Averages from the window
        self.sma_data = getSMAFromVec(train_data, window_size)
        # Size of recent closing price list
        self.window_size = window_size
        # Keeps track of buying prices
        self.inventory = []
        
    def get_reward(self, selling_price, time_sold, bought_price, time_bought):
        delta_t = time_sold - time_bought
        profit = selling_price - bought_price
        reward = max(profit, .0001) // (np.log(delta_t) + 1)
        return reward
    
    def get_weighted_diff(self, v1, v2):
        return (abs(v2 - v1)) / v1
        
    def get_state(self, t):
        '''
        Our state representation.
        :param t: time
        :return: n-day state representation ending at time t with sma indicator at end
        '''
        n = self.window_size + 1
        d = t - n + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1] # pad with t0
        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))

        # add sigmoid of price and sma
        res.append(sigmoid(self.get_weighted_diff(self.data[t], self.sma_data[t])))
        res = np.array([res])
        return res
    
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
