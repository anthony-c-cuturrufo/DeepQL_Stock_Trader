import numpy as np
import math
import pandas as pd
from ta import trend

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file. Only takes the close
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        close = line.split(",")[4]
        if close != 'null':
            vec.append(float(line.split(",")[4]))

    return vec

# returns Simple Moving Average from closing prices
def getSMAFromVec(v, window_size):
    df = pd.DataFrame.from_dict({'close': v})
    sma = trend.SMAIndicator(df['close'], window_size, True)
    return sma.sma_indicator().values


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])
