import pickle
import requests

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Stocks API Key
api_key = 'VOQRXUCBM8M146AW'

def get_stock_data(symbol, interval):
    # So the url is split into different parts:
    # 1. base URL: https://www.alphavantage.co/query?...
    #    This is just the base website
    # 2. function: TIME_SERIES_INTRADAY
    #    This decides what information we'll get. In this case, it's the stock prices throughout the day
    # 3. symbol: IBM, AAPL, etc
    #    This is just the stock's representative name
    # 4. interval: 5min, 15min, 30min, 60min
    #    This is the interval between each data point. 5 min means that the data is the stock's price every five minutes
    # 5. apikey: VOQRXUCBM8M146AW
    #    The api key we use to access the API. It only has 25 uses per day.
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + symbol + '&interval=' + \
          interval + f'&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Gather all the 'close' data (Which means the final price of the stock at the end of the interval)
    close_list = [float(value['4. close']) for value in data['Time Series (' + interval + ')'].values()]

    # The data is from recent to old, so instead we're going to reverse the list
    close_list.reverse()

    # Now the x data :D
    x_data = []
    for x in range(len(close_list)):
        x_data.append(x)

    # Testing the data let's print it all
    print("get_stock_data:", x_data, close_list)


def get_test_data():
    return deserialize('serialized_x_values.pkl'), deserialize('serialized_y_values.pkl')


def serialize(file_name, data_to_serialize):
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(data_to_serialize, file)


def deserialize(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

########### REGRESSION ###########

# Prediction using Linear Regression - Linear
def linear_reg_linear():
    # Generate a Training set
    x, y = np.array(get_test_data())
    # m is the number of training examples
    m = len(x)
    # Plot as a scatter plot
    plt.scatter(x, y)

    # Form the Design Matrix
    X = np.transpose([np.ones(m), x])
    print("Shape of X (LR Linear):", np.shape(X))

    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    print("Theta (LR Linear):", theta)

    xPlot = np.arange(min(x), max(x), 0.1)
    yPlot = theta[0] + theta[1] * xPlot

    plt.plot(xPlot, yPlot, 'g-', label="Linear Regression (Linear Fit)")
    plt.xlabel("Hours")
    plt.ylabel("Stock Price (USD)")
    plt.title("Linear Regression - Linear Fit to Predict the Stock Data")
    plt.legend()
    plt.show()


# Prediction using Linear Regression - Quadratic
def linear_reg_quadratic():
    # Generate a Training set
    x, y = np.array(get_test_data())
    m = len(x) 
    plt.scatter(x, y)

    # Form the design matrix
    X = np.transpose([np.ones(m), x, x ** 2])
    print("Shape of X (LR Quadratic):", np.shape(X))

    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    print("Theta (LR Quadratic):", theta)

    xPlot = np.arange(min(x), max(x), 0.1)
    yPlot = theta[0] + theta[1] * xPlot + theta[2] * xPlot ** 2

    plt.plot(xPlot, yPlot, 'r-', label="Linear Regression (Quadratic Fit)")
    plt.xlabel("Hours")
    plt.ylabel("Stock Price (USD)")
    plt.title("Linear Regression - Quadratic Fit to Predict the Stock Data")
    plt.legend()
    plt.show()


# Prediction using Locally Weighted Regression (LOWESS)
# Linear Regression: Green line with the magenta stars
# LOWESS: Red line with the black stars
def lowess():
    # Generate Training examples
    x, y = np.array(get_test_data())
    m = len(x) 
    plt.scatter(x, y)

    # Create Design Matrix
    X = np.transpose([np.ones(m), x])
    print("Shape of X (LOWESS):", np.shape(X))
    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    yp = theta[0] + theta[1] * x
    plt.plot(x, yp, 'g', label="Linear Regression (Linear Fit)") # Comment out this line if we don't need Linear Regression

    # Note: Query points are out of order (check the ranges)

    ########## Query Point - 1 ##########
    x_Range1, yp_1, xt1, yt1, yt1_lowess = query_point(m, x, y, X, theta, 0.125, 0.475, 0.25)

    plt.plot(x_Range1, yp_1, 'r', label="LOWESS")
    plt.plot(xt1, yt1, 'm*', label="yt1 (Linear Regression)")
    plt.plot(xt1, yt1_lowess, 'k*', label="yt1 (LOWESS)")

    ########## Query Point - 2 ##########
    x_Range2, yp_2, xt2, yt2, yt2_lowess = query_point(m, x, y, X, theta, 0.375, 0.725, 0.5)

    plt.plot(x_Range2, yp_2, 'r')
    plt.plot(xt2, yt2, 'm*')
    plt.plot(xt2, yt2_lowess, 'k*')

    ########## Query Point - 3 ##########
    x_Range3, yp_3, xt3, yt3, yt3_lowess = query_point(m, x, y, X, theta, 0.625, 0.875, 0.75)

    plt.plot(x_Range3, yp_3, 'r')
    plt.plot(xt3, yt3, 'm*')
    plt.plot(xt3, yt3_lowess, 'k*')

    ########## Query Point - 4 ##########
    x_Range4, yp_4, xt4, yt4, yt4_lowess = query_point(m, x, y, X, theta, 0.875, 1, 1)

    plt.plot(x_Range4, yp_4, 'r')
    plt.plot(xt4, yt4, 'm*')
    plt.plot(xt4, yt4_lowess, 'k*')

    ########## Query Point - 5 ##########
    x_Range5, yp_5, xt5, yt5, yt5_lowess = query_point(m, x, y, X, theta, 0, 0.125, 0)

    plt.plot(x_Range5, yp_5, 'r')
    plt.plot(xt5, yt5, 'm*')
    plt.plot(xt5, yt5_lowess, 'k*')

    plt.xlabel("Hours")
    plt.ylabel("Stock Price (USD)")
    plt.title("LOWESS to Predict the Stock Data")
    plt.legend()
    plt.show()

# Queries a point for LOWESS
"""
Takes m for the number of training examples, x and y for the training sets, X for the design matrix, theta,
2 range multiplier values for x_Range, and the percentage of the number of training examples you want to use
"""
def query_point(m, x, y, X, theta, range_m1, range_m2, percentTrainEx):
    xt1 = percentTrainEx*m # % of the number of training examples

    # Assigning weights to training examples
    T = 0.05*m # Bandwidth parameter; 0.05 * the number of training examples
    w = np.exp(-(x - xt1) ** 2 / (2 * T ** 2))
    W = np.diag(w)

    theta_1 = inv(np.transpose(X) @ W @ X) @ np.transpose(X) @ W @ y
    x_Range1 = np.arange(m*range_m1, m*range_m2, 0.01) # Change the range
    yp_1 = theta_1[0] + theta_1[1] * x_Range1

    # Used for prediction with Linear Regression (LR)
    yt1 = theta[0] + theta[1] * xt1

    # Used for prediction with LOWESS
    yt1_lowess = theta_1[0] + theta_1[1] * xt1

    return x_Range1, yp_1, xt1, yt1, yt1_lowess

if __name__ == '__main__':
    # Current serialized data used the parameters: 'IBM', '60min'
    print("Test Data:", get_test_data())

    # Linear Regression
    linear_reg_linear()
    linear_reg_quadratic()

    # LOWESS
    lowess()
