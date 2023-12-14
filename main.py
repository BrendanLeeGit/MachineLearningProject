import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.svm import SVC
import requests

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
    print(x_data, close_list)


def get_test_data():
    return deserialize('serialized_x_values.pkl'), deserialize('serialized_y_values.pkl')


def serialize(file_name, data_to_serialize):
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(data_to_serialize, file)


def deserialize(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


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
    print(np.shape(X))

    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    print(theta)

    xPlot = np.arange(min(x) - 0.5, max(x) + 0.5, 0.1)
    yPlot = theta[0] + theta[1] * xPlot

    plt.plot(xPlot, yPlot, 'g-')
    plt.xlabel("Hours")
    plt.ylabel("Stock Price (USD)")
    plt.title("Linear Regression - Linear Fit to Predict the Stock Data")
    plt.show()


# Prediction using Linear Regression - Quadratic
def linear_reg_quadratic():
    # Generate a Training set
    x, y = np.array(get_test_data())
    m = len(x) 
    plt.scatter(x, y)

    # Form the design matrix
    X = np.transpose([np.ones(m), x, x ** 2])
    print(np.shape(X))

    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    print(theta)

    xPlot = np.arange(min(x), max(x), 0.1)
    yPlot = theta[0] + theta[1] * xPlot + theta[2] * xPlot ** 2

    plt.plot(xPlot, yPlot, 'r-')
    plt.xlabel("Hours")
    plt.ylabel("Stock Price (USD)")
    plt.title("Linear Regression - Quadratic Fit to Predict the Stock Data")
    plt.show()


# Prediction using Locally Weighted Regression (LOWESS)
def lowess():
    # Generate Training examples
    m = 1000
    x = np.arange(0, 2, 0.002)
    y = np.sin(np.pi * x)
    y = y + 0.25 * np.random.randn(m)

    plt.scatter(x, y)

    # Create Design Matrix
    X = np.transpose([np.ones(m), x])
    print(np.shape(X))
    theta = inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    yp = theta[0] + theta[1] * x
    plt.plot(x, yp, 'g')

    # Query Point - 1
    xt1 = 0.55

    # Assigning weights to training examples
    T = 0.05  # Bandwidth parameter # 0.05, Make it large - the LR and LOWESS predictions overlap w/ the default plot
    w = np.exp(-(x - xt1) ** 2 / (2 * T ** 2))
    W = np.diag(w)

    theta_1 = inv(np.transpose(X) @ W @ X) @ np.transpose(X) @ W @ y
    x_Range1 = np.arange(0.25, 0.95, 0.01)
    yp_1 = theta_1[0] + theta_1[1] * x_Range1
    plt.plot(x_Range1, yp_1, 'r')

    # Prediction with Linear Regression (LR)
    yt1 = theta[0] + theta[1] * xt1
    plt.plot(xt1, yt1, 'm*')

    # Prediction with LOWESS
    yt1_lowess = theta_1[0] + theta_1[1] * xt1
    plt.plot(xt1, yt1_lowess, 'k*')

    ###########################################
    # Query Point - 2
    xt2 = 1.5

    # Assigning weights to training examples
    T = 0.05  # Bandwidth parameter
    w = np.exp(-(x - xt2) ** 2 / (2 * T ** 2))
    W = np.diag(w)

    theta_2 = inv(np.transpose(X) @ W @ X) @ np.transpose(X) @ W @ y
    x_Range2 = np.arange(1.25, 1.75, 0.01)
    yp_2 = theta_2[0] + theta_2[1] * x_Range2
    plt.plot(x_Range2, yp_2, 'r')

    # Prediction with Linear Regression (LR)
    yt2 = theta[0] + theta[1] * xt2
    plt.plot(xt2, yt2, 'm*')

    # Prediction with LOWESS
    yt2_lowess = theta_2[0] + theta_2[1] * xt2
    plt.plot(xt2, yt2_lowess, 'k*')


# Prediction using SVM
# Commenting this out until we have the X and Y values
# def SVM():
# dataset = pd.read_csv('/content/Dataset.csv')
# X = dataset.iloc[:, [2, 3]].values # column numbers 2 and 3 which has salary, etc.
# Y = dataset.iloc[:, 4].values

# X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# sc_X = StandardScaler()

# X_Train = sc_X.fit_transform(X_Train)
# X_Test = sc_X.transform(X_Test) # we use transform here to use the same mu and sigma from the training set

# classifier = SVC(kernel = 'poly', degree = 5) # See the differet decision boundaries (default is degree = 5), Can change the degree = 3, degree = 1, degree = 7
# classifier.fit(X_Train, Y_Train)

# Y_Pred = classifier.predict(X_Test) # Prediction
# cm = confusion_matrix(Y_Test, Y_Pred) # Confusion Matrix
# print(cm)

# X_Set, Y_Set = X_Train, Y_Train

# X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
# plt.figure()
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#                                         alpha = 0.25, cmap = ListedColormap(('red', 'green')))
# for i, j in enumerate(np.unique(Y_Set)):

# plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
#             c = ListedColorMap(('red', 'green'))(i), label = j)

# plt.title('SVM (Training set)')
# plt.xlabel('X')
# plt.ylabel('X2')
# plt.legend()

# # Visualizing Test set results

# X_Set, Y_Set = X_Test, Y_Test

# X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
# plt.figure()
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#                                         alpha = 0.25, cmap = ListedColormap(('red', 'green')))
# for i, j in enumerate(np.unique(Y_Set)):

# plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
#             c = ListedColorMap(('red', 'green'))(i), label = j)

# plt.title('SVM (Test set)')
# plt.xlabel('X')
# plt.ylabel('X2')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    # Current serialized data used the parameters: 'IBM', '60min'
    # print(get_test_data())

    # Linear Regression
    linear_reg_linear()
    linear_reg_quadratic()
