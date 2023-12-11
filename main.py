import requests

# Stocks API Key
api_key = 'VOQRXUCBM8M146AW'

def get_test_data():
    url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}')
    r = requests.get(url)
    data = r.json()
    print(data)

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    get_test_data()
    print_hi('Machine Learning')
