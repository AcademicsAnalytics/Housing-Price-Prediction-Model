import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import datetime
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set your API key
api_key = 'PKERRKZWOPFJRDBG'

# Create a TimeSeries object with your API key
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch data
df, meta_data = ts.get_daily(symbol='GOOGL', outputsize='full')

# Print column names for debugging
print(df.columns)

# Select necessary columns
df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]

# Calculate percentage changes
df['HL_PCT'] = (df['2. high'] - df['4. close']) / df['4. close'] * 100.0
df['PCT_change'] = (df['4. close'] - df['1. open']) / df['1. open'] * 100.0

# Select final columns
df = df[['4. close', 'HL_PCT', 'PCT_change', '5. volume']]

# Forecast column
forecast_col = '4. close'
df.fillna(-99999, inplace=True)

# Forecast length
forecast_out = int(math.ceil(0.01 * len(df)))

# Create label column
df['label'] = df[forecast_col].shift(-forecast_out)

# Drop NaNs **before** creating `X`
df.dropna(inplace=True)

# Define feature set
X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)

# Define labels
y = np.array(df['label'])

print(len(X), len(y))  # Debugging

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f"Model Accuracy: {accuracy}")

# Forecast future values
X_lately = X[-forecast_out:]
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# Prepare for plotting
df['Forecast'] = np.nan
last_date = df.index[-1]
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Plot results
df['4. close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

