#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# 2. Rename Timestamp to Date and convert to datetime objects
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 3. Index the dataframe on Date
df = df.set_index('Date')

# 4. Fill missing values in Close with the previous row (Forward Fill)
df['Close'] = df['Close'].ffill()

# 5. Fill missing High, Low, and Open with the Close value of the SAME row
# We use fillna() here so we don't overwrite existing valid data
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# 6. Set missing Volume values to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 7. Resample to daily intervals ('D') from 2017 onwards
# This combines the slicing and the aggregation logic
df = df.loc['2017':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plotting the result
df.plot()
plt.show()
