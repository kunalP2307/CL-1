# -*- coding: utf-8 -*-
"""DMV_Lab_Assignment_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r7bmBZ1U_hfqlMO_2BsOjsB8-BpJhd48
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/DMV Datasets/AirQuality.csv', sep=';')

df.head()

df.columns

df.info()

df.shape

df.describe()

df.isna().sum()

df.duplicated().sum()

df.columns

df = df.iloc[:,:-2]

df.replace(to_replace=',',value='.',regex=True,inplace=True)

columns_to_convert = ['CO(GT)','C6H6(GT)', 'T', 'RH', 'AH']
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df.replace(-200,np.nan,inplace=True)
df.info()

df.drop('NMHC(GT)', axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S').dt.time
df.describe()

numerical_columns = df.select_dtypes(include=[np.number]).columns

for column in numerical_columns:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

for column in numerical_columns:
    plt.figure(figsize=(6,3))
    sns.histplot(x=df[column], stat="count", color="blue", bins=15, kde={'alpha': 0.5})
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

sns.pairplot(df, diag_kind='kde')
plt.show()

df.isnull().sum()

df = df.apply(lambda column: column.interpolate(method="linear") if column.dtype != 'datetime64[ns]' and column.dtype != '<m8[ns]' else column)

df.info()

df.isnull().sum()

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson', min_periods=1),annot=True)

df.columns

df.isna().sum()

df = df.dropna()

df.isna().sum()

df['Datetime'] = df.apply(lambda row: pd.to_datetime(str(row['Date']) + ' ' + str(row['Time'])), axis=1)

df.head()

df.info()

pollutants = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)']
df_pollutants = df[['Datetime'] + pollutants]
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-darkgrid')

for i, pollutant in enumerate(pollutants):
    plt.subplot(len(pollutants), 1, i + 1)
    plt.plot(df_pollutants['Datetime'], df_pollutants[pollutant], label=pollutant)
    plt.xlabel('Datetime')
    plt.ylabel('Value')
    plt.title(f'{pollutant} Trend Over Time')
    plt.legend()

plt.tight_layout()
plt.show()

"""### **Use bar plots or stacked bar plots to compare the AQI values across different dates ortime periods**"""

df['AQI'] = (df['CO(GT)'] + df['C6H6(GT)'] + df['NOx(GT)'] + df['NO2(GT)'] + df['PT08.S5(O3)']) / 5
df_aqi_by_date = df.groupby('Date')['AQI'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.style.use('seaborn-darkgrid')
plt.bar(df_aqi_by_date['Date'], df_aqi_by_date['AQI'], color='b', alpha=0.7)

plt.xlabel('Date')
plt.ylabel('Mean AQI')
plt.title('Mean AQI by Date')
plt.xticks(rotation=45)
plt.tight_layout()

"""### **Create box plots or violin plots to analyze the distribution of AQI values for different pollutant categories**"""

df['AQI'] = (df['CO(GT)'] + df['C6H6(GT)'] + df['NOx(GT)'] + df['NO2(GT)'] + df['PT08.S5(O3)']) / 5
pollutant_categories = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)']
df_plot = df[['AQI'] + pollutant_categories]

plt.figure(figsize=(12, 6))
plt.style.use('seaborn-darkgrid')

plt.subplot(1, 2, 1)
sns.boxplot(data=df_plot, orient='h')
plt.xlabel('AQI')
plt.title('Box Plot of AQI by Pollutant Categories')

plt.subplot(1, 2, 2)
sns.violinplot(data=df_plot, orient='h')
plt.xlabel('AQI')
plt.title('Violin Plot of AQI by Pollutant Categories')
plt.tight_layout()

"""### **Use scatter plots or bubble charts to explore the relationship between AQI values and pollutant levels.**

"""

df['AQI'] = (df['CO(GT)'] + df['C6H6(GT)'] + df['NOx(GT)'] + df['NO2(GT)'] + df['PT08.S5(O3)']) / 5

pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)']

plt.figure(figsize=(16, 10))
plt.style.use('seaborn-darkgrid')

for i, pollutant in enumerate(pollutants, start=1):
    plt.subplot(2, 3, i)
    plt.scatter(df[pollutant], df['AQI'], alpha=0.5)
    plt.xlabel(pollutant)
    plt.ylabel('AQI')
    plt.title(f'AQI vs. {pollutant}')

plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-darkgrid')

plt.scatter(df['NOx(GT)'], df['AQI'], c='b', alpha=0.5, s=df['CO(GT)']*10)  # Adjust the s multiplier for proper scaling
plt.xlabel('NOx(GT)')
plt.ylabel('AQI')
plt.title('AQI vs. NOx(GT) with CO(GT) Bubble Size')
plt.tight_layout()

