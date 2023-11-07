# -*- coding: utf-8 -*-
"""DMV3_TELE_CUS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nR-7nvxhEEtxj9ZQeVcB_DSgUzSWBBMM
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('Telco_customer_churn.csv')
pd.set_option('display.max_columns', None)



df.info()

df.columns



df.head()

cols = ['Multiple Lines','Internet Service','Contract','Payment Method', 'Churn Reason']
for col in cols:
    print(col, df[col].unique())

missing = df.isnull().sum().sum()
missing



df['Churn Reason'].value_counts()

churn_val_counts = df['Churn Reason'].value_counts()

available = len(df.index) - missing

available

total = 0
update_counts = []
for item in churn_val_counts:
    item = item / available
    item = int(item * missing)
    total = total + item
    update_counts.append(item)

total

rows_with_null = df[df['Churn Reason'].isnull()]
row_numbers = rows_with_null.index.tolist()

len(row_numbers)

i = 0
k = 0
for tup in churn_val_counts.items():
    reason = tup[0]
    count = update_counts[i]
    i += 1
    j = 0
    while j < count:
        df.at[row_numbers[k], 'Churn Reason'] = reason
        j += 1
        k += 1

df['Churn Reason'].value_counts()

df['Churn Reason'].isnull().sum()

# replacing remaing null values with mode

mode_reason = df['Churn Reason'].mode()

df['Churn Reason'].fillna(mode_reason[0], inplace=True)

df['Churn Reason'].isnull().sum()

len(df[df.duplicated()])

del df['Country']



df["Senior Citizen"] = df["Senior Citizen"].replace({"Yes":1,"No":0})

df.head()

from sklearn.preprocessing import MinMaxScaler

num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']

df.dtypes

value_exists = df['Total Charges'] == " "

value_exists[2234]

df = df[-value_exists]

value_exists = df['Total Charges'] == " "

df['Total Charges'] = df['Total Charges'].astype(float)

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
sns.histplot(df['Total Charges'], kde=True, bins=10)
plt.title('Total Charges Histogram')

plt.subplot(1,3,2)
sns.histplot(df['Tenure Months'], kde=True, bins=10)
plt.title('Total Charges Histogram')

plt.subplot(1,3,3)
sns.histplot(df['Monthly Charges'], kde=True, bins=10)
plt.title('Total Charges Histogram')
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head(1)

plt.figure(figsize=(12,4))
for i, col in enumerate(num_cols):
    plt.subplot(1, len(num_cols), i + 1)
    sns.histplot(df[col], kde=True, bins=10)
    plt.title(f'{col} Histogram (Normalized)')
plt.tight_layout()

from sklearn.model_selection import train_test_split

X = df.loc[:, df.columns != 'Churn Label']

y = df['Churn Label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

X_train.shape

y_train.shape

df.to_csv('output.csv', index=True)



























