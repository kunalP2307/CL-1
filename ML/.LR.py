import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,r2_score 
import seaborn as sns

df = pd.read_csv("uber.csv")
df.head()

df.drop(['Unnamed: 0', 'key'], axis=1, inplace=True)

df.fillna(method='ffill', inplace=True)

df[df['fare_amount'].values <=0]

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

corr_mat = df.corr()

sns.heatmap(corr_mat,annot=True,cmap='coolwarm')

df.drop(df[df['fare_amount'].values <=0].index,inplace=True)

X = df[['passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']]
y = df['fare_amount']

lr_model = LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

lr_model.fit(X_train,y_train)

y_pred_linear = lr_model.predict(X_test)

y_pred_linear[3]

y_test.iloc[3]

from math import sqrt

rsme = sqrt(mean_squared_error(y_test, y_pred_linear))

r2 = r2_score(y_test,y_pred_linear)

print("RSME",rsme)
print("R2",r2)

lasso = Lasso()

lasso.fit(X_train,y_train)

y_pred_lasso = lasso.predict(X_test)

