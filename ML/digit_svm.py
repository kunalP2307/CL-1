# -*- coding: utf-8 -*-
"""digit_svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_WPkwMjIFkjHpMrCmBId5aWeKw6dpt7A
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('/content/drive/MyDrive/Datasets/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Datasets/test.csv')

X = train.drop('label', axis=1)
y = train['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.svm import SVC

model = SVC(kernel='rbf')
model.fit(X_train,y_train)

y_svm_pred = model.predict(X_test)

print("Predicted Values",y_svm_pred[17:23])
print("Actual Values",y_test[17:23])

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_svm_pred)
print(accuracy)

import numpy as np
import matplotlib.pyplot as plt
for i in range(100,104):
  mat = np.reshape(X_test[i], (28,28))
  plt.title(y_svm_pred[i])
  plt.imshow(mat,cmap='gray')
  plt.show()


















