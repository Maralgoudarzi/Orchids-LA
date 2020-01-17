
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('/Users/maral/Downloads/xAPI-Edu-Data.csv')
df.head()

df['TotalQ'] = df['Class']
df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0

continuous_subset = df.iloc[:, 9:13]
continuous_subset['gender'] = np.where(df['gender'] == 'M', 1, 0)
continuous_subset['Parent'] = np.where(df['Relation'] == 'Father', 1, 0)

X = np.array(continuous_subset).astype('float64')
y = np.array(df['TotalQ'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

