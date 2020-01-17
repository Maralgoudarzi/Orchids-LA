import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv('/Users/maral/Downloads/xAPI-Edu-Data.csv')
df.head()

df.rename(index=str, columns={'gender':'Gender',
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                               inplace=True)
df.columns

sns.countplot(x="Topic", data=df, palette="muted");
plt.show()

sns.countplot(x='ParentschoolSatisfaction',data = df, hue='Class',palette='bright')
plt.show()


sns.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='bright')
plt.show()