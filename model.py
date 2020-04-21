import pandas as pd 
from sklearn.linear_model import LogisticRegression

#Create DF
train = pd.read_csv('titanic.csv')
train.dropna(inplace=True)


#features and target
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']

X=train[features]
y=train[target]

#model
model = LogisticRegression()
model.fit(X,y)
model.score(X,y)

import pickle
pickle.dump(model, open('model.pkl','wb'))