#Q1: Basic Logistic Regression
#📊 Problem
#X = [1,2,3,4]
#y = [0,0,1,1]
#👉 Train model and predict for x = 2.5

import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[1],[2],[3],[4]])
y = np.array([[0],[0],[1],[1]])
model=LogisticRegression()
model.fit(X,y)
print("class:",model.predict([[2.5]]))
print("prediction:",model.predict_proba([[2.5]]))


#Q2: Pass/Fail Prediction
#📊 Problem
#Study Hours = [1,2,3,4,5]
#Result = [0,0,0,1,1]
#👉 Predict for 3.5 hours

import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[1],[2],[3],[4]])
y = np.array([[0],[0],[1],[1]])
model=LogisticRegression()
model.fit(X,y)
print("prediction for 3.5 hours:",model.predict([[3.5]]))
print("probability:",model.predict_proba([[3.5]]))

#Check Model Accuracy
#📊 Problem
#Use same data → calculate accuracy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
print("Shape of y:", y.shape)
model=LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
print("Accuracy:",accuracy_score(y,y_pred))

#Q4: Multiple Features
#📊 Problem
#X = [[1,2],[2,3],[3,4],[4,5]]
#y = [0,0,1,1]

#👉 Predict for [3,3]

import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([0,0,1,1])
model=LogisticRegression()
model.fit(X,y)
print("Prediction:", model.predict([[3,3]]))
print("Probability:", model.predict_proba([[3,3]]))

#Q5: Confusion Matrix
#📊 Problem
#Evaluate model performance

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model=LogisticRegression()
model.fit(X,y)
y_pred=model.predict(X)
print("confusion matrix:\n",confusion_matrix(y,y_pred))




