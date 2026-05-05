#Q1: Basic Classification
#📊 Problem
#Train a Naive Bayes model:
#X = [1,2,3,4]
#y = [0,0,1,1]
#👉 Predict for 2.5#
import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[1],[2],[3],[4]])
y = np.array([[0],[0],[1],[1]])
model=GaussianNB()
model.fit(X,y)
print("prediction:",model.predict([[2.5]]))
print("probabilities:",model.predict_proba([[2.5]]))


#Q2: Check Model Accuracy
#📊 Problem
#Use same data → calculate accuracy
import numpy as np
from sklearn.metrics import accuracy_score
y_pred=model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))


#Q3: Multi-Feature Dataset
#📊 Problem
#X = [[1,2],
#     [2,3],
#     [3,4],
#     [4,5]]
#y = [0,0,1,1]
#👉 Predict for [3,3]

import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[1,2],
              [2,3],
              [3,4],
              [4,5]])

y = np.array([0,0,1,1])
model=GaussianNB()
model.fit(X,y)
print("prediction:",model.predict([[3,3]]))

#Q4: Train-Test Split (IMPORTANT 🔥)
#📊 Problem
#Split data and evaluate accuracy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
model=GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


#Q5: Confusion Matrix
#📊 Problem
#Evaluate model performance#
from sklearn.metrics import confusion_matrix
print("confusion matrix:",confusion_matrix(y_test, y_pred))
