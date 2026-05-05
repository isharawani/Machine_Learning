#Q1: Basic KNN Classification
#📊 Problem
#Train KNN:
#X = [[1],[2],[3],[4]]
#y = [0,0,1,1]
#👉 Predict for 2.5

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
X = np.array([[1],[2],[3],[4]])
y = np.array([0,0,1,1])

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
print("prediction:",model.predict([[2.5]]))

##Q2: Multi-Feature KNN
#📊 Problem
#X = [[1,2],
#     [2,3],
#     [3,4],
#     [4,5]]
#y = [0,0,1,1]
#👉 Predict for [3,3]

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([0,0,1,1])
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
print("prediction:",model.predict([[3,3]]))

#Q2: Check Model Accuracy
#📊 Problem
#Use same data → calculate accuracy
import numpy as np
from sklearn.metrics import accuracy_score
y_pred=model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))

#Q4: Train-Test Split (IMPORTANT 🔥)
#📊 Problem
#Split data and evaluate accuracy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Q5: With Feature Scaling (VERY IMPORTANT)
# Problem
#apply scaling before KNN\
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)
x_new = scaler.transform([[3.5]])
print("Prediction:", model.predict(x_new))#