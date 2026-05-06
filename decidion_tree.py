#✅ Q1: Basic Decision Tree Classification
#📊 Problem
#X = [[1],[2],[3],[4]]
#y = [0,0,1,1]
#👉 Predict for 2.5
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4]])
y = np.array([0,0,1,1])
model=DecisionTreeClassifier()
model.fit(X,y)
print("prediction:",model.predict([[2.5]]))

#✅ Q2: Multi-Feature Classification
#📊 Problem
#X = [[1,2],
#     [2,3],
#     [3,4],
#     [4,5]]
#y = [0,0,1,1]
#👉 Predict for [3,3]
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1,2],
              [2,3],
              [3,4],
              [4,5]])
y = np.array([0,0,1,1])
model=DecisionTreeClassifier()
model.fit(X,y)
print("prediction:",model.predict([[3,3]]))

#✅ Q3: Check Accuracy
#📊 Problem
#Evaluate model performance
import numpy as np
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4]])
y = np.array([0,0,1,1])
model=DecisionTreeClassifier()
model.fit(X,y)
y_pred=model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))

#✅ Q4: Train-Test Split (IMPORTANT 🔥)
#📊 Problem
#Split data and evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Q5: Control Overfitting
# 📊 Problem
# Limit tree depth#
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)

print("Prediction:", model.predict([[3.5]]))

#✅ Q6: Feature Importance
#📊 Problem
#Find which feature is most important
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1,2],
              [2,3],
              [3,4],
              [4,5]])
y = np.array([0,0,1,1])
model=DecisionTreeClassifier()
model.fit(X,y)
print("Feature importance:", model.feature_importances_)

#Q7: Visualize Decision Tree
#📊 Problem
#Plot the tree#
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
X = np.array([[1,2],
              [2,3],
              [3,4],
              [4,5]])
y = np.array([0,0,1,1])
model=DecisionTreeClassifier()
model.fit(X,y)
plt.figure(figsize=(10,6))
plot_tree(model, filled=True)
plt.show()


#Q8: Decision Tree Regression
#📊 Problem
#X = [[1],[2],[3],[4]]
#y = [10,20,30,40]
#👉 Predict value for 2.5
import numpy as np
from sklearn.tree import DecisionTreeRegressor
X = np.array([[1],[2],[3],[4]])
y = np.array([10,20,30,40])
model=DecisionTreeRegressor()
model.fit(X,y)
print("prediction:",model.predict([[2.5]]))


