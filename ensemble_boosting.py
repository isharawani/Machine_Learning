#✅ Q1: Basic AdaBoost Classification
#📊 Problem
#Train an AdaBoost classifier and predict output.#

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10
)
model.fit(X, y)
print("Prediction:", model.predict([[3.5]]))

#✅ Q2: Accuracy of AdaBoost
#📊 Problem
#Calculate training accuracy.
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model=AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10
)
model.fit(X,y)
y_pred=model.predict(X)
print("accuracy:",accuracy_score(y,y_pred))

#✅ Q3: Compare Different Estimators
#📊 Problem
#Train:
#10 estimators
#100 estimators
#Compare predictions.#
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
model1 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10
)
model2 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100
)
model1.fit(X, y)
model2.fit(X, y)
print("10 Estimators:",
      model1.predict([[4]]))
print("100 Estimators:",
      model2.predict([[4]]))

#✅ Q4: Gradient Boosting Classification
#📊 Problem#
#Train Gradient Boosting classifier.
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1 # Control contribution of each tree
)
model.fit(X, y)
print("Prediction:",model.predict([[4]]))

#✅ Q4: Train-Test Split with Bagging
#📊 Problem
#Evaluate Boosting on test data.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",
      accuracy_score(y_test, y_pred))#

#✅ Q7: Compare AdaBoost vs Decision Tree
#📊 Problem
#Compare accuracy.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])
tree = DecisionTreeClassifier(max_depth=1)
boost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50
)
tree.fit(X, y)
boost.fit(X, y)
pred_tree = tree.predict(X)
pred_boost = boost.predict(X)
print("Tree Accuracy:",accuracy_score(y, pred_tree))
print("Boost Accuracy:",accuracy_score(y, pred_boost))

#✅ Q8: Hyperparameter Tuning using GridSearchCV (Detailed Explanation)
#We are tuning a Gradient Boosting Classifier.#
#n_estimators = 10
#learning_rate = 0.1
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
# Dataset
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])
y = np.array([0,0,0,1,1,1])
# Hyperparameter combinations
params = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 1]
}
# Model
model = GradientBoostingClassifier()
# GridSearchCV
grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3
)
# Training
grid.fit(X, y)
# Best result
print("Best Parameters:")
print(grid.best_params_)
print("Best Accuracy:")
print(grid.best_score_)