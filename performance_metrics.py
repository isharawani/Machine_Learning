#Q1: Accuracy Calculation
#📊 Problem
#y_true = [1,0,1,1]
#y_pred = [1,0,0,1]

#👉 Find Accuracy#

from sklearn.metrics import accuracy_score
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
print("accuracy:",accuracy_score(y_true,y_pred))


#Q2: Precision, Recall, F1
#📊 Problem
#Use same data → find all metrics
from sklearn.metrics import precision_score , recall_score , f1_score
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
print("PRECISION:",precision_score(y_true,y_pred))
print("RECALL:",recall_score(y_true,y_pred))
print("F1:",f1_score(y_true,y_pred))


#Q3: Confusion Matrix
#📊 Problem
#Print confusion matrix

from sklearn.metrics import confusion_matrix
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
print("CONFUSION:",confusion_matrix(y_true,y_pred))

#Q4: Full Classification Report
#📊 Problem

#Generate full report#

from sklearn.metrics import classification_report
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
print("Classification report:",classification_report(y_true,y_pred))

#Regression Metrics
#📊 Problem
#y_true = [10,20,30]
#y_pred = [12,18,29]

#👉 Find MAE, MSE, R²#

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_true = [1,0,1,1]
y_pred = [1,0,0,1]
print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("R2:", r2_score(y_true, y_pred))