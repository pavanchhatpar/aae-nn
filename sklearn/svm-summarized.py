from sklearn import svm
import numpy as np
from math import ceil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

data = np.load('../data/aae-summarized-data.npy').tolist()
X = np.array(data['X_data'])
y = np.array(data['y_data']).flatten()

Yt = []
Pt = []
skf = StratifiedKFold(n_splits=10, shuffle=True) # stratified 10-fold cross-validation

for train_index, test_index in skf.split(X, y):
    X_train , X_test , y_train , y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    clf = svm.SVC(C=1.5, kernel='linear', tol=0.001, cache_size=200, class_weight=None)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    Yt.extend(y_test)
    Pt.extend(pred)

cfm = confusion_matrix(Yt, Pt)
TN = cfm[0][0]
FP = cfm[0][1]
FN = cfm[1][0]
TP = cfm[1][1]
P = TP + FN
N = FP + TN
print(cfm)
print("Accuracy: {:.2f}".format(accuracy_score(Yt, Pt)*100))
print("Sensitivity: {:.2f}".format(TP/P*100))
print("Specificity: {:.2f}".format(TN/N*100))
print("Precision: {:.2f}".format(TP/(TP+FP)*100))
