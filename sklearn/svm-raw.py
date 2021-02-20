from sklearn import svm
import numpy as np
from math import ceil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve


np.random.seed(10)

data = np.load('../data/aae-raw-data.npy').tolist()['raw']
data = np.array(data)
X = data[:,:-1]
y = data[:,-1]

skf = StratifiedKFold(n_splits=5, shuffle=True) # stratified 10-fold cross-validation

accs = []
sens = []
specs = []
precs = []
f1 = []

for train_index, test_index in skf.split(X, y):
    X_train , X_test , y_train , y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    clf = svm.SVC(C=1.5, kernel='linear', tol=0.001, cache_size=200, class_weight=None, probability=True)

    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)[:, 1]
    val_size = len(probas)//2
    prec, rec, thresholds = precision_recall_curve(y_test[:val_size], probas[:val_size])
    best_thr = thresholds[np.argmax(2/(1/(prec+1e-10) + 1/(rec+1e-10)))]

    cfm = confusion_matrix(
        y_test[val_size:],
        (probas[val_size:] >= best_thr).astype(int))
    TN = cfm[0][0]
    FP = cfm[0][1]
    FN = cfm[1][0]
    TP = cfm[1][1]
    P = TP + FN
    N = FP + TN
    accs.append(((TP+FP)/(TP+FP+TN+FN))*100)
    sens.append(TP/P*100)
    specs.append(TN/N*100)
    precs.append(TP/(TP+FP)*100)
    f1.append(2*TP/(2*TP+FN+FP)*100)
print("Accuracy: {:.2f}".format(np.mean(accs)))
print("Sensitivity: {:.2f}".format(np.mean(sens)))
print("Specificity: {:.2f}".format(np.mean(specs)))
print("Precision: {:.2f}".format(np.mean(precs)))
print("F1-score: {:.2f}".format(np.mean(f1)))
