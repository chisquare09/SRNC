from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np

def classification_rejection_v2(X, Y, Y_all_labels, X_train, Y_train,X_test,predictive_alg, threshold_rejection,sss=None):
    if predictive_alg == "svm":
        clf = svm.SVC(probability=True, max_iter=100000).fit(X_train, Y_train)
    if predictive_alg == "LinearSVM":
        svc = LinearSVC()
        clf = CalibratedClassifierCV(svc, cv=sss).fit(X_train, Y_train)
        # fix to 5 fold instead of 10 fold ?
    if predictive_alg == "lr":
        clf = LogisticRegression(max_iter=10000).fit(X_train, Y_train)
    if predictive_alg == "lda":
        clf = LinearDiscriminantAnalysis().fit(X_train, Y_train)
    if predictive_alg == "dt":
        clf = DecisionTreeClassifier().fit(X_train, Y_train)
    if predictive_alg == "lightGBM":
        clf = LGBMClassifier(verbose=-1).fit(X_train, Y_train)


    probs_max = [np.max(x) for x in clf.predict_proba(X_test)]
    Y_predict = list(clf.predict(X_test))
    Y_predict_rejection = [Y_predict[i] if probs_max[i] >= threshold_rejection else int(10*len(Y_all_labels) + 1)  for i in range(X_test.shape[0])]
    return  Y_predict_rejection