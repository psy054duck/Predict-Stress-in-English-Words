import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import helper
import submission

if __name__ == '__main__':
    # training_data = helper.read_data('./asset/training_data.txt')
    test_data = helper.read_data('./asset/tiny_test.txt')
    # X = submission.generate_feature_matrix(training_data)
    # Y = submission.generate_target_matrix(training_data).ravel()

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # clf = RandomForestClassifier(n_estimators=100)

    # clf.fit(X_train, Y_train)

    # prediction = clf.predict(X_test)
    # print(f1_score(Y_test, prediction, average='micro'))
    # # print(cross_val_score(clf, X, Y, cv=20))

    X = submission.generate_feature_matrix(test_data)
    print(X)