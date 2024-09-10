import math
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier as xgbc, XGBClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# ProtT5 feature
with open("../Features/Pre-trained_features/positive.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_P = np.array([item for item in feature_dict.values()])
train_pre_P = np.insert(train_pre_P, 0, values=[1 for _ in range(train_pre_P.shape[0])], axis=1)
print("train_pre_P:", train_pre_P.shape)
with open("../Features/Pre-trained_features/negative.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_N = np.array([item for item in feature_dict.values()])
train_pre_N = np.insert(train_pre_N, 0, values=[0 for _ in range(train_pre_N.shape[0])], axis=1)
print("train_pre_N:", train_pre_N.shape)
train_pre = np.row_stack((train_pre_P, train_pre_N))

# ESM-1b feature
train_esm1b_P = pd.read_csv('../Features/Feature_csv/positive_esm1b.csv')
train_esm1b__P = np.array(train_esm1b_P)
print("train_esm1b_P:", train_esm1b__P.shape)
train_esm1b_N = pd.read_csv('../Features/Feature_csv/negative_esm1b.csv')
train_esm1b_N = np.array(train_esm1b_N)
print("train_esm1b_N:", train_esm1b_N.shape)
train_esm1b = np.row_stack((train_esm1b_P, train_esm1b_N))
print("train_esm1b:", train_esm1b.shape)

# Tape feature
train_tape_P = pd.read_csv('../Features/Feature_csv/positive_tape_bert.csv')
train_tape_P = np.array(train_tape_P)
print("train_tape_P:", train_tape_P.shape)
train_tape_N = pd.read_csv('../Features/Feature_csv/negative_tape_bert.csv')
train_tape_N = np.array(train_tape_N)
print("train_tape_N:", train_tape_N.shape)
train_tape = np.row_stack((train_tape_P, train_tape_N))
train1 = np.hstack((train_esm1b, train_tape))

train = np.hstack((train_pre, train1))
print("train:", train.shape)

# Divide features and labels
Y, X = train[:, 0], train[:, 1:]
# Divide the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Standardized dataset
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)  # normalize X to 0-1 range
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = VotingClassifier(
    estimators=[('svc', SVC(probability=True, kernel='rbf')),
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, random_state=7)),
                ('xgb', xgbc(n_estimators=800, max_depth=5, random_state=7)),
                ('lr', LogisticRegression(solver='liblinear', random_state=7)),
                ('knn', KNeighborsClassifier(n_neighbors=6)),
                ('mlp', MLPClassifier(hidden_layer_sizes=[64, 32], max_iter=1000)),
                ],
    voting='soft')

# clf = LogisticRegression(solver='liblinear', random_state=7)
clf.fit(X_train, y_train)

# The classifier selects the optimal hyperparameters
Acc = []
Sen = []
Spe = []
Mcc = []
Pre = []
F_score = []
# 10 times 10-fold CV
for i in range(1):
    print('The %d times 10-fold CV...' % i)
    cv = KFold(n_splits=10, shuffle=True)
    proba_y = []
    NBtest_index = []
    pre_y = []
    pro_y1 = []
    for train, test in cv.split(X):  # train test  是下标
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        NBtest_index.extend(test)
        proba_ = clf.fit(x_train, y_train).predict_proba(x_test)
        y_train_pre = clf.predict(x_test)
        y_train_proba = clf.predict_proba(x_test)
        proba_y.extend(y_train_proba[:, 1])
        pre_y.extend(y_train_pre)
        pro_y1.extend(y_test)
        cm = confusion_matrix(pro_y1, pre_y)
        # print(cm)
        TN, FP, FN, TP = cm.ravel()
        ACC = (TP + TN) / (TP + TN + FP + FN)
        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
        PR = TP / (TP + FP)
        F_SCORE = (2 * SN * PR) / (SN + PR)
        Acc.append(ACC)
        Sen.append(SN)
        Spe.append(SP)
        F_score.append(F_SCORE)
        Pre.append(PR)
        Mcc.append(MCC)
print('10-fold CV mean:')
print('meanAcc:', np.mean(Acc))
print('meanSen:', np.mean(Sen))
print('meanSpe:', np.mean(Spe))
print('meanF_score:', np.mean(F_score))
print('meanPre', np.mean(Pre))
print('meanMcc:', np.mean(Mcc))
print('10-fold CV std:')
print('stdAcc:', np.std(Acc))
print('stdSen:', np.std(Sen))
print('stdSpe:', np.std(Spe))
print('stdF_score:', np.std(F_score))
print('stdPre', np.std(Pre))
print('stdMcc:', np.std(Mcc))

# ProtT5 feature
with open("../Features/Pre-trained_features/positive_t.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_t5_P = np.array([item for item in feature_dict.values()])
test_t5_P = np.insert(test_t5_P, 0, values=[1 for _ in range(test_t5_P.shape[0])], axis=1)
print("test_t5_P:", test_t5_P.shape)
with open("../Features/Pre-trained_features/negative_t.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
test_t5_N = np.array([item for item in feature_dict.values()])
test_t5_N = np.insert(test_t5_N, 0, values=[0 for _ in range(test_t5_N.shape[0])], axis=1)
print("test_t5_N:", test_t5_N.shape)
test_t5 = np.row_stack((test_t5_P, test_t5_N))

# ESM-1b feature
test_esm1b_P = pd.read_csv('../Features/Feature_csv/positive_t_esm1b.csv')
test_esm1b_P = np.array(test_esm1b_P)
print("test_esm1b_P:", test_esm1b_P.shape)
test_esm1b_N = pd.read_csv('../Features/Feature_csv/negative_t_esm1b.csv')
test_esm1b_N = np.array(test_esm1b_N)
print("test_esm1b_N:", test_esm1b_N.shape)
test_esm1b = np.row_stack((test_esm1b_P, test_esm1b_N))
print("test_esm1b:", test_esm1b.shape)

# Tape feature
test_tape_P = pd.read_csv('../Features/Feature_csv/positive_t_tape_bert.csv')
test_tape_P = np.array(test_tape_P)
print("test_tape_P:", test_tape_P.shape)
test_tape_N = pd.read_csv('../Features/Feature_csv/negative_t_tape_bert.csv')
test_tape_N = np.array(test_tape_N)
print("test_tape_N:", test_tape_N.shape)
test_tape = np.row_stack((test_tape_P, test_tape_N))
test1 = np.hstack((test_esm1b, test_tape))
test = np.hstack((test_t5, test1))

# Divide features and labels
test_y, test_x = test[:, 0], test[:, 1:]
print("label:", test_y.shape)
print("feature:", test_x.shape)

y_test_pre = clf.predict(test_x)
cm = confusion_matrix(test_y, y_test_pre)
# print(cm)
TN, FP, FN, TP = cm.ravel()
ACC = (TP + TN) / (TP + TN + FP + FN)
SN = TP / (TP + FN)
SP = TN / (TN + FP)
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
PR = TP / (TP + FP)
F_score = (2 * SN * PR) / (SN + PR)

print('Acc:', ACC)
print('Sen:', SN)
print('Spe:', SP)
print('F-score:', F_score)
print('Pre:', PR)
print('Mcc:', MCC)


