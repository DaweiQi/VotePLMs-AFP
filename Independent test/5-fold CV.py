import pickle
import math
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from keras.layers import Input, InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    Conv1D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, AveragePooling1D, MaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.python.keras.layers import multiply, add
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from xgboost import XGBClassifier as xgbc, XGBClassifier

# ProtT5 feature
with open("../Features/Pre-trained_features/30_pos.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_P = np.array([item for item in feature_dict.values()])
train_pre_P = np.insert(train_pre_P, 0, values=[1 for _ in range(train_pre_P.shape[0])], axis=1)
print("train_pre_P:", train_pre_P.shape)
with open("../Features/Pre-trained_features/30_neg.txt_t5.pkl", "rb") as tf:
    feature_dict = pickle.load(tf)
train_pre_N = np.array([item for item in feature_dict.values()])
train_pre_N = np.insert(train_pre_N, 0, values=[0 for _ in range(train_pre_N.shape[0])], axis=1)
print("train_pre_N:", train_pre_N.shape)
train_pre = np.row_stack((train_pre_P, train_pre_N))

# ESM-1b feature
train_esm1b_P = pd.read_csv('../Features/Feature_csv/30_pos_esm1b_change.csv')
train_esm1b__P = np.array(train_esm1b_P)
print("train_esm1b_P:", train_esm1b__P.shape)
train_esm1b_N = pd.read_csv('../Features/Feature_csv/30_neg_esm1b_change.csv')
train_esm1b_N = np.array(train_esm1b_N)
print("train_esm1b_N:", train_esm1b_N.shape)
train_esm1b = np.row_stack((train_esm1b_P, train_esm1b_N))
print("train_esm1b:", train_esm1b.shape)

# Tape feature
train_tape_P = pd.read_csv('../Features/Feature_csv/30_pos_tape_bert.csv')
train_tape_P = np.array(train_tape_P)
print("train_tape_P:", train_tape_P.shape)
train_tape_N = pd.read_csv('../Features/Feature_csv/30_neg_tape_bert.csv')
train_tape_N = np.array(train_tape_N)
print("train_tape_N:", train_tape_N.shape)
train_tape = np.row_stack((train_tape_P, train_tape_N))
train1 = np.hstack((train_esm1b, train_tape))
train = np.hstack((train_pre, train1))
print("train:", train.shape)
Y, X = train[:, 0], train[:, 1:]
print("label:", Y.shape)
print("ferature:", X.shape)

# Divide the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 按照比例划分数据集为训练集与测试集

clf = LogisticRegression(solver='liblinear', random_state=7)
# clf = VotingClassifier(
#     estimators=[('svc', SVC(probability=True, kernel='rbf')),
#                 ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, random_state=7)),
#                 ('xgb', xgbc(n_estimators=800, max_depth=5, random_state=7)),
#                 ('lr', LogisticRegression(solver='liblinear', random_state=7)),
#                 ('mlp', MLPClassifier(hidden_layer_sizes=[64, 32], max_iter=1000)),
#                 ],
#     voting='soft')
clf.fit(X_train, y_train)

Acc = []
Sen = []
Spe = []
Pre = []
Mcc = []
F_score = []
Auc = []

all_true_labels = []
all_pred_labels = []
all_pred_probabilities = []

# 5-fold CV
for i in range(10):
    print('The %d times 5-fold CV...' % i)
    cv = KFold(n_splits=5, shuffle=True)
    proba_y = []
    NBtest_index = []
    pre_y = []
    pro_y1 = []
    for train, test in cv.split(X):
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]
        NBtest_index.extend(test)
        proba_ = clf.fit(x_train, y_train).predict_proba(x_test)
        y_train_pre = clf.predict(x_test)
        y_train_proba = clf.predict_proba(x_test)
        proba_y.extend(y_train_proba[:, 1])
        pre_y.extend(y_train_pre)
        pro_y1.extend(y_test)
        cm = confusion_matrix(y_test, y_train_pre)
        # print(cm)
        TN, FP, FN, TP = cm.ravel()
        ACC = (TP + TN) / (TP + TN + FP + FN)
        SN = TP / (TP + FN)
        SP = TN / (TN + FP)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
        PR = TP / (TP + FP)
        F_SCORE = (2 * SN * PR) / (SN + PR)
        AUC = roc_auc_score(y_test, y_train_proba[:, 1])
        Acc.append(ACC)
        Sen.append(SN)
        Spe.append(SP)
        F_score.append(F_SCORE)
        Pre.append(PR)
        Mcc.append(MCC)
        Auc.append(AUC)

        all_true_labels.extend(y_test)
        all_pred_labels.extend(y_train_pre)
        all_pred_probabilities.extend(y_train_proba[:, 1])

# Calculate performance metrics
print('mean:')
print('meanAcc:', np.mean(Acc))
print('meanSen:', np.mean(Sen))
print('meanSpe:', np.mean(Spe))
print('meanF_score:', np.mean(F_score))
print('meanPre', np.mean(Pre))
print('meanMcc:', np.mean(Mcc))
print('meanAuc:', np.mean(Auc))

print('std:')
print('stdAcc:', np.std(Acc))
print('stdSen:', np.std(Sen))
print('stdSpe:', np.std(Spe))
print('stdF_score:', np.std(F_score))
print('stdPre', np.std(Pre))
print('stdMcc:', np.std(Mcc))
print('stdAuc:', np.std(Auc))

# Calculate and save ROC curve
fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probabilities)
roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
# roc_data.to_csv('30_roc_data.csv', index=False)
# print('ROC data has been saved to roc_data.csv')

# ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % np.mean(Auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()