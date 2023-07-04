import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class_names = np.array([
    "Sedex",
    "MVT",
    "VMS",
    "Skarn"
])
path_file = "sphalerite_iqr_initi_knn.xlsx"
data = pd.read_excel(path_file)
df = data.loc[:, ['type', 'Cd', 'Mn', 'Ag', 'Cu', 'Pb', 'Sn', 'Ga', 'In', 'Sb', 'Co', 'Ge', 'Fe']]
for col_i in range(1, 13):
    df.iloc[:, col_i] = pd.to_numeric(df.iloc[:, col_i], errors="coerce")
X = df.copy(deep=True)
y_label = X.pop('type')
y_int, index = pd.factorize(y_label, sort=True)
y = y_int
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify = y)
log_transformer = FunctionTransformer(np.log, validate=True)
pipe_svm_clf = make_pipeline(log_transformer, StandardScaler(), SVC(kernel='rbf',C=1.0,gamma=0.25,cache_size=1000, class_weight=None, probability=True))
pipe_svm_clf.fit(X_train, y_train)
y_test_pred = pipe_svm_clf.predict(X_test)
y_train_pred = pipe_svm_clf.predict(X_train)
# print('训练集准确率:%.3f' % pipe_svm_clf.score(X_train, y_train))
# print('测试集准确率:%.3f' % pipe_svm_clf.score(X_test, y_test))
joblib.dump(pipe_svm_clf, 'SVM_Sphalerite Classifier.pkl')