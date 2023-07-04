import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
rf_clf = RandomForestClassifier(n_estimators=31, max_depth=8, max_features=3, random_state=0)
pipe_rf_clf = make_pipeline(log_transformer, StandardScaler(), rf_clf)
pipe_rf_clf.fit(X_train, y_train)
y_test_pred = pipe_rf_clf.predict(X_test)
y_train_pred = pipe_rf_clf.predict(X_train)
# print('训练集准确率:%.3f' % pipe_svm_clf.score(X_train, y_train))
# print('测试集准确率:%.3f' % pipe_svm_clf.score(X_test, y_test))
joblib.dump(pipe_rf_clf, 'RF_Sphalerite Classifier.pkl')