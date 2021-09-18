#%%
import numpy as np
import numpy.random as npr
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(max_depth=16, min_samples_leaf=8, min_samples_split=8, n_estimators=100, random_state=0)
svm_model = svm.SVC(gamma=1.000, C=100.0)
# model3 = GridSearchCV(random_forest_model, param_grid=params, cv=3, refit=True)
encoder = LabelEncoder()

train = pd.read_csv('./will-vote-data/train.csv')
test_final = pd.read_csv('./will-vote-data/test_x.csv')
sub = pd.read_csv('./will-vote-data/sample_submission.csv')

columns = ['age_group', 'gender', 'race', 'religion']

for column in columns:
    encoder.fit(train[column])
    labels = encoder.transform(train[column])
    train[column] = labels

for column in columns:
    encoder.fit(test_final[column])
    labels = encoder.transform(test_final[column])
    test_final[column] = labels

drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
             'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
             'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
             'QpE', 'QqE', 'QrE', 'QsE', 'QtE', 
             'hand', 'voted', 'index', 'engnat', 
             'urban', 'wf_01', 'wf_02', 'wf_03', 
             'wr_01', 'wr_02', 'wr_03', 'wr_04', 
             'wr_05', 'wr_06', 'wr_07', 'wr_08', 
             'wr_09', 'wr_10', 'wr_11', 'wr_12', 
             'wr_13']

data = train.drop(drop_list, axis=1)
target = train['voted']

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=120)

def data_analysis(model):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(f'{model} 예측 정확도:', accuracy_score(Y_test, pred))
    ftr_importances_values = model.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
    ftr_top30 = ftr_importances.sort_values(ascending=False)[:30]
    plt.figure(figsize=(8,6))
    plt.title('Top 30 Feature Importances')
    sns.barplot(x=ftr_top30, y=ftr_top30.index)
    plt.show()

data_analysis(random_forest_model)
