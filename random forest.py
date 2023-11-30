import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import modin.pandas as pd
import modin.config as cfg
cfg.StorageFormat.put('hdk')
# 以下两行导入Intel® Extension for Scikit-learn库，调用patch函数加速。如果使用原生的scikit-learn, 注释这两行即可

from sklearnex import patch_sklearn

patch_sklearn()

  

from sklearn import config_context

from sklearn.metrics import mean_squared_error, r2_score, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sklearn.linear_model as lm
import time

dt_start = time.time()

  

# 训练该数据集需要大约30G内存，如果内存足够，使用该行代码读取所有数据

df = pd.read_csv('creditcard.csv')

  

# 如果内存有限，可以只读取部分数据集，如：

# df = pd.read_csv('ipums_education2income_1970-2010.csv.gz', nrows=5000000)

  

print("read_csv time: ", time.time() - dt_start)
dt_start = time.time()

  

# 预处理

df = df.drop(['Time'], axis=1)

  

# # 清洗无效数据

# df = df[df["INCTOT"] != 9999999]

# df = df[df["EDUC"] != -1]

# df = df[df["EDUCD"] != -1]

  

# # 根据通货膨胀调整收入

# df["INCTOT"] = df["INCTOT"] * df["CPI99"]

  

# for column in keep_cols:

#     df[column] = df[column].fillna(-1)

#     df[column] = df[column].astype("float64")

  

#设置目标列为EDUC，并从features中移除EDUC和CPI99

y = df["Class"]

X = df.drop(["Class"], axis=1)

  

# 数据标准化

scaler = StandardScaler()

X = scaler.fit_transform(X)

  

# 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=777)

  

# # 使用逻辑回归模型进行训练

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()

# model.fit(X_train, y_train)

  

# # 在测试集上进行预测

# y_pred = model.predict(X_test)

  
  

# # 划分训练集和测试集

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  

# 初始化随机森林模型

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

  

# 训练模型

rf_model.fit(X_train, y_train)

  

# 预测

y_pred = rf_model.predict(X_test)

  

# 评估模型

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

  

print("ETL time: ", time.time() - dt_start)

  

# 计算F1分数

f1 = f1_score(y_test, y_pred)

  

# 输出F1分数和推理时间

print("F1 Score:", f1)
