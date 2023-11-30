# 前言
- 团队名称：CY
- 问题陈述：实现信用卡交易欺诈检测
- 主要使用了OneAPI的Intel® Extension for Scikit-learn库
- 主要学到了OneAPI的强大性能，对于加速确实有很大帮助。

```python
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")
# 使用原生pandas
# import pandas as pd

# 使用modin
import modin.pandas as pd

import modin.config as cfg
cfg.StorageFormat.put('hdk')
# 以下两行导入Intel® Extension for Scikit-learn库，调用patch函数加速。如果使用原生的scikit-learn, 注释这两行即可
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn import config_context
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# 设置目标列为Class，并从features中移除Class
y = df["Class"]
X = df.drop(["Class"], axis=1)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=777)

# 初始化决策树模型
dt_model = DecisionTreeClassifier(random_state=42)

# 训练模型
dt_model.fit(X_train, y_train)

# 预测
y_pred = dt_model.predict(X_test)

# 评估模型
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("ETL time: ", time.time() - dt_start)

# 计算F1分数
f1 = f1_score(y_test, y_pred)

# 输出F1分数和推理时间
print("F1 Score:", f1)

```
结果截图
![[Pasted image 20231122170736.png]]
这段代码是一个使用决策树（Decision Tree）的机器学习模型，用于对信用卡交易数据进行分类，判断是否为欺诈。下面是代码的主要解释：

1. **数据导入与预处理**：
   - 通过`pd.read_csv`读取信用卡交易数据。
   - 删除数据中的时间列（'Time'）。
   - 设置目标列为'Class'，表示交易是否为欺诈。
   - 对特征数据进行标准化，使用`StandardScaler`将数据转换成标准正态分布。

2. **训练集与测试集划分**：
   - 使用`train_test_split`函数将数据集划分为训练集（`X_train`, `y_train`）和测试集（`X_test`, `y_test`）。

3. **决策树模型的建立**：
   - 初始化决策树模型：`DecisionTreeClassifier(random_state=42)`。
   - 使用训练集对决策树模型进行训练：`dt_model.fit(X_train, y_train)`。

4. **模型评估**：
   - 使用测试集进行预测：`y_pred = dt_model.predict(X_test)`。
   - 输出混淆矩阵（`confusion_matrix`）、分类报告（`classification_report`）、准确度（`accuracy_score`）以评估模型性能。

5. **F1分数计算**：
   - 使用`f1_score`计算模型的F1分数，F1分数是精确率和召回率的调和平均值，对模型整体性能进行评估。

6. **输出结果**：
   - 打印混淆矩阵、分类报告、准确度和F1分数等模型评估结果。

总体而言，这段代码展示了如何使用决策树模型对信用卡交易数据进行分类，并评估模型性能。在实际应用中，你可能需要调整模型的超参数以提高性能，也可以考虑使用其他机器学习算法或集成方法进行比较。


```python
import os
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the environment variable to enable oneAPI optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('creditcard.csv')

# Preprocess data
df = df.drop(['Time'], axis=1)
y = df["Class"]
X = df.drop(["Class"], axis=1)

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=777)

# Define the neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model with oneAPI optimizations
model.compile(optimizer=tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
dt_start = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

print("Training time: ", time.time() - dt_start)

```
结果截图
![[Pasted image 20231122170811.png]]
此代码使用TensorFlow和Keras库创建了一个简单的前馈神经网络（feedforward neural network）模型，包含两个隐藏层和一些dropout层用于正则化。模型使用Adam优化器进行二分类任务的训练，并计算损失和准确度。最后，对模型在测试集上的性能进行了评估，包括混淆矩阵、分类报告、准确率和F1分数。



```python
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
```
结果截图
![[Pasted image 20231122170831.png]]
这段代码主要是一个机器学习模型的训练和评估过程，使用了Modin库加速数据读取和处理，以及Intel Extension for Scikit-learn库进行Scikit-learn的加速。以下是代码的解释：

1. **导入库和配置：**
   - `os`: 提供与操作系统交互的功能。
   - `numpy` 和 `warnings`: 数组操作和警告处理库。
   - `modin.pandas as pd`: 使用Modin库的并行化Pandas，加速数据处理。
   - `modin.config` 和 `cfg.StorageFormat.put('hdk')`: 配置Modin库使用HDD格式来存储数据。
   - `sklearnex` 和 `patch_sklearn()`: 导入Intel Extension for Scikit-learn库并调用`patch_sklearn()`函数以加速Scikit-learn。

2. **数据读取：**
   - 使用Modin库的`pd.read_csv`函数读取CSV文件（信用卡欺诈数据集）。
   - 通过设置存储格式为HDD来优化数据读取。

3. **数据预处理：**
   - 删除不需要的列（"Time"列）。
   - 设置目标列为"Class"，将其从特征中移除。
   - 对特征数据进行标准化，使用`StandardScaler`。

4. **训练集和测试集划分：**
   - 使用`train_test_split`函数将数据集划分为训练集和测试集。

5. **随机森林模型训练和评估：**
   - 初始化一个包含100个决策树的随机森林分类器（`RandomForestClassifier`）。
   - 训练模型使用`fit`方法。
   - 使用训练好的模型对测试集进行预测。
   - 输出混淆矩阵、分类报告和准确率来评估模型性能。

6. **F1分数计算和输出：**
   - 使用`f1_score`函数计算F1分数。
   - 打印F1分数和整体推理时间。
