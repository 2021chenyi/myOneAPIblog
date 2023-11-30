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
