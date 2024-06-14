from numpy.random import seed
seed(123)
import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
import scipy
import tensorflow as tf
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Layer, InputSpec, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, activations, initializers, constraints
from keras import backend as K
from keras.constraints import UnitNorm, Constraint
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import RandomNormal
from sklearn.datasets import make_circles
np.set_printoptions(suppress=True)

#Import data
raw_data = pd.read_csv('data.csv')
display(raw_data.head())
print(raw_data.above_10.value_counts())

# First, split the data into train and temp (temporary) sets
x_train, x_temp, y_train, y_temp = train_test_split(
    raw_data.drop('above_10', axis=1),
    raw_data.above_10,
    test_size=0.30,  # 30% of the data will be used for validation and test sets combined
    stratify=raw_data.above_10,
    random_state=42  # for reproducibility
)

# Now, split the temp set into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,  # 50% of the temp set for validation and 50% for test
    stratify=y_temp,
    random_state=42  # for reproducibility
)

# Standardize the numeric columns using MinMaxScaler
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
# Split data into negative and positive labels
positives = x_train[y_train == 1]
negatives = x_train[y_train == 0]

# Input layer
input_layer = Input(shape=negatives.shape[1:])
# Hidden layer 
encoded = Dense(10, activation='relu', name="2")(input_layer)
# Output layer
output_layer = Dense(negatives.shape[1], activation='sigmoid', name="3")(encoded)
#Fit the autoencoder
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")
history = autoencoder.fit(negatives, negatives, batch_size = 15, epochs = 10000, shuffle = True,validation_data=(x_val, x_val))
autoencoder.summary()
#plot validation loss and normal loss
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#Now use only the encoding part to transform the data
hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
x_train_transformed = hidden_representation.predict(x_train)
x_test_transformed = hidden_representation.predict(x_test)

#Create prediction ANN
predictor = Sequential()
predictor.add(Dense(3, activation='relu', input_shape=x_train_transformed.shape[1:], name="2"))
predictor.add(Dense(1, activation='sigmoid', name="3"))
predictor.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
history = predictor.fit(x_train_transformed, y_train, batch_size=15, epochs=1000, shuffle=True)

y_predict = predictor.predict(x_test_transformed)
#Print statistics
print(classification_report(y_test, y_predict >= 0.5))
predictor.summary()

#Print loss and accuracy graphs
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import sklearn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import decomposition
#Print reconstruction errors
train_predictions = autoencoder.predict(x_train)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(x_train, train_predictions))
test_predictions = autoencoder.predict(x_test)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(x_test, test_predictions))

# Gradient
# This is for plotting the gradient, sd and loss graphs for autoencoder
optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_model(X, y, model, n_epochs=10000, batch_size=15):
    "Run training loop manually"
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    gradhistory = []
    losshistory = []
    def recordweight():
        data = {}
        for g,w in zip(grads, model.trainable_weights):
            if '/kernel:' not in w.name:
                continue # skip bias
            name = w.name.split("/")[0]
            data[name] = g.numpy()
        gradhistory.append(data)
        losshistory.append(loss_value.numpy())
    for epoch in range(n_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, y_pred)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step == 0:
                recordweight()
    # After all epochs, record again
    recordweight()
    return gradhistory, losshistory

gradhistory, losshistory = train_model(negatives, negatives, autoencoder)

def plot_gradient(gradhistory, losshistory):
    "Plot gradient mean and sd across epochs"
    fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 12))
    ax[0].set_title("Mean gradient")
    for key in gradhistory[0]:
        ax[0].plot(range(len(gradhistory)), [w[key].mean() for w in gradhistory], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in gradhistory[0]:
        ax[1].semilogy(range(len(gradhistory)), [w[key].std() for w in gradhistory], label=key)
    ax[1].legend()
    ax[2].set_title("Loss")
    ax[2].plot(range(len(losshistory)), losshistory)
    plt.show()

plot_gradient(gradhistory, losshistory)


print(negatives.shape[1:])


# Gradient
# This is for plotting the gradient, sd and loss graphs for the prediction ANN model
optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_model(X, y, model, n_epochs=200, batch_size=10):
    "Run training loop manually"
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    gradhistory = []
    losshistory = []
    def recordweight():
        data = {}
        for g,w in zip(grads, model.trainable_weights):
            if '/kernel:' not in w.name:
                continue # skip bias
            name = w.name.split("/")[0]
            data[name] = g.numpy()
        gradhistory.append(data)
        losshistory.append(loss_value.numpy())
    for epoch in range(n_epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, y_pred)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step == 0:
                recordweight()
    # After all epochs, record again
    recordweight()
    return gradhistory, losshistory

gradhistory, losshistory = train_model(x_train_transformed, y_train, predictor)

def plot_gradient(gradhistory, losshistory):
    "Plot gradient mean and sd across epochs"
    fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 12))
    ax[0].set_title("Mean gradient")
    for key in gradhistory[0]:
        ax[0].plot(range(len(gradhistory)), [w[key].mean() for w in gradhistory], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in gradhistory[0]:
        ax[1].semilogy(range(len(gradhistory)), [w[key].std() for w in gradhistory], label=key)
    ax[1].legend()
    ax[2].set_title("Loss")
    ax[2].plot(range(len(losshistory)), losshistory)
    plt.show()

plot_gradient(gradhistory, losshistory)


print(negatives.shape[1:])

#ROC curve
from sklearn.metrics import roc_curve
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, y_predict, pos_label=1)
# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score
# auc scores
auc_score1 = roc_auc_score(y_test, y_predict)

print(auc_score1)
# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='-',color='orange')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


