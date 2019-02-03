"""
    Title: Keras Convnet CIFAR-100 fruits
    Description: Deep convolutional neural network created with keras to classify fruits of CIFAR-100 dataset
    Author:      Pascal Schlaak
    Date:        02/03/2019
    Python:      3.6.7
"""

import numpy as np

from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split


# Load train- and test-data from CIFAR-100-dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Get indices where class-label is 0 (apple), 53 (orange), 57 (pear) to keep fruits from dataset
indices_train = np.where((y_train == 0) | (y_train == 53) | (y_train == 57))[0]
indices_test = np.where((y_test == 0) | (y_test == 53) | (y_test == 57))[0]

# Reduce class labels
y_train = np.array(y_train[indices_train])
y_test = np.array(y_test[indices_test])
# Reduce train- and test-data
x_train = x_train[indices_train]
x_test = x_test[indices_test]

# Replace current class labels (0, 53, 57) to (0, 1, 2); Needed for one-hot encoding
y_train = np.array(list(map(lambda i: [1] if i == 53 else ([2] if i == 57 else [0]), y_train)))
y_test = np.array(list(map(lambda i: [1] if i == 53 else ([2] if i == 57 else [0]), y_test)))

# Convert class labels to one-hot labels
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# Cast train- and test-data to float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize data from [0;255] to [0;1]
x_train /= 255
x_test /= 255

# Split train-data for validation (size of validation-data equals size of test-data)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20)

# Create linear stack of layers
model = Sequential()

# Add first convolution
model.add(
    Conv2D(filters=32,
           kernel_size=3,
           activation='relu',
           data_format="channels_last",
           input_shape=(32, 32, 3),
           strides=1,
           padding="valid",
           kernel_initializer="he_uniform",
           name="h1_conv"
           )
)

# Add first max-pooling
model.add(
    MaxPooling2D(
        pool_size=2,
        name="h1_pool"
    )
)

# Add first dropout
model.add(
    Dropout(
        rate=0.2,
        name="h1_drop"
    )
)

# Add second convolution
model.add(
    Conv2D(filters=32,
           kernel_size=3,
           activation='relu',
           strides=1,
           padding="valid",
           kernel_initializer="he_uniform",
           name="h2_conv"
           )
)

# Add second max-pooling
model.add(
    MaxPooling2D(
        pool_size=2,
        name="h2_pool"
    )
)

# Add second dropout
model.add(
    Dropout(
        rate=0.3,
        name="h2_drop"
    )
)

# Add third convolution
model.add(
    Conv2D(filters=32,
           kernel_size=3,
           activation='relu',
           strides=1,
           padding="valid",
           kernel_initializer="he_uniform",
           name="h3_conv"
           )
)

# Add third max-pooling
model.add(
    MaxPooling2D(
        pool_size=2,
        name="h3_pool"
    )
)

# Add third dropout
model.add(
    Dropout(
        rate=0.4,
        name="h3_drop"
    )
)

# Reduce dimensionality
model.add(
    Flatten()
)

# Add fully-connected layer
model.add(
    Dense(
        units=1024,
        activation='relu',
        name="h4_dense"
    )
)

# Add fully-connected output layer
model.add(
    Dense(
        units=3,
        activation='softmax',
        name="out"
    )
)

# Configure training parameters
model.compile(
    loss='mean_squared_error',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

# Fit model on train- and validation-data
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=50,
    validation_data=(x_valid, y_valid)
)

# Evaluate unseen test-data
score = model.evaluate(
    x=x_test,
    y=y_test
)

# Print test classification results
print("Test-accuracy: " + str(round(score[1], 3) * 100) + "%\n" + "Test-loss: " + str((score[0])))

# Save model to file
model.save("model.h5")
