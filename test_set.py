import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_csv("mnist_test.csv")
X = df.drop("label", axis=1).values / 255.0
y = to_categorical(df["label"].values, 10)
X = X.reshape(-1, 28, 28, 1)

# Build CNN model
model = Sequential([
 
Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and save model
model.fit(X, y, epochs=5, batch_size=128, validation_split=0.2)
model.save("mnist_cnn_model.h5")
