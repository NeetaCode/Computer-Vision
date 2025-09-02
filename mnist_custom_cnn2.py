#CUSTOM CNN 2 for MNIST:
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import cv2

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize MNIST images to a smaller size (32x32) and convert to 3 channels
x_train_resized = [cv2.cvtColor(cv2.resize(x, (32, 32)), cv2.COLOR_GRAY2RGB) for x in x_train]
x_test_resized = [cv2.cvtColor(cv2.resize(x, (32, 32)), cv2.COLOR_GRAY2RGB) for x in x_test]

x_train_resized = tf.convert_to_tensor(x_train_resized, dtype=tf.float32)
x_test_resized = tf.convert_to_tensor(x_test_resized, dtype=tf.float32)

# Normalize the data to values between 0 and 1
x_train_resized /= 255.0
x_test_resized /= 255.0

# Create a simple CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Train the model
history = model.fit(x_train_resized, y_train, epochs=10, batch_size=32, validation_data=(x_test_resized, y_test))

# Plot the training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

plt.tight_layout()
plt.show()
