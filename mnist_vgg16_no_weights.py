# VGG16 Model pretrained with NO Weights for MNIST
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_resized = [cv2.cvtColor(cv2.resize(x, (48, 48)), cv2.COLOR_GRAY2RGB) for x in x_train]
x_test_resized = [cv2.cvtColor(cv2.resize(x, (48, 48)), cv2.COLOR_GRAY2RGB) for x in x_test]

x_train_resized = tf.convert_to_tensor(x_train_resized, dtype=tf.float32)
x_test_resized = tf.convert_to_tensor(x_test_resized, dtype=tf.float32)

x_train_resized /= 255.0
x_test_resized /= 255.0

base_model = VGG16(weights=None, include_top=False, input_shape=(48, 48, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

history = model.fit(x_train_resized, y_train, epochs=10, batch_size=32, validation_data=(x_test_resized, y_test))

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
