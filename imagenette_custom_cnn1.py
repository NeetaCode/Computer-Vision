#Custom CNN 1 for Imagenette

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess the Imagenette dataset
#(x_train, y_train), (x_test, y_test) = imagenette.load_data()

dataset_name = "imagenette/320px"
(ds_train, ds_test), ds_info = tfds.load(
    name=dataset_name,

    split=["train[:90%]", "train[90%:]"],
    as_supervised=True,
    with_info=True,
)

# Define a function to preprocess and resize images
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Apply preprocessing to the dataset
ds_train = ds_train.map(preprocess_image).batch(32).shuffle(buffer_size=1000)
ds_test = ds_test.map(preprocess_image).batch(32)

def create_custom_cnn(input_shape, num_classes):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Example usage:
input_shape = (224, 224, 3)  # Specify the input shape for Imagenette
num_classes = 10  # Number of output classes

model = create_custom_cnn(input_shape, num_classes)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(ds_train, epochs=5, validation_data=ds_test)

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
