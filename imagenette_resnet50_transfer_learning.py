import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load and preprocess the Imagenette dataset using TensorFlow Datasets (TFDS)
(ds_train, ds_test), ds_info = tfds.load(
    'imagenette/full-size-v2',  # Use the Imagenette dataset
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True
)

# Resize the images to (224, 224, 3) for ResNet-50
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

ds_train = ds_train.map(preprocess_image).batch(32)
ds_test = ds_test.map(preprocess_image).batch(32)

# Create a ResNet-50 model with pre-trained weights (excluding top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for Imagenette classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for labels without one-hot encoding
              metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10

history = model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test
)

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
