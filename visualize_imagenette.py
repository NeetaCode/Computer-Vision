import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the imagenette dataset
imagenet_dir = "imagenette/320px"  # Use the correct dataset name

dataset, info = tfds.load(imagenet_dir, with_info=True)

# Now you can work with the dataset as needed

# Visualize a few random samples from the training dataset
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))  # Create a single row of subplots

for i, example in enumerate(dataset['validation'].take(num_samples)):
    image, label = example['image'], example['label']
    ax = axes[i]
    ax.imshow(image)
    ax.set_title(f"Label: {label.numpy()}")
    ax.axis('off')

plt.show()
