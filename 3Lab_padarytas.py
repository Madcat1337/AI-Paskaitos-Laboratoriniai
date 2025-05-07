import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.io import loadmat
import random
import urllib.request
import tarfile

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 1. Prepare the programming environment
start_time = time.time()
print(f"Starting environment setup at {time.strftime('%H:%M:%S')}...")

# Verify TensorFlow and GPU
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected:", physical_devices)
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. Falling back to CPU.")
    print("Ensure T4 GPU is selected: Runtime > Change runtime type > T4 GPU.")

print(f"Environment setup completed in {time.time() - start_time:.2f} seconds.")

# 2. Get the training data ready
start_time = time.time()
print(f"\nStarting dataset preparation at {time.strftime('%H:%M:%S')}...")

# Download and extract Oxford 102 Flowers dataset
DATASET_PATH = "/content/flowers/"
os.makedirs(DATASET_PATH, exist_ok=True)

print("Downloading dataset...")
image_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
labels_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
urllib.request.urlretrieve(image_url, os.path.join(DATASET_PATH, "102flowers.tgz"))
urllib.request.urlretrieve(labels_url, os.path.join(DATASET_PATH, "imagelabels.mat"))

print("Extracting images...")
with tarfile.open(os.path.join(DATASET_PATH, "102flowers.tgz"), "r:gz") as tar:
    tar.extractall(DATASET_PATH)

IMAGE_PATH = os.path.join(DATASET_PATH, "jpg/")
LABELS_FILE = os.path.join(DATASET_PATH, "imagelabels.mat")

# Load labels
labels_mat = loadmat(LABELS_FILE)
labels = labels_mat['labels'][0]

# Select top 5 classes with highest image counts
NUM_CLASSES = 5
class_counts = pd.Series(labels).value_counts()
top_classes = class_counts.head(NUM_CLASSES).index.tolist()
print(f"Selected classes: {top_classes}")

# Filter images and labels for selected classes
image_files = [f"image_{str(i).zfill(5)}.jpg" for i in range(1, len(labels) + 1)]
selected_images = []
selected_labels = []
for idx, label in enumerate(labels):
    if label in top_classes:
        selected_images.append(image_files[idx])
        selected_labels.append(label)

# Map labels to 0-based indices
label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(set(top_classes)))}
selected_labels = [label_map[label] for label in selected_labels]

# Split data: 60% train, 20% validation, 20% test
print("Splitting data...")
train_images, test_images, train_labels, test_labels = train_test_split(
    selected_images, selected_labels, test_size=0.2, stratify=selected_labels, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.25, stratify=train_labels, random_state=42
)

# Image preprocessing parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Load and preprocess images
def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(IMAGE_PATH + image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0  # Normalize to [0,1]
    return img, label

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomBrightness(0.2),
    layers.RandomCrop(IMG_HEIGHT, IMG_WIDTH),
    layers.RandomContrast(0.2),
])

# Create datasets
print("Creating datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dataset preparation completed in {time.time() - start_time:.2f} seconds.")

# 3. Create SDNT model (with Input layer to fix warning)
def create_sdnt_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

print("\nBuilding SDNT model...")
start_time = time.time()
sdnt_model = create_sdnt_model()
sdnt_model.summary()
print(f"SDNT model creation completed in {time.time() - start_time:.2f} seconds.")

# 4. Train SDNT model
start_time = time.time()
print(f"\nStarting SDNT training at {time.strftime('%H:%M:%S')}...")

sdnt_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_sdnt_best = tf.keras.callbacks.ModelCheckpoint(
    "/content/sdnt_best_model.keras", monitor='val_loss', save_best_only=True, mode='min'
)
checkpoint_sdnt_last = tf.keras.callbacks.ModelCheckpoint(
    "/content/sdnt_last_model.keras", monitor='val_loss', save_best_only=False, mode='min'
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train
EPOCHS = 20
sdnt_history = sdnt_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_sdnt_best, checkpoint_sdnt_last, early_stopping],
    verbose=1
)

print(f"SDNT training completed in {time.time() - start_time:.2f} seconds.")

# 5. Create and train VGG16 model (Transfer Learning)
print("\nBuilding VGG16 model for transfer learning...")
start_time = time.time()

# Load VGG16 base
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
vgg_base.trainable = False  # Freeze the base

# Build model
vgg_model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    vgg_base,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

vgg_model.summary()
print(f"VGG16 model creation completed in {time.time() - start_time:.2f} seconds.")

# Train VGG16 (transfer learning)
start_time = time.time()
print(f"\nStarting VGG16 transfer learning at {time.strftime('%H:%M:%S')}...")

vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_vgg_best = tf.keras.callbacks.ModelCheckpoint(
    "/content/vgg_best_model.keras", monitor='val_loss', save_best_only=True, mode='min'
)
checkpoint_vgg_last = tf.keras.callbacks.ModelCheckpoint(
    "/content/vgg_last_model.keras", monitor='val_loss', save_best_only=False, mode='min'
)

vgg_history = vgg_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_vgg_best, checkpoint_vgg_last, early_stopping],
    verbose=1
)

print(f"VGG16 transfer learning completed in {time.time() - start_time:.2f} seconds.")

# 6. Fine-tune VGG16
print("\nStarting VGG16 fine-tuning...")
start_time = time.time()

# Unfreeze last convolutional block (block5)
for layer in vgg_base.layers:
    if 'block5' in layer.name:
        layer.trainable = True

vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
vgg_fine_history = vgg_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=fine_tune_epochs,
    callbacks=[checkpoint_vgg_best, checkpoint_vgg_last, early_stopping],
    verbose=1
)

print(f"VGG16 fine-tuning completed in {time.time() - start_time:.2f} seconds.")

# 7. Evaluate both models
def evaluate_model(model, model_name, history, test_dataset, output_dir):
    print(f"\nEvaluating {model_name}...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    # Handle history (History object or dict)
    if isinstance(history, dict):
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_metrics.png"))
    plt.show()

    # Predictions
    test_images_np = np.concatenate([x for x, y in test_dataset], axis=0)
    test_labels_np = np.concatenate([y for x, y in test_dataset], axis=0)
    predictions = model.predict(test_dataset)
    predicted_classes = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(test_labels_np, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.show()

    # Classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(test_labels_np, predicted_classes, target_names=[f"Class {i}" for i in range(NUM_CLASSES)], zero_division=0))

    # Visualize predictions
    num_samples = 5
    sample_indices = random.sample(range(len(test_images_np)), num_samples)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_images_np[idx])
        plt.title(f"True: {test_labels_np[idx]}\nPred: {predicted_classes[idx]}")
        plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{model_name}_sample_predictions.png"))
    plt.show()

# Create output directory
OUTPUT_DIR = "/content/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Evaluate SDNT model
print("Loading best SDNT model...")
sdnt_best_model = tf.keras.models.load_model("/content/sdnt_best_model.keras")
evaluate_model(sdnt_best_model, "SDNT", sdnt_history, test_dataset, OUTPUT_DIR)

# Evaluate VGG16 model (combine transfer learning and fine-tuning history)
print("Loading best VGG16 model...")
vgg_best_model = tf.keras.models.load_model("/content/vgg_best_model.keras")
vgg_combined_history = {
    'accuracy': vgg_history.history['accuracy'] + vgg_fine_history.history['accuracy'],
    'val_accuracy': vgg_history.history['val_accuracy'] + vgg_fine_history.history['val_accuracy'],
    'loss': vgg_history.history['loss'] + vgg_fine_history.history['loss'],
    'val_loss': vgg_history.history['val_loss'] + vgg_fine_history.history['val_loss']
}
evaluate_model(vgg_best_model, "VGG16", vgg_combined_history, test_dataset, OUTPUT_DIR)

# 8. Compare models
print("\nModel Comparison:")
sdnt_test_accuracy = sdnt_best_model.evaluate(test_dataset)[1]
vgg_test_accuracy = vgg_best_model.evaluate(test_dataset)[1]
print(f"SDNT Test Accuracy: {sdnt_test_accuracy:.4f}")
print(f"VGG16 Test Accuracy: {vgg_test_accuracy:.4f}")

print(f"Evaluation and comparison completed in {time.time() - start_time:.2f} seconds.")