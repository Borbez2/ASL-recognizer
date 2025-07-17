import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Use mixed precision (faster training on modern GPUs/Apple Silicon)
mixed_precision.set_global_policy('mixed_float16')

# Image and batch settings
batch_size = 32
img_height, img_width = 200, 200

# Basic data augmentation: adds variety so the model generalizes better
data_augment = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),
])

# Load training and validation sets (10% split)
# The folder must contain one subfolder per label (A-Z + space, del, nothing)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ASL Alphabet Archive/asl_alphabet_train/asl_alphabet_train',
    validation_split=0.1,
    subset='training',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'ASL Alphabet Archive/asl_alphabet_train/asl_alphabet_train',
    validation_split=0.1,
    subset='validation',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Convert images to float32 (EfficientNet wants this)
# Prefetch to keep GPU fed during training
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE)

# Build model using pretrained EfficientNetB0 (trained on ImageNet)
inputs = Input(shape=(img_height, img_width, 3))
x = data_augment(inputs)  # Apply augmentations only on training input
x = tf.keras.applications.efficientnet.preprocess_input(x)  # Normalize per EfficientNet expectations

# Load EfficientNet without top classification layers
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
base_model.trainable = False  # Freeze it — we’re only training the head for now

# Add new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce feature map to 1D vector
x = Dropout(0.5)(x)  # Dropout helps prevent overfitting
outputs = Dense(train_ds.element_spec[1].shape[-1], activation='softmax', dtype='float32')(x)  # 29 output classes

model = Model(inputs, outputs)

# Compile model with soft labels (label smoothing helps generalization)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks: early stopping, adaptive learning rate, and save the best model only
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('ASL_model_final.keras', save_best_only=True)  # Save model only when validation improves
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# Save final model again (just in case the last epoch was the best)
model.save('ASL_model_final.keras')

# Plot training vs. validation loss so you can visually spot overfitting or plateaus
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.grid(True)
plt.show()