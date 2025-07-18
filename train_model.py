import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Enable mixed precision for performance
mixed_precision.set_global_policy('mixed_float16')

# Load landmark CSV
df = pd.read_csv('asl_landmarks_dataset.csv')

# Separate features (X) and labels (y)
X = df.drop('label', axis=1).values.astype('float32')
y = df['label'].values

# Encode labels to one-hot
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Split data into training/validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.1, random_state=42, stratify=y_onehot
)

# Build simple MLP model
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_onehot.shape[1], activation='softmax', dtype='float32')  # Output layer for 29 classes
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Set up callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('ASL_landmark_model.keras', save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Save final model
model.save('ASL_landmark_model.keras')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.grid(True)
plt.show()
