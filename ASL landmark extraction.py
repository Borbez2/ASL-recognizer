import os
import cv2
import mediapipe as mp
import numpy as np
import csv
from pathlib import Path

# Init MediaPipe for static image landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Store final data rows
output = []

# Set dataset path (adjust this if needed)
base_dir = Path(__file__).parent
dataset_dir = base_dir / "ASL Alphabet Archive" / "asl_alphabet_train" / "asl_alphabet_train"

# Live progress tracking
sample_count = 0

# Loop through A-Z, space, del, nothing (29 labels total)
for label in sorted(os.listdir(str(dataset_dir))):
    label_dir = dataset_dir / label
    if not label_dir.is_dir():
        continue

    print(f"Processing label: {label}")

    for img_file in os.listdir(str(label_dir)):
        if not (img_file.endswith(".jpg") or img_file.endswith(".png")):
            continue

        img_path = label_dir / img_file
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            # Flatten x, y, z for each of the 21 points
            flattened = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
            flattened.append(label)
            output.append(flattened)
            sample_count += 1

            # Print every 500 samples to track progress
            if sample_count % 500 == 0:
                print(f"  → {sample_count} samples so far...")

# Define CSV headers
header = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
csv_path = base_dir / "asl_landmarks_dataset.csv"

# Save to file
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output)

print(f"\nDone: Extracted {sample_count} samples → {csv_path.name}")