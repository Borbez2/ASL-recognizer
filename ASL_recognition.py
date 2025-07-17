import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import mediapipe as mp
import os

# Load the trained ASL model
model = tf.keras.models.load_model('ASL_model_final.keras')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Global state
running = False
last_prediction_time = 0
prediction_interval = 3  # seconds - reduced for better responsiveness
cap = None
is_flashing = False
flash_start_time = 0
flash_duration = 0.3  # seconds - how long the flash effect lasts

# GUI setup
root = tk.Tk()
root.title("ASL Real-Time Recognition")
root.geometry("1000x720")
root.configure(bg="#f0f0f0")

# Create a main title
title_label = tk.Label(root, text="ASL Real-Time Recognition", font=("Helvetica", 20, "bold"), 
                      bg="#f0f0f0", fg="#333333", pady=10)
title_label.pack(fill="x")

# Create main frame for content
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Left frame for video feed
left_frame = tk.Frame(main_frame, bg="#f0f0f0")
left_frame.pack(side="left", padx=10, fill="both", expand=True)

# Camera display with border
video_frame = tk.Frame(left_frame, bd=2, relief="ridge", bg="#ffffff")
video_frame.pack(pady=10)
video_label = tk.Label(video_frame, bg="#000000")
video_label.pack(padx=2, pady=2)

# Progress bar for next prediction
progress_frame = tk.Frame(left_frame, bg="#f0f0f0")
progress_frame.pack(pady=5, fill="x")
progress_label = tk.Label(progress_frame, text="Next capture in:", font=("Helvetica", 10), bg="#f0f0f0")
progress_label.pack(side="left", padx=5)
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(side="left", padx=5, fill="x", expand=True)

# Right frame for reference chart
right_frame = tk.Frame(main_frame, bg="#f0f0f0")
right_frame.pack(side="right", padx=10, fill="both", expand=True)

# Load and display reference chart
if os.path.exists("asl_reference_chart.png"):
    ref_img = Image.open("asl_reference_chart.png")
    ref_img = ref_img.resize((400, 350))
    ref_photo = ImageTk.PhotoImage(ref_img)
    ref_label = tk.Label(right_frame, image=ref_photo, bd=2, relief="ridge")
    ref_label.image = ref_photo  # Keep a reference
    ref_label.pack(pady=10)
else:
    ref_label = tk.Label(right_frame, text="ASL Reference Chart\n(Image not found)", 
                        font=("Helvetica", 14), height=10, width=30, bd=2, relief="ridge")
    ref_label.pack(pady=10)

# Bottom frame for text and buttons
bottom_frame = tk.Frame(root, bg="#f0f0f0")
bottom_frame.pack(fill="x", padx=20, pady=10)

# Prediction text output with label
text_label = tk.Label(bottom_frame, text="Recognized Signs:", font=("Helvetica", 12, "bold"), 
                     bg="#f0f0f0", anchor="w")
text_label.pack(fill="x", pady=(0, 5))

text_box = tk.Text(bottom_frame, height=3, width=60, font=("Helvetica", 14), 
                  bd=2, relief="ridge", bg="#ffffff")
text_box.pack(fill="x", pady=5)

# Button frame
button_frame = tk.Frame(bottom_frame, bg="#f0f0f0")
button_frame.pack(pady=10)

def clear_text():
    text_box.delete(1.0, tk.END)

def copy_text():
    root.clipboard_clear()
    root.clipboard_append(text_box.get(1.0, tk.END))
    messagebox.showinfo("Copied", "Predicted text copied to clipboard!")

def toggle_running():
    global running, cap
    running = not running
    toggle_btn.config(text="Stop" if running else "Start")
    if running:
        cap = cv2.VideoCapture(0)
        root.after(0, video_loop)
    else:
        if cap is not None:
            cap.release()
            cap = None
        video_label.config(image='')


# Create styled buttons
button_style = {"font": ("Helvetica", 12), "width": 10, "bd": 2, "relief": "raised", 
               "bg": "#4a7abc", "fg": "black", "padx": 10, "pady": 5}

toggle_btn = tk.Button(button_frame, text="Start", command=toggle_running, **button_style)
toggle_btn.grid(row=0, column=0, padx=15)

tk.Button(button_frame, text="Clear", command=clear_text, **button_style).grid(row=0, column=1, padx=15)
tk.Button(button_frame, text="Copy", command=copy_text, **button_style).grid(row=0, column=2, padx=15)


def predict_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # Get hand bounding box from landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        xmin = max(int(min(x_coords) * w) - 20, 0)
        ymin = max(int(min(y_coords) * h) - 20, 0)
        xmax = min(int(max(x_coords) * w) + 20, w)
        ymax = min(int(max(y_coords) * h) + 20, h)

        # Crop hand region and resize
        hand_img = frame[ymin:ymax, xmin:xmax]
        if hand_img.size == 0:
            return None

        resized = cv2.resize(hand_img, (200, 200))
        normed = resized / 255.0
        reshaped = np.expand_dims(normed, axis=0).astype('float32')
        prediction = model.predict(reshaped, verbose=0)[0]
        return labels[np.argmax(prediction)]

    return None  # No hand detected

def video_loop():
    global last_prediction_time, cap, is_flashing, flash_start_time

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if cap is None or not ret:
        root.after(10, video_loop)
        return

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand tracking
    results = hands.process(rgb)

    # Resize for display - use 16:9 aspect ratio for better appearance
    display_frame = cv2.resize(frame, (480, 270))

    # Update progress bar
    current_time = time.time()
    time_since_last = current_time - last_prediction_time
    progress_percentage = min(100, (time_since_last / prediction_interval) * 100)
    progress_bar["value"] = int(progress_percentage)

    # Draw hand landmarks if detected
    if results is not None and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Create custom drawing specs for smaller markers
            custom_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            custom_connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            
            # Draw landmarks with smaller markers
            mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=custom_landmark_drawing_spec,
                connection_drawing_spec=custom_connection_drawing_spec
            )
            
            # Draw bounding box around the hand (matching the crop size used for model input)
            h, w, _ = display_frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            xmin = max(int(min(x_coords) * w) - 20, 0)
            ymin = max(int(min(y_coords) * h) - 20, 0)
            xmax = min(int(max(x_coords) * w) + 20, w)
            ymax = min(int(max(y_coords) * h) + 20, h)
            
            # Draw rectangle around the hand
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        # Predict when interval has passed
        if current_time - last_prediction_time >= prediction_interval:
            # Start flash effect
            is_flashing = True
            flash_start_time = current_time

            # Make prediction
            last_prediction_time = current_time
            pred_label = predict_frame(frame)
            if pred_label:
                text_box.insert(tk.END, pred_label + " ")
                text_box.see(tk.END)


    # Apply flash effect if active
    if is_flashing:
        # Calculate flash intensity based on time elapsed
        flash_time_elapsed = current_time - flash_start_time
        if flash_time_elapsed < flash_duration:
            # Create flash effect with fading intensity
            flash_intensity = 1.0 - (flash_time_elapsed / flash_duration)
            flash_frame = cv2.addWeighted(
                display_frame,
                1.0 - (flash_intensity * 0.5),  # Reduce original image intensity
                np.ones_like(display_frame) * 255,  # White overlay
                flash_intensity * 0.5,  # Flash intensity
                0
            )
            display_frame = flash_frame
        else:
            # End flash effect
            is_flashing = False

    # Add a border and convert to RGB
    display_frame = cv2.copyMakeBorder(display_frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    # Create PhotoImage and update label
    imgtk: ImageTk.PhotoImage = ImageTk.PhotoImage(image=img)  # type hint helps PyCharm
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, video_loop)

def on_closing():
    global running, cap
    running = False
    if cap:
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()