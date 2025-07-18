import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import mediapipe as mp
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load model trained on 63 landmark coords
model = tf.keras.models.load_model('ASL_landmark_model.keras')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Global state
running = False
last_prediction_time = 0
prediction_interval = 3
cap = None
is_flashing = False
flash_start_time = 0
flash_duration = 0.3

# GUI setup
root = tk.Tk()
root.title("ASL Real-Time Recognition")
root.geometry("900x820")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="ASL Real-Time Recognition", font=("Helvetica", 20, "bold"),
                       bg="#f0f0f0", fg="#333333", pady=10)
title_label.pack(fill="x")

main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Left side: video feed + progress
left_frame = tk.Frame(main_frame, bg="#f0f0f0")
left_frame.pack(side="left", padx=10, fill="both", expand=True)

video_frame = tk.Frame(left_frame, bd=2, relief="ridge", bg="#ffffff")
video_frame.pack(pady=10)
video_label = tk.Label(video_frame, bg="#000000")
video_label.pack(padx=2, pady=2)

progress_frame = tk.Frame(left_frame, bg="#f0f0f0")
progress_frame.pack(pady=5, fill="x")
tk.Label(progress_frame, text="Next capture in:", font=("Helvetica", 10), bg="#f0f0f0").pack(side="left", padx=5)
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(side="left", padx=5, fill="x", expand=True)

# Right side: chart + reference
right_frame = tk.Frame(main_frame, bg="#f0f0f0")
right_frame.pack(side="right", padx=10, fill="both", expand=True)

# Chart frame
chart_frame = tk.Frame(right_frame, bg="#f0f0f0")
chart_frame.pack(pady=5)

fig, ax = plt.subplots(figsize=(4, 2))
bars = ax.bar(labels, [0]*len(labels), color="#4a7abc")
ax.set_ylim([0, 1])
ax.set_title("Prediction Confidence", fontsize=10)
ax.tick_params(axis='x', labelrotation=90, labelsize=7)
ax.set_ylabel("Confidence", fontsize=9)
fig.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=chart_frame)
canvas.get_tk_widget().pack()

def update_confidence_chart(pred_array):
    for bar, conf in zip(bars, pred_array):
        bar.set_height(conf)
    canvas.draw()

# Reference image
if os.path.exists("asl_reference_chart.png"):
    ref_img = Image.open("asl_reference_chart.png").resize((400, 350))
    ref_photo = ImageTk.PhotoImage(ref_img)
    ref_label = tk.Label(right_frame, image=ref_photo, bd=2, relief="ridge")
    ref_label.image = ref_photo
    ref_label.pack(pady=10)
else:
    ref_label = tk.Label(right_frame, text="ASL Reference Chart\n(Image not found)",
                         font=("Helvetica", 14), height=10, width=30, bd=2, relief="ridge")
    ref_label.pack(pady=10)

# Bottom frame: prediction text + buttons
bottom_frame = tk.Frame(root, bg="#f0f0f0")
bottom_frame.pack(fill="x", padx=20, pady=10)

tk.Label(bottom_frame, text="Recognized Signs:", font=("Helvetica", 12, "bold"),
         bg="#f0f0f0", anchor="w").pack(fill="x", pady=(0, 5))

text_box = tk.Text(bottom_frame, height=3, width=60, font=("Helvetica", 14),
                   bd=2, relief="ridge", bg="#ffffff")
text_box.pack(fill="x", pady=5)

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
        if cap: cap.release()
        cap = None
        video_label.config(image='')

# Buttons
button_style = {"font": ("Helvetica", 12), "width": 10, "bd": 2, "relief": "raised",
                "bg": "#4a7abc", "fg": "black", "padx": 10, "pady": 5}
toggle_btn = tk.Button(button_frame, text="Start", command=toggle_running, **button_style)
toggle_btn.grid(row=0, column=0, padx=15)
tk.Button(button_frame, text="Clear", command=clear_text, **button_style).grid(row=0, column=1, padx=15)
tk.Button(button_frame, text="Copy", command=copy_text, **button_style).grid(row=0, column=2, padx=15)

# Prediction function
def predict_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        xmin = max(int(min(x_coords) * w) - 20, 0)
        ymin = max(int(min(y_coords) * h) - 20, 0)
        xmax = min(int(max(x_coords) * w) + 20, w)
        ymax = min(int(max(y_coords) * h) + 20, h)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        landmark_vector = np.array(
            [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
        ).reshape(1, -1).astype('float32')

        prediction = model.predict(landmark_vector, verbose=0)[0]
        update_confidence_chart(prediction)

        label_index = np.argmax(prediction)
        confidence = prediction[label_index]

        if confidence > 0.4:
            return labels[label_index]

    return None

# Live loop
def video_loop():
    global last_prediction_time, cap, is_flashing, flash_start_time
    if not running or cap is None: return

    ret, frame = cap.read()
    if not ret:
        root.after(10, video_loop)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_frame = cv2.resize(frame, (480, 270))
    current_time = time.time()
    time_since_last = current_time - last_prediction_time
    progress_bar["value"] = min(100, (time_since_last / prediction_interval) * 100)

    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
            h, w, _ = display_frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = max(int(min(x_coords) * w) - 20, 0)
            ymin = max(int(min(y_coords) * h) - 20, 0)
            xmax = min(int(max(x_coords) * w) + 20, w)
            ymax = min(int(max(y_coords) * h) + 20, h)
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        if current_time - last_prediction_time >= prediction_interval:
            is_flashing = True
            flash_start_time = current_time
            last_prediction_time = current_time
            pred_label = predict_frame(frame)
            if pred_label:
                if pred_label == 'del':
                    current_text = text_box.get("1.0", tk.END).rstrip()
                    text_box.delete("1.0", tk.END)
                    text_box.insert(tk.END, current_text[:-1])
                elif pred_label == 'space':
                    text_box.insert(tk.END, " ")
                elif pred_label == 'nothing':
                    pass  # do nothing
                else:
                    text_box.insert(tk.END, pred_label)
                text_box.see(tk.END)

    if is_flashing:
        flash_time_elapsed = current_time - flash_start_time
        if flash_time_elapsed < flash_duration:
            flash_intensity = 1.0 - (flash_time_elapsed / flash_duration)
            flash_frame = cv2.addWeighted(display_frame, 1.0 - (flash_intensity * 0.5),
                                          np.ones_like(display_frame) * 255, flash_intensity * 0.5, 0)
            display_frame = flash_frame
        else:
            is_flashing = False

    display_frame = cv2.copyMakeBorder(display_frame, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, video_loop)

def on_closing():
    global running, cap
    running = False
    if cap: cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()