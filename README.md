# Real-time ASL Recognizer

This project is a real-time American Sign Language (ASL) recognition tool that uses your webcam to interpret hand gestures.

## What it does

This app recognizes ASL alphabet signs (A-Z) plus "space", "delete", and "nothing" gestures in real-time. It uses computer vision to track your hand movements and a trained neural network to interpret the signs.

## Features

- Real-time hand tracking and gesture recognition  
- Clean, intuitive interface with live video feed  
- Built-in ASL reference chart  
- Text output with copy and clear functions

## Getting started

### Requirements

- Python 3.8+
- Webcam
- The packages listed in `requirements.txt`

### Installation

1. Clone this repository:
```
git clone https://github.com/Borbez2/ASL-recognizer.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the ASL Alphabet dataset from Kaggle:
[https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

4. Unzip the downloaded dataset

5. Place the unzipped folder in the project directory

6. Train the model (this will create the required model files):
```python train_model.py```

## Usage

1. Run the application:
```python ASL_recognition.py```
2. Click "Start" to begin webcam capture  
3. Position your hand in the frame and make ASL signs  
4. The app will recognize signs every 3 seconds when your hand is detected  
5. Use the buttons to clear or copy the recognized text

## How it works

The app uses a pre-trained EfficientNetB0 model (trained with `train_model.py`) to classify hand gestures. MediaPipe's hand tracking isolates your hand in the video frame before passing the image to the neural network for classification.

The dataset contains thousands of labeled images representing the ASL alphabet, which are essential for training the model to accurately recognize gestures.

