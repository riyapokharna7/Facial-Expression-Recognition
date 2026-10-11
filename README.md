# Facial-Expression-Recognition
This repository features a real-time facial expression recognition system built with TensorFlow and Keras. It trains a Convolutional Neural Network on 7 emotion classes and uses OpenCV and dlib for live webcam face detection and emotion classification.

**Features**
Live Webcam Integration: Captures real-time video directly within Google Colab using a custom JavaScript interface.   
Robust Face Detection: Utilizes dlib's frontal face detector to accurately locate faces within the video stream.   
Real-Time Classification: Processes cropped face regions—resized to 48x48 grayscale—through the trained CNN to predict expressions instantly.   
Visual Feedback: Overlays bounding boxes, predicted emotion labels, and confidence percentages directly on the live video frame.  

**Tech Stack**
Machine Learning: TensorFlow, Keras   
Computer Vision: OpenCV (cv2), dlib   
Data Processing: NumPy, Keras ImageDataGenerator   
Environment: Google Colab (utilizing IPython display modules)

**DATASET**
The model is trained on a Micro Expressions dataset structured into 7 emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise. The training data is preprocessed into 48x48 pixel grayscale images and rescaled for neural network ingestion.

**Model Architecture**
The custom sequential CNN architecture consists of:
Convolutional Blocks: A 32-filter Conv2D layer followed by a 64-filter Conv2D layer, both using ReLU activation and MaxPooling2D for downsampling.   
Fully Connected Network: A Flatten layer leading to a 128-neuron Dense layer.   
Regularization: A Dropout layer (0.5 rate) to prevent overfitting during training.   
Output: A 7-neuron Dense layer with Softmax activation for multi-class probability distribution.   
Note: The model achieved a baseline test accuracy of 64.62% over 15 epochs and is exported as an HDF5 file 
