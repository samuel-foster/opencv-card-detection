import cv2
import tkinter as tk
from tkinter import filedialog
import torch
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create the GUI
root = tk.Tk()
root.title("Playing Card Identification")

def select_file():
    # Open a file dialog box to select an image file
    file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    # Load the image using OpenCV
    img = cv2.imread(file_path)
    # Identify the card using OpenCV
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to segment the card
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop the image to the bounding rectangle
    card_img = img[y:y+h, x:x+w]
    # Resize the image to a fixed size
    card_img = cv2.resize(card_img, (300, 420))
    # Use YOLOv5 to detect the rank and suit of the card
    results = model(card_img[...,::-1])
    # Parse the results to get the rank and suit
    results = results.xyxy[0].numpy()
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    ranks_idx = np.argmax(results[:, 5:18], axis=1)
    suits_idx = np.argmax(results[:, 18:], axis=1)
    rank = ranks[ranks_idx[0]]
    suit = suits[suits_idx[0]]
    # Display the results on the GUI
    result_label.config(text=f"Rank: {rank}\nSuit: {suit}")

button = tk.Button(root, text="Select Image", command=select_file)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack