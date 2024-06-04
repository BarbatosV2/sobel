import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def apply_custom_sobel_filter(image, save_dir):
    # Define the Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]])
    
    sobel_y_kernel = np.array([[ 1,  2,  1], 
                               [ 0,  0,  0], 
                               [-1, -2, -1]])
    
    # Apply the Sobel filter using cv2.filter2D
    sobel_x = cv2.filter2D(image, cv2.CV_64F, sobel_x_kernel)
    sobel_y = cv2.filter2D(image, cv2.CV_64F, sobel_y_kernel)
    
    # Calculate the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to range [0, 255]
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    # Construct file paths
    sobel_x_path = os.path.join(save_dir, 'sobel_snap_x.jpg')
    sobel_y_path = os.path.join(save_dir, 'sobel_snap_y.jpg')
    sobel_magnitude_path = os.path.join(save_dir, 'sobel_snap_magnitude.jpg')
    
    # Save the results
    cv2.imwrite(sobel_x_path, sobel_x)
    cv2.imwrite(sobel_y_path, sobel_y)
    cv2.imwrite(sobel_magnitude_path, sobel_magnitude)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Sobel Snap X')
    plt.imshow(sobel_x, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Sobel Snap Y')
    plt.imshow(sobel_y, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Sobel Snap Magnitude')
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Sobel images saved as {sobel_x_path}, {sobel_y_path}, and {sobel_magnitude_path}")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    
    # Capture a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        sys.exit(1)
    
    # Release the webcam
    cap.release()
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply the custom Sobel filter
    apply_custom_sobel_filter(gray_frame, script_dir)
