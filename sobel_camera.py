import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_custom_sobel_filter(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]])
    
    sobel_y_kernel = np.array([[ 1,  2,  1], 
                               [ 0,  0,  0], 
                               [-1, -2, -1]])
    
    # Apply the Sobel filter using cv2.filter2D
    sobel_x = cv2.filter2D(gray_frame, cv2.CV_64F, sobel_x_kernel)
    sobel_y = cv2.filter2D(gray_frame, cv2.CV_64F, sobel_y_kernel)
    
    # Calculate the magnitude of the gradients
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to range [0, 255]
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    return sobel_x, sobel_y, sobel_magnitude

if __name__ == "__main__":
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Apply the custom Sobel filter to the frame
        sobel_x, sobel_y, sobel_magnitude = apply_custom_sobel_filter(frame)
        
        # Display the resulting frames
        cv2.imshow('Original', frame)
        cv2.imshow('Sobel X', sobel_x)
        cv2.imshow('Sobel Y', sobel_y)
        cv2.imshow('Sobel Magnitude', sobel_magnitude)
        
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
