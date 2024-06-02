import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def apply_sobel_filter(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No image found at {image_path}")
        return
    
    # Apply Sobel filter in the x direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel filter in the y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude of the gradients
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize to range [0, 255]
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    # Save the results
    #cv2.imwrite('sobel_x.jpg', sobel_x)
    #cv2.imwrite('sobel_y.jpg', sobel_y)
    #cv2.imwrite('sobel_magnitude.jpg', sobel_magnitude)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Sobel X')
    plt.imshow(sobel_x, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Sobel Y')
    plt.imshow(sobel_y, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Sobel Magnitude')
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    #print("Sobel images saved as sobel_x.jpg, sobel_y.jpg, and sobel_magnitude.jpg")

if __name__ == "__main__":
    # Check if an image filename was provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python sobel_filter.py <image_filename>")
        sys.exit(1)
    
    image_filename = sys.argv[1]
    apply_sobel_filter(image_filename)
