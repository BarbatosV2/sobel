import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def apply_custom_sobel_filter(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No image found at {image_path}")
        return
    
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
    
    # Save the results
    cv2.imwrite('custom_sobel_x.jpg', sobel_x)
    cv2.imwrite('custom_sobel_y.jpg', sobel_y)
    cv2.imwrite('custom_sobel_magnitude.jpg', sobel_magnitude)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Custom Sobel X')
    plt.imshow(sobel_x, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Custom Sobel Y')
    plt.imshow(sobel_y, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Custom Sobel Magnitude')
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Sobel images saved as custom_sobel_x.jpg, custom_sobel_y.jpg, and custom_sobel_magnitude.jpg")

if __name__ == "__main__":
    # Check if an image filename was provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python sobel_filter.py <image_filename>")
        sys.exit(1)
    
    image_filename = sys.argv[1]
    apply_custom_sobel_filter(image_filename)
