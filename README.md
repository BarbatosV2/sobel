# Read me 

The Sobel operator is primarily used for edge detection in image processing and computer vision. It is designed to highlight the edges in an image by emphasizing regions of high spatial frequency where the intensity of the image changes rapidly. Here are the main purposes and applications of the Sobel operator:

**Edge Detection:**
  
The Sobel operator calculates the gradient of the image intensity at each pixel, identifying points where the intensity changes significantly, which correspond to edges.
It produces two images, one highlighting edges in the horizontal direction (Sobel X) and the other in the vertical direction (Sobel Y).

**Image Segmentation:**
  
By detecting edges, the Sobel operator helps in segmenting the image into different regions, which is useful for object recognition and image analysis.

**Feature Extraction:**
  
The detected edges can be used as features for various image processing tasks, such as object detection, recognition, and classification.

**Image Enhancement:**
  
Edge detection is a key step in many image enhancement techniques, where the goal is to improve the visual quality of the image.

# How it works

The Sobel operator uses convolution with two 3x3 kernels (one for detecting horizontal edges and one for vertical edges):

**Sobel X Kernal**

```
-1  0  1
-2  0  2
-1  0  1
```

**Sobel Y Kernal**

```
 1  2  1
 0  0  0
-1 -2 -1
```
        
# How to run
```
python sobel_kernal.py <your_image>
```
# Results 

**Original**

![image](https://github.com/BarbatosV2/sobel/assets/63419320/88abafd0-881d-466c-9969-b12ac7d49d34)

**Sobel X**

![sobel_x](https://github.com/BarbatosV2/sobel/assets/63419320/f2feeb97-0163-42e5-9678-d8f104a30e30)

**Sobel Y**

![sobel_y](https://github.com/BarbatosV2/sobel/assets/63419320/44b5e5ce-4495-4ea1-b278-113830d9424f)

**Sobel Magnitude**

![sobel_magnitude](https://github.com/BarbatosV2/sobel/assets/63419320/d58b4f0e-f2a4-4816-9fba-8ea90696a41c)
