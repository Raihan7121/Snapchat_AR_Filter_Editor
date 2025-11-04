# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 12:55:28 2025

@author: Acer
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
def gaussianFunction(x,y,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(x**2 + y**2) /(2*sigma**2))

def gaussian_Derivative_kernel(sigma=0.5):
    # k = kernel size
    k=3//2
    coords=np.arange(-k,k+1)
    x,y=np.meshgrid(coords,coords)
    gaussValue=gaussianFunction(x, y, sigma)
    k_x = -(x / sigma**2) * gaussValue
    k_y= -(y / sigma**2) * gaussValue

    k_x= k_x/np.sum(np.abs(k_x))
    k_y= k_y/np.sum(np.abs(k_y))
    
    return k_x, k_y

def gaussian_window(sigma=0.6):
    k=3//2
    coords=np.arange(-k,k+1)
    x,y=np.meshgrid(coords,coords)
    gaussValue=gaussianFunction(x, y, sigma)
    gaussValue= gaussValue/np.sum(gaussValue)
    return gaussValue


img=np.array(
    [
        [90, 88, 90, 89, 90, 89, 52, 52, 55, 53, 54, 56, 55],
        [88, 89, 89, 88, 87, 155, 58, 53, 55, 54, 51, 55, 53],
        [88, 92, 86, 87, 154, 155, 154, 51, 54, 52, 54, 52, 52],
        [89, 91, 91, 157, 156, 156, 159, 155, 155, 54, 52, 53, 51],
        [22, 23, 22, 159, 155, 157, 158, 158, 155, 53, 51, 52, 54],
        [22, 22, 23, 159, 155, 157, 158, 157, 157, 56, 55, 53, 52],
        [21, 20, 22, 155, 154, 157, 159, 157, 157, 51, 55, 53, 56],
        [20, 24, 22, 156, 155, 156, 157, 156, 154, 52, 53, 53, 51],
        [21, 25, 23, 155, 156, 154, 160, 154, 155, 56, 55, 55, 51],
        [23, 26, 23, 88, 88, 87, 90, 88, 87, 54, 55, 54, 56],
        [21, 24, 90, 94, 89, 90, 93, 90, 89, 52, 53, 54, 53],
        [23, 88, 87, 89, 88, 91, 89, 88, 89, 53, 52, 52, 55],
        [20, 92, 91, 93, 89, 88, 94, 87, 89, 53, 55, 51, 55]

    ]
    
).astype(np.uint8)

print("Original image shape:", img.shape)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Input Image')
plt.tight_layout()
plt.show()

gx, gy = gaussian_Derivative_kernel(sigma=0.5)
print("Gaussian Derivative Kernel (Gx):\n", gx)
print("Gaussian Derivative Kernel (Gy):\n", gy)


w = gaussian_window(sigma=0.6)

print("Gaussian Window:\n", w)


border_replicate_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

print("Image shape after border replication:", border_replicate_img.shape)
print("Border Replicated Image:\n", border_replicate_img)

plt.imshow(border_replicate_img, cmap='gray', vmin=0, vmax=255)
plt.title('Border Replicated Image')
plt.tight_layout()
plt.show()

#gx=np.flip(gx)
#gy=np.flip(gy)

#print(gx)
#print(gy)

#Gx = cv2.filter2D(border_replicate_img, ddepth=cv2.CV_32F, kernel=gx)
#Gy = cv2.filter2D(border_replicate_img, ddepth=cv2.CV_32F, kernel=gy)

def manual_convolution(image, kernel):
    
    # Get dimensions
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # Flip kernel for true convolution
    kernel = np.flipud(np.fliplr(kernel))

    # Zero-padding
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Output array
    output = np.zeros_like(image, dtype=float)

    # Convolution with rounding at each pixel
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+k_h, j:j+k_w]
            value = np.sum(region * kernel)
            output[i, j] = np.round(value)   # 👈 round to nearest integer

    return output.astype(int)  # convert to int type


Gx=manual_convolution(border_replicate_img, gx)
Gy=manual_convolution(border_replicate_img, gy)


print(Gx.shape)
print(Gy.shape)
Ix=Gx[1:-1, 1:-1]
Iy=Gy[1:-1, 1:-1]

print(Ix.shape)
print(Iy.shape)

print("Gx Image:\n", Ix)
print("Gy Image:\n", Iy)

plt.imshow(Ix, cmap='gray')
plt.title('Gx Image')

plt.figure()
plt.imshow(Iy, cmap='gray')
plt.title('Gy Image')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(Ix, cmap='gray')
plt.title('Ix Image with Values')

# Add text annotations for each cell
for i in range(Ix.shape[0]):
    for j in range(Ix.shape[1]):
        plt.text(j, i, f'{Ix[i, j]:.2f}', 
                ha='center', va='center', 
                color='red', fontsize=8, fontweight='bold')

plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(Iy, cmap='gray')
plt.title('Iy Image with Values')

# Add text annotations for each cell
for i in range(Iy.shape[0]):
    for j in range(Iy.shape[1]):
        plt.text(j, i, f'{Iy[i, j]:.2f}', 
                ha='center', va='center', 
                color='red', fontsize=8, fontweight='bold')

plt.colorbar()
plt.tight_layout()
plt.show()


def apply_smoothing(image, window):
   

    # Zero-pad the image
    image_0_border = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    print("Image shape after zero padding:", image_0_border.shape)

    # Ensure types are float32 for both image and kernel
    src = image_0_border.astype(np.float32)
    k = window.astype(np.float32)

    # Use ddepth = -1 (output same depth as src; here float32)
    smoothed = cv2.filter2D(src, ddepth=-1, kernel=k, borderType=cv2.BORDER_CONSTANT)

    # Remove the padded border
    smoothed = smoothed[1:-1, 1:-1]

    # ✅ Round each pixel value to the nearest integer
    smoothed = np.round(smoothed).astype(int)

    print("Smoothed image shape:", smoothed.shape)
    return smoothed


Ix2=Ix*Ix
Iy2=Iy*Iy
Ixy=Ix*Iy

Ix2_smoothed=apply_smoothing(Ix2, w)
Iy2_smoothed=apply_smoothing(Iy2, w)
Ixy_smoothed=apply_smoothing(Ixy, w)


print("Ix squared Smoothed:\n", Ix2_smoothed)
print("Iy squared Smoothed:\n", Iy2_smoothed)
print("Ixy Smoothed:\n", Ixy_smoothed)
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.imshow(Ix2_smoothed, cmap='gray')
plt.title('Ix^2 Smoothed')

plt.subplot(1, 3, 2)
plt.imshow(Iy2_smoothed, cmap='gray')
plt.title('Iy^2 Smoothed')

plt.subplot(1, 3, 3)
plt.imshow(Ixy_smoothed, cmap='gray')
plt.title('Ixy Smoothed')

plt.tight_layout()
plt.show()

def calc_corner_response(Ix2, Iy2, Ixy, k=0.04):
    import numpy as np
    
    # Harris matrix components
    det_M = (Ix2 * Iy2) - (Ixy ** 2)
    trace_M = Ix2 + Iy2
    
    # Harris corner response
    R = det_M - k * (trace_M ** 2)
    
    # ✅ Round all results to nearest integer
    det_M = np.round(det_M).astype(int)
    trace_M = np.round(trace_M).astype(int)
    R = np.round(R).astype(int)
    
    return det_M, trace_M, R

det_M, trace_M, R = calc_corner_response(Ix2_smoothed, Iy2_smoothed, Ixy_smoothed, k=0.04)
print("Actual Without normalization Corner Response R:\n", R)
print("Determinant of M:\n", det_M)
print("Trace of M:\n", trace_M)

trace_M2=(trace_M ** 2)

def scale_to_255(image):
    """Scale image values to 0-255 range"""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max == img_min:
        return np.zeros_like(image)
    scaled = 255 * (image - img_min) / (img_max - img_min)
    return scaled.astype(np.uint8)
R_scaled = scale_to_255(R)
det_M_scaled = scale_to_255(det_M)
trace_M_scaled = scale_to_255(trace_M)

print("Scaled Corner Response R:\n", R_scaled)
print("Scaled Determinant of M:\n", det_M_scaled)
print("Scaled Trace of M:\n", trace_M_scaled)

plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.imshow(R_scaled, cmap='gray')
plt.title("R Scaled")

plt.subplot(1, 3, 2)
plt.imshow(det_M_scaled, cmap='gray')
plt.title("Det M Scaled")

plt.subplot(1, 3, 3)
plt.imshow(trace_M_scaled, cmap='gray')
plt.title("Trace M Scaled")

plt.tight_layout()
plt.show()
def apply_threshold(response, threshold_factor=0.7):
    """
    Apply threshold to cornerness response
    T = mean(R) + threshold_factor * std(R)
    """
    mean_R = np.mean(response)
    std_R = np.std(response)
    threshold_val =mean_R + threshold_factor * std_R
    
    # Apply threshold: keep values > T, set others to 0
    thresholded = np.where(response > threshold_val, response, 0)

    return thresholded, threshold_val, mean_R, std_R

th_r, th_val, mean_R, std_R = apply_threshold(R_scaled, threshold_factor=0.7)

print("Thresholded R:\n", th_r)
print(f"Threshold Value: {th_val}, Mean R: {mean_R}, Std R: {std_R}")

plt.imshow(th_r, cmap='gray')
plt.title('Thresholded R')
plt.tight_layout()
plt.show()
th_r_scaled = scale_to_255(th_r)

print("Scaled Thresholded R:\n", th_r_scaled)
plt.imshow(th_r_scaled, cmap='gray')
plt.title('Scaled Thresholded R')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

# Overlay corners on original image
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
corner_locations = np.where(th_r_scaled > 0)
plt.scatter(corner_locations[1], corner_locations[0], c='red', s=20, marker='x')
plt.title(f'Detected Corners\n({len(corner_locations[0])} corners found)')
plt.axis('off')

plt.tight_layout()
plt.show()

def non_maximum_suppression_3x3(response):
    """3x3 NMS: keep a pixel only if it is strictly greater than all its 8 neighbors."""
    rows, cols = response.shape
    nms = np.zeros_like(response)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            val = response[i, j]
            if val == 0:
                continue
            neigh = response[i-1:i+2, j-1:j+2]
            # remove center and check max of neighbors
            neighbors = neigh.copy().flatten()
            neighbors = np.delete(neighbors, 4)  # center index 4 in flattened 3x3
            if val > np.max(neighbors):
                nms[i, j] = val
    return nms

# Apply NMS to the thresholded response (use th_r computed previously)
print("Applying 3x3 non-maximum suppression...")
nms_result = non_maximum_suppression_3x3(th_r)

before_count = np.sum(th_r > 0)
after_count = np.sum(nms_result > 0)
suppression_rate = 0.0
if before_count > 0:
    suppression_rate = 100.0 * (before_count - after_count) / before_count

print(f"Corners before NMS: {before_count}")
print(f"Corners after NMS: {after_count}")
print(f"Suppression rate: {suppression_rate:.1f}%")

# Scale for display using existing scale_to_255()
nms_scaled = scale_to_255(nms_result)

# Visualize: thresholded vs NMS vs overlay on original
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(scale_to_255(th_r), cmap='gray', vmin=0, vmax=255)
plt.title('Thresholded R (scaled)')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(nms_scaled, cmap='gray', vmin=0, vmax=255)
plt.title('After 3x3 NMS (scaled)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
ys, xs = np.where(nms_result > 0)
plt.scatter(xs, ys, c='red', s=50, marker='o')
plt.title(f'Final Corners: {after_count}')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print coordinates and response values
if after_count > 0:
    print("Corner coordinates (row, col) and response:")
    for r, c in zip(ys, xs):
        print(f"  ({r}, {c}) -> {nms_result[r, c]:.6f}")
else:
    print("No corners remain after NMS.")
