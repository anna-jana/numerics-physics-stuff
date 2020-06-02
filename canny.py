from __future__ import print_function, division

import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# apply convolution kernel
def apply_convolution_kernel(kernel, im):
    n = kernel.shape[0]
    k = n // 2
    idx = np.arange(n) - k
    kernel_c, kernel_r = np.meshgrid(idx, idx)
    kernel_c = kernel_c.ravel()[None, None, :].repeat(im.shape[0], 0).repeat(im.shape[1], 1)
    kernel_r = kernel_r.ravel()[None, None, :].repeat(im.shape[0], 0).repeat(im.shape[1], 1)
    im_rows = np.arange(im.shape[0])
    im_cols = np.arange(im.shape[1])
    im_c, im_r = np.meshgrid(im_cols, im_rows)
    off_rows = (kernel_r + im_r[:, :, None]) % im.shape[0]
    off_cols = (kernel_c + im_c[:, :, None]) % im.shape[1]
    big_kernel = kernel.ravel()[None, None, :].repeat(im.shape[0], 0).repeat(im.shape[1], 1)
    off_pixel = im[off_rows, off_cols]
    res = np.sum(off_pixel * big_kernel, axis=2)
    return res

def apply_convolution_kernel2(kernel, image):
    image_with_zeros = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1), dtype=int)
    image_with_zeros[kernel.shape[0] // 2 : -kernel.shape[0] // 2 + 1, kernel.shape[1] // 2 : -kernel.shape[1] // 2 + 1] = image
    ans = np.zeros(image.shape)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ans += kernel[i, j] * image_with_zeros[i : image.shape[0] + kernel.shape[0] // 2 + i - 1,
                                                   j : image.shape[1] + kernel.shape[1] // 2 + j - 1]
    return ans

def canny(im):
    start = time.time()

    # back and white image
    bw = np.sqrt(im[:,:,0]**2 + im[:,:,1]**2 + im[:,:,2]**2)

    # gaussian blur
    k = 2
    sigma = 1.4
    n = 2*k + 1
    i = np.arange(1, n + 1)
    j = i[:, None]
    d = (i - (k + 1))**2 + (j - (k + 1))**2
    H = 1/(2*np.pi*sigma**2)*np.exp(-d/(2*sigma**2))
    blur = apply_convolution_kernel(H, bw)

    # soebel operator to compute the gradient of the image
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    grad_x = apply_convolution_kernel(sobel_x, blur)

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    grad_y = apply_convolution_kernel(sobel_y, blur)

    grad_mag = np.hypot(grad_x, grad_y)
    grad_angle = np.arctan2(grad_x, grad_y)

    # non maximum suppression:


    # map the [0..2pi] angle to [0..pi] angle by glueing opposed angles togerther
    remaped_angle = np.where(grad_angle > np.pi, np.pi - grad_angle, grad_angle)

    # 0 - 0
    #   - 1 / 8 pi -
    # 45 _ pi/4       # done
    #   - 3 / 8 pi -
    # 90 - pi/2       # done
    #   - 5 / 8 pi -
    # 135 - 3/4pi     # <-
    #   - 7 / 8 pi -

    mapped0 = np.where((remaped_angle < np.pi/8.0) | (remaped_angle >= 7*np.pi/8), 0.0, remaped_angle)
    mapped45 = np.where((mapped0 >= np.pi/8.0) & (mapped0 < 3*np.pi/8), np.pi/4, mapped0)
    mapped90 = np.where((mapped45 >= 3*np.pi/8) & (mapped45 < 5*np.pi/8), np.pi/2, mapped45)
    discrete_angle = np.where(mapped90 >= 5*np.pi/8, 3*np.pi/4, mapped90)

    edges = grad_mag.copy()

    edges[(discrete_angle == 0.0) & ((grad_mag < np.roll(grad_mag, 1, 0)) | (grad_mag < np.roll(grad_mag, -1, 0)))] = 0.0
    edges[(discrete_angle == np.pi/2.0) & ((grad_mag < np.roll(grad_mag, 1, 1)) | (grad_mag < np.roll(grad_mag, -1, 1)))] = 0.0
    edges[(discrete_angle == 3*np.pi/4) & ((grad_mag < np.roll(np.roll(grad_mag, -1, 1), -1, 0)) |
                                           (grad_mag < np.roll(np.roll(grad_mag, 1, 1), 1, 0)))] = 0.0
    edges[(discrete_angle == np.pi/4) & ((grad_mag < np.roll(np.roll(grad_mag, -1, 1), 1, 0)) |
                                     (grad_mag < np.roll(np.roll(grad_mag, 1, 1), -1, 0)))] = 0.0


    # double threshold (hysteresis)
    lower_threshold = 0.20
    upper_threshold = 0.40
    offs = -1,0,1

    is_edge = edges > upper_threshold
    count = is_edge.sum()

    while True:
        neighbors = sum(np.roll(np.roll(is_edge, i, 0), j, 1) for i in offs for j in offs)
        is_new_edge = (neighbors > 0) & (edges > lower_threshold)
        is_edge[is_new_edge] = True

        new_count = is_edge.sum()
        if new_count == count:
            break
        count = new_count

    stop = time.time()
    print("took", stop - start, "seconds")

    return is_edge

if __name__  == "__main__":
    im_path = sys.argv[1]
    image = plt.imread(im_path)
    is_edge = canny(image)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(is_edge, cmap="Greys_r")
    plt.show()
