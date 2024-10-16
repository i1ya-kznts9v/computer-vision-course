import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Image
rgb_image = cv.cvtColor(cv.imread("lahta.jpg"), cv.COLOR_BGR2RGB)
plt.imshow(rgb_image), plt.title('Image'), plt.show()

# Grayscale
gray_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap="gray"), plt.title('Grayscale'), plt.show()

# HSV
hsv_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2HSV)
plt.imshow(hsv_image), plt.title('HSV'), plt.show()

# ORB features
orb = cv.ORB_create()
kp = orb.detect(gray_image, None)
orb_image = cv.drawKeypoints(rgb_image, kp, None, color=(0, 255, 0))
plt.imshow(orb_image), plt.title('ORB features'), plt.show()

# SIFT features
sift = cv.SIFT_create()
kp = sift.detect(gray_image, None)
sift_image = cv.drawKeypoints(rgb_image, kp, None, color=(0, 255, 0))
plt.imshow(sift_image), plt.title('SIFT features'), plt.show()

# Canny edges
canny_image = cv.Canny(gray_image, 180, 120, 60)
plt.imshow(canny_image, cmap="gray"), plt.title('Canny edges'), plt.show()

# Right flip
right_flip_image = cv.flip(rgb_image, 1)
plt.imshow(right_flip_image), plt.title('Right flip'), plt.show()

# Bottom flip
bottom_flip_image = cv.flip(rgb_image, 0)
plt.imshow(bottom_flip_image), plt.title('Bottom flip'), plt.show()


def rotate_around_point(x, y, angle):
    rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1)
    return cv.warpAffine(rgb_image, rotation_matrix, (rgb_image.shape[1], rgb_image.shape[0]))


# Rotate 45' arond center
rotate_45_image = rotate_around_point(
    rgb_image.shape[0] // 2,
    rgb_image.shape[1] // 2,
    -45
)
plt.imshow(rotate_45_image), plt.title('Rotate 45\' arond center'), plt.show()

# Rotate 30' arond point
rotate_45_image = rotate_around_point(
    rgb_image.shape[0] // 2 + 10,
    rgb_image.shape[1] // 2 + 10,
    -30
)
plt.imshow(rotate_45_image), plt.title('Rotate 30\' arond point'), plt.show()

# Shift right 10px
transform_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
shift_right_10px_image = cv.warpAffine(
    rgb_image,
    transform_matrix,
    (rgb_image.shape[1], rgb_image.shape[0])
)
plt.imshow(shift_right_10px_image), plt.title('Shift right 10px'), plt.show()

# Brightness
brightness_image = cv.convertScaleAbs(rgb_image, alpha=1, beta=50)
plt.imshow(brightness_image), plt.title('Brightness'), plt.show()

# Contrast
brightness_image = cv.convertScaleAbs(rgb_image, alpha=0.8, beta=0)
plt.imshow(brightness_image), plt.title('Contrast'), plt.show()

# Gamma-correction
gamma = 0.5
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
gamma_correction_image = cv.LUT(rgb_image, lookUpTable)
plt.imshow(gamma_correction_image), plt.title('Gamma-correction'), plt.show()

# Histogram equalization
histogram_equalization_image = cv.equalizeHist(gray_image)
plt.imshow(histogram_equalization_image, cmap="gray"),
plt.title('Histogram equalization'), plt.show()


def balance_hue(hue_lambda):
    h, s, v = cv.split(hsv_image)
    h = hue_lambda(h)
    h = np.clip(h, 0, 179)
    return cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2RGB)


# Warmer white
warmer_white_image = balance_hue(lambda h: h + 10)
plt.imshow(warmer_white_image), plt.title('Warmer white'), plt.show()

# Colder white
colder_white_image = balance_hue(lambda h: h - 10)
plt.imshow(colder_white_image), plt.title('Colder white'), plt.show()

# Color palette
color_palette = {
    ((0, 0, 0), (255, 128, 64)): (255, 255, 255)
}
color_palette_image = rgb_image.copy()
for key, value in color_palette.items():
    lbound = np.array(key[0])
    ubound = np.array(key[1])
    mask = cv.inRange(color_palette_image, lbound, ubound)
    color_palette_image[mask > 0] = value
plt.imshow(color_palette_image), plt.title('Color palette'), plt.show()

# Binary
_, binary_image = cv.threshold(gray_image, 64, 255, cv.THRESH_BINARY)
plt.imshow(binary_image, cmap='gray'), plt.title('Binary'), plt.show()

# Binary canny edges
binary_canny_image = cv.Canny(binary_image, 180, 120, 60)
plt.imshow(binary_canny_image, cmap="gray"), plt.title('Binary canny edges'), plt.show()

# Laplacian edges
laplacian_image = cv.Laplacian(gray_image, cv.CV_64F, ksize=15)
plt.imshow(laplacian_image, cmap="gray"), plt.title('Laplacian edges'), plt.show()

# Gaussian blur
gaussian_blur_image = cv.GaussianBlur(rgb_image, (5, 5), 0)
plt.imshow(gaussian_blur_image), plt.title('Gaussian blur'), plt.show()


def fourier_transform(fast):
    fourier = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(fourier)
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    radius = 100
    cv.circle(mask, (crow, ccol), radius, 1, -1)
    if not fast:
        mask = 1 - mask
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    fast_fourier_transform_image = np.fft.ifft2(f_ishift)
    fast_fourier_transform_image = np.abs(fast_fourier_transform_image)
    return np.uint8(fast_fourier_transform_image)


# Fast fourier transform
fast_fourier_transform_image = fourier_transform(True)
plt.imshow(fast_fourier_transform_image), plt.title('Fast fourier transform'), plt.show()

# Slow fourier transform
slow_fourier_transform_image = fourier_transform(False)
plt.imshow(slow_fourier_transform_image), plt.title('Slow fourier transform'), plt.show()

# Erosion
erosion_image = cv.erode(laplacian_image, np.ones((3, 3)))
plt.imshow(erosion_image, cmap="gray"), plt.title('Erosion'), plt.show()

# Dilation
dilation_image = cv.dilate(laplacian_image, np.ones((3, 3)))
plt.imshow(dilation_image, cmap="gray"), plt.title('Dilation'), plt.show()
