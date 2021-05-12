import numpy as np
import cv2

imageName = 'breathtaking.jpg'
src = cv2.imread(cv2.samples.findFile(imageName))
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, src_thresh = cv2.threshold(src_gray, 100, 200, cv2.THRESH_BINARY)

# Applying Homogeneous blur
cv2.imwrite('homo_blur_' + imageName, cv2.blur(src, (30, 30)))
# Applying Gaussian blur
cv2.imwrite('gauss_blur_' + imageName, cv2.GaussianBlur(src, (27, 27), 0))
# Applying Median blur
cv2.imwrite('median_blur_' + imageName, cv2.medianBlur(src, 25, 0))
# Applying Bilateral Filter
cv2.imwrite('bilateral_blur_' + imageName, cv2.bilateralFilter(src, 18, 18 * 2, 18 / 2))
# Applying 2D Convolution
cv2.imwrite('filter2D_' + imageName, cv2.filter2D(src, -1, np.ones((8, 8), np.float32) / 64))
# Applying 2D Sharpening Filter
cv2.imwrite('filter2D_' + imageName, cv2.filter2D(src, -1, np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])))
# Saving gray image
cv2.imwrite('gray_' + imageName, src_gray)
# Saving image in HSV
cv2.imwrite('HSV_' + imageName, cv2.cvtColor(src, cv2.COLOR_BGR2HSV))
# Threshed
cv2.imwrite('threshed_' + imageName, src_thresh)
# Adaptive Threshold
cv2.imwrite('adapt_thresh_' + imageName, cv2.adaptiveThreshold(src_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
