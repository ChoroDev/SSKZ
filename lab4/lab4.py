import cv2
import numpy as np

srcName = '1.jpg'
srcToSearchIn = '2.png'
srcToSearchFor = '2_front.jpg'
src = cv2.imread(srcName)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

### Harris Corner Detection
gray = np.float32(gray)
res = np.copy(src)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
res[dst > 0.01 * dst.max()] = [0, 255, 0]
cv2.imwrite('corner_Harris_' + srcName, res)
### End

### Shi-Tomasi Corner Detection
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = np.copy(src)
cornersCount = 250
corners = cv2.goodFeaturesToTrack(gray, cornersCount, 0.01, 10)
corners = np.int0(corners)

for i in corners:
  x,y = i.ravel()
  cv2.circle(res, (x,y), 3, (0, 255, 0), -1)
cv2.imwrite('corner_Shi_' + srcName, res)
### End

### SIFT
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
res = cv2.drawKeypoints(src, kp, res)
cv2.imwrite('SIFT_' + srcName, res)
### End

### SURF (non-free, patented)
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# surf = cv2.xfeatures2d.SURF_create(400)
# kp = surf.detect(gray, None)
# res = cv2.drawKeypoints(src, kp, res)
# cv2.imwrite('SURF_' + srcName, res)
### End

### FAST Feature Detection
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(gray, None)
res = cv2.drawKeypoints(src, kp, res)
cv2.imwrite('FAST_' + srcName, res)
### End

### ORB
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# Initiate STAR detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(gray, None)
# compute the descriptors with ORB
kp, res = orb.compute(gray, kp)
# draw only keypoints location,not size and orientation
res = cv2.drawKeypoints(src, kp, res)
cv2.imwrite('ORB_' + srcName, res)
### End

### Brute-Force Matching with ORB Descriptors
img1 = cv2.imread(srcToSearchFor) # queryImage
img2 = cv2.imread(srcToSearchIn) # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, res1 = orb.detectAndCompute(gray1, None)
kp2, res2 = orb.detectAndCompute(gray2, None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors
matches = bf.match(res1, res2)
# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches
res = np.copy(img1)
res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], res, flags=2)
cv2.imwrite('ORB_detector_' + srcName, res)
### End

### Brute-Force Matching with SIFT Descriptors and Ratio Test
img1 = cv2.imread(srcToSearchFor) # queryImage
img2 = cv2.imread(srcToSearchIn) # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, res1 = sift.detectAndCompute(gray1, None)
kp2, res2 = sift.detectAndCompute(gray2, None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(res1, res2, k=2)
# Apply ratio test
good = []
for m, n in matches:
  if m.distance < 0.75 * n.distance:
    good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
res = np.copy(img1)
res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, res, flags=2)
cv2.imwrite('SIFT_detector_' + srcName, res)
### End

### FLANN based Matcher
img1 = cv2.imread(srcToSearchFor) # queryImage
img2 = cv2.imread(srcToSearchIn) # trainImage
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, res1 = sift.detectAndCompute(gray1, None)
kp2, res2 = sift.detectAndCompute(gray2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(res1, res2, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
  if m.distance < 0.7 * n.distance:
    matchesMask[i]=[1, 0]
draw_params = dict(matchColor = (0, 255, 0),
                   singlePointColor = (255, 0, 0),
                   matchesMask = matchesMask,
                   flags = 0)
res = np.copy(img1)
res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, res, **draw_params)
cv2.imwrite('FLANN_detector_' + srcName, res)
### End
