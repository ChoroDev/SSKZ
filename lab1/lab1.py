import numpy as np
import cv2

def loadTransformSave(fileName, thresh):
    img = cv2.imread(fileName + '.jpg')
    height, width, channels = img.shape
    img_binary = np.zeros((height, width, 1))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_binary) = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite(fileName + '_binary.jpg', img_binary)
    cv2.imshow('binary image', img_binary)
    contours, hierarchy =  cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imshow('image with contour', img)
    cv2.imwrite(fileName + '_with_contour.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

loadTransformSave('1', 230)
loadTransformSave('2', 248)
