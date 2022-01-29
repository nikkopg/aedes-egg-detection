import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors

class ImgProcessing:
    def __init__(self):
        self.min_H, self.max_H = 0, 179
        self.min_S, self.max_S = 0, 255
        self.min_V, self.max_V = 0, 50
        self.kernel = np.ones((3,3), np.uint8)

    def scale_img(self, img, scale):
        scale_percent = scale # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        return resized

    def preprocess_img(self, img):
        scaled = self.scale_img(img, 500)
        hsv_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, (self.min_H, self.min_S, self.min_V), (self.max_H, self.max_S, self.max_V))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        return opening

    def canny_edge(self, opening):
        v = np.median(opening)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edge = cv2.Canny(opening, lower, upper)
        return edge

    def get_contour(self, canny_edge):
        cnts, h = cv2.findContours(canny_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return max(cnts, key=cv2.contourArea)

    def merge_contour(self, c1, c2):
        return np.concatenate((np.array(c1).reshape(-1,1,2), np.array(c2).reshape(-1,1,2)))
    
    def extract_efd(self, img):
        cont = img.reshape(-1,2)
        coeffs = elliptic_fourier_descriptors(cont, order=10, normalize='False')
        return coeffs.flatten()