import cv2
import numpy as np


def template_matching():
    img_rgb = cv2.imread('Stock/d2.jpg')
    img_rgb = cv2.resize(img_rgb, (480, 640))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('Tmpl/t1.jpg', 0)

    # Save template width/height
    w, h = template.shape[::-1]
    # Find matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # Percentage of convergence
    threshold = 0.4

    # Find coordinates
    loc = np.where(res >= threshold)

    # Print rectangle
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    # Show result

    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
