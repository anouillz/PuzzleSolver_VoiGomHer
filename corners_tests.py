from cmath import polar
from math import cos, radians
from itertools import combinations
import cv2
import numpy as np
import matplotlib.pyplot as plt

from contours import find_piece_contours

ANGLE_THRESHOLD = 0 # to be determined

def normalize_vec(vec):
    return vec / np.linalg.norm(vec)

def find_corners(contour, img):
    approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
    black = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(black, [approx], 0, 255, cv2.FILLED)
    cv2.imshow("black", black)

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    distances = []
    angles = []

    for c in contour:
        c = c[0]
        dist = np.linalg.norm(c - [cX, cY])
        angle = np.atan2(c[1] - cY, c[0] - cX)
        distances.append(dist)
        angles.append(angle)

    polar_contour = list(zip(angles, distances))

    for i, (angle1, dist1) in enumerate(polar_contour):
        before = polar_contour[(i - 1) % len(polar_contour)]
        after = polar_contour[(i + 1) % len(polar_contour)]

        der_before = (dist1 - before[1]) / ((angle1 - before[0]) % (np.pi * 2))
        der_after = (after[1] - dist1) / ((after[0] - angle1) % (np.pi * 2))

        if der_after * der_before < 0:
            der_tot = der_after / der_before
            if der_tot <= -2:
                # draw point on image
                x = int(cX + dist1 * cos(angle1))
                y = int(cY + dist1 * np.sin(angle1))

                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                cv2.imshow("img", img)
                cv2.waitKey(-1)


    plt.scatter(angles, distances)
    plt.show()



if __name__ == '__main__':
    #img = cv2.imread("res/49_pieces_sweater_kevin_small2.jpg")
    img = cv2.imread("res/p24_1_pull.jpg")
    pieces = find_piece_contours(img)
    n_pieces = len(pieces)
    print(f"{n_pieces} pieces")
    for i, (contour, box) in enumerate(pieces):
        print(f"Piece {i+1} / {n_pieces}")
        find_corners(contour, img)

    cv2.imwrite("res/img_out/p24_1_pull_out_corners.jpg", img)