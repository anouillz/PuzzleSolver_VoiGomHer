from math import cos, radians
from itertools import combinations
import cv2
import numpy as np

from contours import find_piece_contours

ANGLE_THRESHOLD = cos(radians(85))
DIST_THRESHOLD = 30
AREA_THRESHOLD = 0.8
ASPECT_THRESHOLD = 1.2

def normalize_vec(vec):
    return vec / np.linalg.norm(vec)

def find_corners(contour, img):
    approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
    black = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(black, [approx], 0, 255, cv2.FILLED)

    corners = cv2.goodFeaturesToTrack(black, 20,0.01,50)
    corners = np.intp(corners)
    for corner in corners:
        cv2.circle(img, corner.ravel(), 4, (255, 255, 255), -1)
    base_area = cv2.contourArea(contour)
    min_area = base_area * AREA_THRESHOLD

    for a,b,c in combinations(corners, 3):
        a = a.ravel()
        b = b.ravel()
        c = c.ravel()

        ab = b - a
        ac = c - a
        bc = c - b

        nab = normalize_vec(ab)
        nac = normalize_vec(ac)
        nbc = normalize_vec(bc)
        dot_a = abs(np.dot(nab, nac))
        dot_b = abs(np.dot(nab, nbc))
        dot_c = abs(np.dot(nbc, nac))

        min_dot = min(dot_a, dot_b, dot_c)

        if min_dot <= ANGLE_THRESHOLD:
            if dot_a == min_dot:
                d = (b + ac).astype(np.int64)
                w = np.linalg.norm(ab)
                h = np.linalg.norm(ac)
                pts = [a, b, d, c]

            elif dot_b == min_dot:
                d = (a + bc).astype(np.int64)
                w = np.linalg.norm(ab)
                h = np.linalg.norm(bc)
                pts = [b, c, d, a]

            else:
                d = (a - bc).astype(np.int64)
                w = np.linalg.norm(ac)
                h = np.linalg.norm(bc)
                pts = [c, a, d, b]

            area = w * h
            ratio = max(w, h) / min(w, h)

            f = 4
            if area < min_area or ratio > ASPECT_THRESHOLD:
                continue

            for i in corners:
                i = i.ravel()
                if np.linalg.norm(i - d) <= DIST_THRESHOLD:
                    cv2.circle(img, d, DIST_THRESHOLD, (0, 255, 255), 2)
                    cv2.circle(img, i, 8, (0, 0, 255), -1)
                    cv2.circle(img, a, 6, (0, 255, 0), -1)
                    cv2.circle(img, b, 6, (255, 0, 0), -1)
                    cv2.circle(img, c, 6, (255, 255, 0), -1)
                    cv2.drawContours(img, [np.array(pts)], 0, (200, 200, 200), 4)
                    img2 = cv2.resize(img, [img.shape[0] // f, img.shape[1] // f])
                    cv2.imshow("corners", img2)
                    cv2.waitKey(-1)
                    return


if __name__ == '__main__':
    #img = cv2.imread("res/49_pieces_sweater_kevin_small2.jpg")
    img = cv2.imread("res/24_pieces_sweater_kevin_small.jpg")
    pieces = find_piece_contours(img)
    n_pieces = len(pieces)
    print(f"{n_pieces} pieces")
    for i, (contour, box) in enumerate(pieces):
        print(f"Piece {i+1} / {n_pieces}")
        find_corners(contour, img)