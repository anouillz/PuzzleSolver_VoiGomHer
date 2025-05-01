import cv2
import numpy as np


def find_pieces_mask(red, green, blue):
    ret, mask_r = cv2.threshold(red[..., 2], 190, 255, cv2.THRESH_BINARY)
    ret, mask_g = cv2.threshold(green[..., 1], 190, 255, cv2.THRESH_BINARY)
    ret, mask_b = cv2.threshold(blue[..., 0], 190, 255, cv2.THRESH_BINARY)
    bg_mask = np.mean([mask_r, mask_g, mask_b], axis=0)
    bg_mask = cv2.GaussianBlur(bg_mask, (7, 7), 5).astype(np.uint8)
    contours, hrcy = cv2.findContours(bg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_i = -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_i = i

    screen_contour = contours[largest_i]
    screen_mask = np.zeros((red.shape[0], red.shape[1]), dtype=np.uint8)
    cv2.drawContours(screen_mask, [screen_contour], 0, (255, 255, 255), cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    screen_mask = cv2.erode(screen_mask, kernel, iterations=5)

    full_mask = cv2.bitwise_and(screen_mask, cv2.bitwise_not(bg_mask))
    ret, full_mask = cv2.threshold(full_mask, 180, 255, cv2.THRESH_BINARY)

    return full_mask

def find_piece_contours(image, red, green, blue):
    """
    Finds the contours of puzzle pieces in the given image
    :param image: the original color image
    :return: a list of tuple containing a piece contour and its bounding
             minAreaRect's box (a rectangular contour)
    """

    red = cv2.GaussianBlur(red, (7, 7), 2)
    green = cv2.GaussianBlur(green, (7, 7), 2)
    blue = cv2.GaussianBlur(blue, (7, 7), 2)

    full_mask = find_pieces_mask(red, green, blue)


    # Binary image → find contours (only external)
    contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(full_mask, contours, -1, (255, 255, 255), cv2.FILLED)
    #cv2.imwrite("/tmp/04_full_mask2.jpg", full_mask)
    #thresh = cv2.erode(thresh, kernel, iterations=1)
    #contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #im2 = image.copy()
    #cv2.drawContours(im2, contours, -1, (0, 0, 255), 1)
    #cv2.imwrite("/tmp/05_contours.jpg", im2)
    areas = [cv2.contourArea(contour) for contour in contours]

    # Filter out tiny (probably dust) and very large (not pieces) contours
    a1 = image.shape[0] * image.shape[1] / 2000
    a2 = image.shape[0] * image.shape[1] / 32
    with_areas = [(area, cont) for area, cont in zip(areas, contours) if a1 <= area <= a2]

    pieces = []
    for area, contour in with_areas:
        # Find rough bounding box to isolate piece
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # Filter out pieces with improbable aspect ratios
        if max(w, h) / min(w, h) > 2.5:
            continue
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        pieces.append((contour, box))

    return pieces

if __name__ == '__main__':
    img = cv2.imread("res/49_pieces_sweater_kevin_small2.jpg")
    pieces = find_piece_contours(img)
    print(len(pieces))