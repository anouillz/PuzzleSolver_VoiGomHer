import cv2
import numpy as np

def find_piece_contours(image):
    """
    Finds the contours of puzzle pieces in the given image
    :param image: the original color image
    :return: a list of tuple containing a piece contour and its bounding
             minAreaRect's box (a rectangular contour)
    """

    # Image → grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gray → blurred
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 2)

    # Blurred → Sobel (edges)
    grad_x = cv2.Sobel(blurred_gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(blurred_gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Sobel → increase contrasts
    sobel = cv2.convertScaleAbs(sobel, alpha=2.5, beta=0)

    # Sobel → dilate → erode ←→ thinner but continuous
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(sobel, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Sobel → threshold (B&W)
    ret, thresh = cv2.threshold(eroded, 80, 255, cv2.THRESH_BINARY)

    # Binary image → find contours (only external)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(contour) for contour in contours]

    # Filter out tiny (probably dust) and very large (not pieces) contours
    a1 = 100
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