import cv2
import numpy as np

img = cv2.imread("res/24_pieces_black_small.jpg")
cv2.imshow("img", img)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
img  = cv2.GaussianBlur(img, (5, 5), 4)
#ret, thresh = cv2.threshold(gray, 200, 255, 0)

canny = cv2.Canny(img, 50, 250)
cv2.imshow("canny", canny)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
#dilate = cv2.dilate(canny, kernel, iterations=1)
#cv2.imshow("dilate", canny)

#blur = cv2.GaussianBlur(gray, (7, 7), 0)
blur = cv2.GaussianBlur(canny, (7, 7), 0)
#cv2.imshow("blur", blur)
#ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#cv2.imshow("thresh", thresh)
contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=cv2.FILLED)

for contour in contours:
    #approx = cv2.approxPolyDP(contour, 0.9 * cv2.arcLength(contour, True), True)
    #cv2.polylines(img, approx, True, (0, 255, 0), 2)

    #approx = cv2.approxPolyN(contour, 4)
    #cv2.polylines(img, approx, True, (0, 255, 0), 2)
    #rect = cv2.boundingRect(contour)
    #cv2.polylines(img, rect, True, (0, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, (0,255,0), 2)

cv2.imshow("contours", img)
cv2.waitKey(-1)
#cv2.imwrite("res/24_pieces_black_contours.jpg", img)