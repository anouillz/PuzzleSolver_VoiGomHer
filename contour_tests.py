import cv2
import numpy as np

#img = cv2.imread("res/49_pieces_sweater_kevin_small2.jpg")
img = cv2.imread("res/24_pieces_sweater_kevin_small.jpg")
#img = cv2.imread("res/49_pieces_black_ana_small2.jpg")
cv2.imshow("img", img)
#img2 = cv2.convertScaleAbs(img, alpha=1.5, beta=0)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
img2 = cv2.GaussianBlur(img, (5, 5), 4)
#ret, thresh = cv2.threshold(gray, 200, 255, 0)

canny = cv2.Canny(img2, 40, 250)
cv2.imshow("canny", canny)

gray = cv2.GaussianBlur(gray, (7, 7), 2)
grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
grad = cv2.convertScaleAbs(grad, alpha=2.5, beta=0)
cv2.imshow("Sobel", grad)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilated = cv2.dilate(grad, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
cv2.imshow("dilated", dilated)
cv2.imshow("eroded", eroded)

#blur = cv2.GaussianBlur(gray, (7, 7), 0)
blur = cv2.GaussianBlur(grad, (7, 7), 0)
#cv2.imshow("blur", blur)
ret, thresh = cv2.threshold(eroded, 80, 255, cv2.THRESH_BINARY)# + cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
cv2.imshow("thresh", thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 0, 255))#, thickness=cv2.FILLED)

areas = [cv2.contourArea(contour) for contour in contours]

#with_areas = [(contour, cv2.contourArea(contour)) for contour in contours]
#areas = sorted(areas)
q25 = np.percentile(areas, 25)
q75 = np.percentile(areas, 75)
med = np.median(areas)
iqr = q75 - q25
a1 = q25 - 1.5 * iqr
a2 = q75 + 1.5 * iqr
#with_areas = [(area, cont) for area, cont in zip(areas, contours) if a1 <= area <= a2]

a1 = 100
a2 = img.shape[0] * img.shape[1] / 32
with_areas = [(area, cont) for area, cont in zip(areas, contours) if a1 <= area <= a2]

found = 0
for area, contour in with_areas:
    #approx = cv2.approxPolyDP(contour, 0.9 * cv2.arcLength(contour, True), True)
    #cv2.polylines(img, approx, True, (0, 255, 0), 2)

    #approx = cv2.approxPolyN(contour, 4)
    #cv2.polylines(img, approx, True, (0, 255, 0), 2)
    #rect = cv2.boundingRect(contour)
    #cv2.polylines(img, rect, True, (0, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    if max(w, h) / min(w, h) > 2.5:
        continue
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    """
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [box], -1, 255, -1)

    # Apply mask
    res = np.zeros_like(img)
    res[(mask > 0)] = img[(mask > 0)]
    """
    cv2.drawContours(img, [box], 0, (0,255,0), 2)
    found += 1
print(f"Found: {found}")

cv2.imshow("contours", img)
cv2.waitKey(-1)
#cv2.imwrite("res/24_pieces_black_contours.jpg", img)