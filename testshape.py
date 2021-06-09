import numpy as np
import cv2

img = cv2.imread('tmp.PNG')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret , thrash = cv2.threshold(imgGry, 150 , 255, cv2.CHAIN_APPROX_NONE)
_,contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio >= 1.45 and aspectRatio < 1.56:
            cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()